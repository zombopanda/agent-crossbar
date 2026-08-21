"""ACP Client — async one-shot agent prompt via Agent Client Protocol SDK.

Launches a provider command through ``acp.spawn_agent_process``, initializes
protocol v1, creates a session for *cwd*, optionally sets a model via
``config_option``, sends one text prompt, accumulates assistant text from
``session/update`` notifications, and returns a typed :class:`AcpResult`.

Permission policy:

* ``read_only`` tools → **denied** (selects ``reject_once``).
* ``edit_local`` tools → **allowed** with ``allow_once`` only; ``allow_always``
  is treated as escalating and skipped.

Timeouts and cancellation are supported via ``asyncio.wait_for`` with clean
child-process termination through the context-manager.

The module is a focused abstraction layer; it does NOT integrate with
``server.py``, ``jobs.py``, or the job store.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.schema import (
    AgentMessageChunk,
    AllowedOutcome,
    ClientCapabilities,
    DeniedOutcome,
    PermissionOption,
    RequestPermissionResponse,
    SessionConfigOptionSelect,
    SessionConfigSelectGroup,
    SessionConfigSelectOption,
    ToolCallUpdate,
)

from .models import Autonomy

logger = logging.getLogger(__name__)
ACP_HEARTBEAT_INTERVAL_SEC = 30.0

# ── Public types ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AcpResult:
    """Immutable result of a one-shot ACP prompt."""

    output: str
    stop_reason: str
    session_id: str


class AcpError(Exception):
    """Base exception for all ACP client errors."""


class AcpTimeoutError(AcpError):
    """The ACP prompt exceeded the configured timeout.

    ``stage`` distinguishes a timeout that struck before the prompt was
    ever dispatched to the agent (``"prompt_delivery"``) from one that
    struck while awaiting the agent's response to an already-dispatched
    prompt (``"execution"``, the default) — see ``run_acp_prompt``.
    """

    def __init__(self, message: str, *, stage: str = "execution") -> None:
        super().__init__(message)
        self.stage = stage


class AcpProtocolError(AcpError):
    """The ACP protocol sequence failed — e.g. session not created.

    ``stage`` distinguishes a failure that struck before the prompt was
    ever dispatched to the agent (handshake, session creation, model
    config — ``"prompt_delivery"``) from one that struck while the agent
    was already processing an already-dispatched prompt (``"execution"``,
    the default) — mirrors :class:`AcpTimeoutError`.
    """

    def __init__(self, message: str, *, stage: str = "execution") -> None:
        super().__init__(message)
        self.stage = stage


class AcpProviderUnavailableError(AcpError):
    """The selected provider cannot serve the requested model right now."""

    def __init__(self, code: str, message: str, *, stage: str = "prompt_delivery") -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage


_LIMIT_MARKERS = (
    "429",
    "quota",
    "rate limit",
    "rate-limit",
    "usage limit",
    "limit reached",
    "credits required",
    "insufficient_quota",
    "exhausted",
)
_UNAVAILABLE_MARKERS = (
    "no provider available",
    "provider unavailable",
    "model unavailable",
)


def classify_provider_failure(text: str) -> tuple[str, str] | None:
    """Classify provider stderr/protocol text without returning the raw payload."""
    normalized = text.casefold()
    if any(marker in normalized for marker in _LIMIT_MARKERS):
        return (
            "provider_limit_exhausted",
            "The selected provider has exhausted its quota or rate limit",
        )
    if any(marker in normalized for marker in _UNAVAILABLE_MARKERS):
        return (
            "provider_unavailable",
            "No provider is currently available for the selected model",
        )
    return None


class AcpLaunchError(AcpError):
    """The ACP provider process could not be launched."""


# ── Internal :class:`Client` implementation ─────────────────────────────


class _OneShotClient:
    """Implements the ``acp.Client`` protocol for a single prompt.

    Accumulates ``AgentMessageChunk`` text into a list and selects
    permission options according to the configured autonomy level.
    """

    def __init__(
        self,
        autonomy: Autonomy,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> None:
        self._autonomy = autonomy
        self._on_text_delta = on_text_delta
        self._session_id: str | None = None
        self._output_parts: list[str] = []
        self._stop_reason = "unknown"
        self.prompt_sent = False
        self.last_protocol_activity_at = time.monotonic()

    # -- Client protocol --------------------------------------------------

    def on_connect(self, conn: Any) -> None:
        pass

    async def request_permission(
        self,
        session_id: str,
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        if self._autonomy is Autonomy.EDIT_LOCAL and getattr(tool_call, "kind", None) == "edit":
            return _select_allow_once(options)
        return _select_reject_once(options)

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **kwargs: Any,
    ) -> None:
        # ACP does not expose a provider-neutral "working" state.  Recording
        # protocol activity locally lets the execution heartbeat describe only
        # what this process knows: the one-shot prompt coroutine is alive.
        self.last_protocol_activity_at = time.monotonic()
        if isinstance(update, AgentMessageChunk):
            content = getattr(update, "content", None)
            if content is not None and getattr(content, "type", None) == "text":
                text = getattr(content, "text", "")
                if text:
                    delta = str(text)
                    self._output_parts.append(delta)
                    if self._on_text_delta is not None:
                        try:
                            self._on_text_delta(delta)
                        except Exception:
                            logger.warning("ACP text-delta observer failed", exc_info=True)

    async def write_text_file(self, session_id: str, path: str, content: str, **kwargs: Any) -> Any:
        return None  # Not supported in one-shot mode

    async def read_text_file(
        self,
        session_id: str,
        path: str,
        line: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Any:
        # Return an empty read — one-shot client doesn't serve files
        from acp.schema import ReadTextFileResponse

        return ReadTextFileResponse(content="")

    async def create_terminal(
        self,
        session_id: str,
        command: str,
        args: list[str] | None = None,
        env: list[Any] | None = None,
        cwd: str | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> Any:
        raise AcpProtocolError("Terminal creation is not supported in one-shot mode")

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        raise AcpProtocolError("Terminal output is not supported in one-shot mode")

    async def release_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        return None

    async def wait_for_terminal_exit(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        raise AcpProtocolError("Terminal wait is not supported in one-shot mode")

    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        return None

    async def create_elicitation(self, message: str, mode: Any, **kwargs: Any) -> Any:
        raise AcpProtocolError("Elicitation is not supported in one-shot mode")

    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None:
        pass

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise AcpProtocolError(f"Extension method {method!r} is not supported in one-shot mode")

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        pass


# ── Permission helpers ──────────────────────────────────────────────────


def _select_reject_once(
    options: list[PermissionOption],
) -> RequestPermissionResponse:
    """Select reject_once when offered, otherwise cancel."""
    for opt in options:
        if getattr(opt, "kind", None) == "reject_once":
            return RequestPermissionResponse(
                outcome=AllowedOutcome(option_id=opt.option_id, outcome="selected")
            )
    return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))


def _select_allow_once(
    options: list[PermissionOption],
) -> RequestPermissionResponse:
    """Select the ``allow_once`` (non-escalating) option.

    Deliberately skips ``allow_always`` — that is an escalation.
    Falls back to ``reject_once`` if no ``allow_once`` is present.
    """
    for opt in options:
        if getattr(opt, "kind", None) == "allow_once":
            return RequestPermissionResponse(
                outcome=AllowedOutcome(option_id=opt.option_id, outcome="selected")
            )
    return _select_reject_once(options)


# ── Model config helpers ───────────────────────────────────────────────


def _find_model_config_option(
    config_options: list[Any] | None,
) -> SessionConfigOptionSelect | None:
    """Find the session config option for model selection.

    Prefers `category=='model'` over `id=='model'` as a fallback.
    Returns ``None`` when no matching select option is found.
    """
    if not config_options:
        return None
    # Prefer category == "model"
    for opt in config_options:
        if isinstance(opt, SessionConfigOptionSelect) and getattr(opt, "category", None) == "model":
            return opt
    # Fallback: id == "model"
    for opt in config_options:
        if isinstance(opt, SessionConfigOptionSelect) and getattr(opt, "id", None) == "model":
            return opt
    return None


def _find_effort_config_option(
    config_options: list[Any] | None,
) -> SessionConfigOptionSelect | None:
    """Find the optional reasoning/effort selector advertised by an ACP agent.

    ACP agents may identify this standard semantic selector by category
    ``thought_level``. OpenCode uses the stable ``effort`` id, retained here
    as a compatibility fallback for agents that omit the category.
    """
    if not config_options:
        return None
    for opt in config_options:
        if (
            isinstance(opt, SessionConfigOptionSelect)
            and getattr(opt, "category", None) == "thought_level"
        ):
            return opt
    for opt in config_options:
        if isinstance(opt, SessionConfigOptionSelect) and getattr(opt, "id", None) == "effort":
            return opt
    return None


def _model_value_available(option: SessionConfigOptionSelect, value: str) -> bool:
    """Check if *value* exists among *option*'s flat options or grouped options."""
    for entry in option.options:
        if isinstance(entry, SessionConfigSelectOption) and entry.value == value:
            return True
        if isinstance(entry, SessionConfigSelectGroup):
            for sub in entry.options:
                if sub.value == value:
                    return True
    return False


# ── Public API ──────────────────────────────────────────────────────────


async def run_acp_prompt(
    provider_command: list[str],
    prompt_text: str,
    cwd: str,
    *,
    timeout: float | None = None,
    autonomy: str | Autonomy = Autonomy.READ_ONLY,
    model: str,
    effort: str | None = None,
    startup_timeout: float = 30.0,
    on_process_start: Callable[[int], None] | None = None,
    on_text_delta: Callable[[str], None] | None = None,
    on_execution_heartbeat: Callable[[dict[str, Any]], None] | None = None,
) -> AcpResult:
    """Launch a provider, optionally set model, run one ACP prompt, and return the result.

    Sequence: ``initialize`` → ``session/new`` → ``set_config_option`` for
    model → (optional ``set_config_option`` for effort) → ``session/prompt``.

    During ``session/prompt`` the agent may send ``session/update``
    notifications carrying ``AgentMessageChunk`` — those are accumulated
    into :attr:`AcpResult.output`.  Permission requests are answered
    automatically according to the configured autonomy level.

    The prompt text is NEVER included in any exception message or log
    record — only a byte-length hint is emitted.

    Args:
        provider_command: ``argv`` list for the ACP agent process.
            The first element is the executable; the rest are args.
        prompt_text: Prompt content delivered via
            ``[text_block(prompt_text)]``.
        cwd: Working directory passed to ``session/new``.
        timeout: Optional seconds for the entire operation (including
            launch).  Exceeding this raises :class:`AcpTimeoutError`.
        autonomy: Permission policy for ACP tool calls.
        model: Required model identifier. Looks for a
            ``SessionConfigOptionSelect`` with ``category=="model"`` (or
            ``id=="model"`` as fallback) in the ``NewSessionResponse``
            config options, verifies the model value is available, and
            calls ``set_config_option`` before the prompt.
        effort: Optional reasoning level. When supplied, the agent must
            advertise a compatible thought-level/effort config selector after
            model selection; the value is validated and set before the prompt.
        startup_timeout: Maximum seconds for initialize, session creation,
            and model selection before the job fails as a startup timeout.
        on_process_start: Optional callback receiving the child PID.
        on_text_delta: Optional callback receiving each assistant text chunk.
            Observer failures are logged and never interrupt the provider run.
        on_execution_heartbeat: Optional callback receiving provider-neutral
            liveness data while the ACP subprocess and prompt coroutine remain
            alive.  This does not claim a provider-native working state.

    Returns:
        ``AcpResult`` with ``output``, ``stop_reason``, and ``session_id``.

    Raises:
        AcpTimeoutError: The operation exceeded *timeout*.
        AcpProtocolError: The protocol handshake failed, the requested
            model is unavailable, or ``set_config_option`` failed.
        AcpLaunchError: The provider process could not be started.
    """
    try:
        normalized_autonomy = Autonomy(autonomy)
    except ValueError:
        raise AcpProtocolError(f"Invalid autonomy: {autonomy}", stage="prompt_delivery") from None

    client_impl = _OneShotClient(normalized_autonomy, on_text_delta=on_text_delta)

    async def _run() -> AcpResult:
        try:
            async with spawn_agent_process(
                client_impl,
                provider_command[0],
                *provider_command[1:],
                cwd=cwd,
            ) as (conn, process):
                if on_process_start is not None:
                    on_process_start(process.pid)

                async def _watch_stderr() -> None:
                    stderr = getattr(process, "stderr", None)
                    if stderr is None:
                        await asyncio.Future()
                    while True:
                        line = await stderr.readline()
                        if not line:
                            await asyncio.Future()
                        classified = classify_provider_failure(
                            line.decode("utf-8", errors="replace")
                        )
                        if classified is not None:
                            code, message = classified
                            stage = "execution" if client_impl.prompt_sent else "prompt_delivery"
                            raise AcpProviderUnavailableError(code, message, stage=stage)

                async def _prepare_session() -> str:
                    # 1. initialize
                    init_response = await conn.initialize(
                        protocol_version=PROTOCOL_VERSION,
                        client_capabilities=ClientCapabilities(),
                    )
                    logger.debug(
                        "ACP initialized: protocol_version=%s",
                        getattr(init_response, "protocol_version", None),
                    )

                    # 2. session/new
                    session_response = await conn.new_session(cwd=cwd)
                    session_id: str = session_response.session_id
                    client_impl._session_id = session_id
                    logger.debug("ACP session created: id=%s", session_id)

                    # 2b. required model config
                    config_options: list[Any] | None = getattr(
                        session_response, "config_options", None
                    )
                    model_option = _find_model_config_option(config_options)
                    if model_option is None:
                        raise AcpProtocolError(
                            "No model config option available from agent",
                            stage="prompt_delivery",
                        )
                    if not _model_value_available(model_option, model):
                        raise AcpProtocolError(
                            f"Requested model {model!r} not available from agent",
                            stage="prompt_delivery",
                        )
                    try:
                        set_response = await conn.set_config_option(
                            config_id=model_option.id,
                            session_id=session_id,
                            value=model,
                        )
                    except Exception as exc:
                        raise AcpProtocolError(
                            "Failed to set model config option", stage="prompt_delivery"
                        ) from exc

                    # Validate that the agent accepted the model value. Model
                    # selection can change the available effort variants, so
                    # subsequent selector discovery uses this response.
                    response_options: list[Any] | None = getattr(
                        set_response, "config_options", None
                    )
                    response_model_option = _find_model_config_option(response_options)
                    if (
                        response_model_option is None
                        or response_model_option.current_value != model
                    ):
                        raise AcpProtocolError(
                            f"Agent rejected model {model!r}: the config option was not applied",
                            stage="prompt_delivery",
                        )

                    if effort is not None:
                        effort_option = _find_effort_config_option(response_options)
                        if effort_option is None:
                            raise AcpProtocolError(
                                "No effort config option available from agent",
                                stage="prompt_delivery",
                            )
                        if not _model_value_available(effort_option, effort):
                            raise AcpProtocolError(
                                f"Requested effort {effort!r} not available from agent",
                                stage="prompt_delivery",
                            )
                        try:
                            effort_response = await conn.set_config_option(
                                config_id=effort_option.id,
                                session_id=session_id,
                                value=effort,
                            )
                        except Exception as exc:
                            raise AcpProtocolError(
                                "Failed to set effort config option", stage="prompt_delivery"
                            ) from exc

                        effort_response_options: list[Any] | None = getattr(
                            effort_response, "config_options", None
                        )
                        response_effort_option = _find_effort_config_option(effort_response_options)
                        if (
                            response_effort_option is None
                            or response_effort_option.current_value != effort
                        ):
                            raise AcpProtocolError(
                                f"Agent rejected effort {effort!r}: the config option was not applied",
                                stage="prompt_delivery",
                            )
                    return session_id

                stderr_task = asyncio.create_task(_watch_stderr())
                prompt_task: asyncio.Task[Any] | None = None
                process_watch_task: asyncio.Task[Any] | None = None
                heartbeat_stop = threading.Event()
                heartbeat_thread: threading.Thread | None = None
                try:
                    prepare_task = asyncio.create_task(_prepare_session())
                    done, _pending = await asyncio.wait(
                        {prepare_task, stderr_task},
                        timeout=startup_timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done:
                        prepare_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await prepare_task
                        raise AcpTimeoutError(
                            f"ACP startup timed out after {startup_timeout:.1f}s",
                            stage="prompt_delivery",
                        )
                    if stderr_task in done:
                        prepare_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await prepare_task
                        await stderr_task
                    session_id = await prepare_task

                    # 3. session/prompt
                    client_impl.prompt_sent = True
                    prompt_started_at = time.monotonic()
                    client_impl.last_protocol_activity_at = prompt_started_at
                    prompt_task = asyncio.create_task(
                        conn.prompt(
                            session_id=session_id,
                            prompt=[text_block(prompt_text)],
                        )
                    )

                    def _execution_heartbeat_loop() -> None:
                        """Report ACP liveness outside the provider event loop.

                        Some ACP SDK/provider paths synchronously block the
                        asyncio loop while handling a tool call.  A thread is
                        intentional here: the heartbeat reports only the
                        subprocess/prompt lifecycle owned by this wrapper and
                        never a provider-native working state.
                        """
                        while not heartbeat_stop.wait(ACP_HEARTBEAT_INTERVAL_SEC):
                            process_returncode = getattr(process, "returncode", None)
                            payload = {
                                "process_alive": process_returncode is None,
                                "prompt_active": True,
                                "elapsed_sec": int(time.monotonic() - prompt_started_at),
                                "last_protocol_activity_sec": int(
                                    time.monotonic() - client_impl.last_protocol_activity_at
                                ),
                            }
                            if on_execution_heartbeat is not None:
                                try:
                                    on_execution_heartbeat(payload)
                                except Exception:
                                    logger.warning(
                                        "ACP execution-heartbeat observer failed",
                                        exc_info=True,
                                    )

                    if on_execution_heartbeat is not None:
                        heartbeat_thread = threading.Thread(
                            target=_execution_heartbeat_loop,
                            name="acp-execution-heartbeat",
                            daemon=False,
                        )
                        heartbeat_thread.start()
                    process_wait = getattr(process, "wait", None)
                    if callable(process_wait):

                        async def _watch_process_exit() -> None:
                            exit_code = await process_wait()
                            if prompt_task is not None and not prompt_task.done():
                                raise AcpProtocolError(
                                    f"ACP process exited with code {exit_code}",
                                    stage="execution",
                                )

                        process_watch_task = asyncio.create_task(_watch_process_exit())
                    wait_tasks: set[asyncio.Task[Any]] = {prompt_task, stderr_task}
                    if process_watch_task is not None:
                        wait_tasks.add(process_watch_task)
                    done, _pending = await asyncio.wait(
                        wait_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if stderr_task in done:
                        await stderr_task
                    if process_watch_task is not None and process_watch_task in done:
                        await process_watch_task
                    prompt_response = await prompt_task
                finally:
                    if prompt_task is not None and not prompt_task.done():
                        prompt_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await prompt_task
                    heartbeat_stop.set()
                    if heartbeat_thread is not None:
                        heartbeat_thread.join()
                    if process_watch_task is not None and not process_watch_task.done():
                        process_watch_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await process_watch_task
                    stderr_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await stderr_task
                stop_reason = getattr(prompt_response, "stop_reason", None) or "unknown"
                client_impl._stop_reason = stop_reason
                logger.debug(
                    "ACP prompt finished: stop_reason=%s prompt_bytes=%d",
                    stop_reason,
                    len(prompt_text.encode("utf-8")),
                )

                output = (
                    "".join(client_impl._output_parts)
                    if client_impl._output_parts
                    else "(no output)"
                )

                return AcpResult(
                    output=output,
                    stop_reason=stop_reason,
                    session_id=session_id,
                )
        except FileNotFoundError as exc:
            raise AcpLaunchError(f"Provider binary not found: {provider_command[0]}") from exc
        except AcpError:
            raise
        except Exception as exc:
            classified = classify_provider_failure(str(exc))
            if classified is not None:
                code, message = classified
                stage = "execution" if client_impl.prompt_sent else "prompt_delivery"
                raise AcpProviderUnavailableError(code, message, stage=stage) from exc
            stage = "execution" if client_impl.prompt_sent else "prompt_delivery"
            raise AcpProtocolError(f"ACP protocol sequence failed: {exc}", stage=stage) from exc

    try:
        if timeout is not None:
            return await asyncio.wait_for(_run(), timeout=timeout)
        return await _run()
    except asyncio.TimeoutError:
        stage = "execution" if client_impl.prompt_sent else "prompt_delivery"
        raise AcpTimeoutError(f"ACP prompt timed out after {timeout:.1f}s", stage=stage) from None
