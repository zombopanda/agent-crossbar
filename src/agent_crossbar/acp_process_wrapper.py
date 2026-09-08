"""Launch an ACP provider in an owned POSIX process group.

The ACP SDK intentionally exposes no ``start_new_session`` transport option.
This tiny repository-owned wrapper provides the missing fencing without
patching the installed SDK: it creates a new session/process group and then
execs the provider, preserving the provider's stdio protocol byte-for-byte.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("acp_process_wrapper requires a provider executable")
    if hasattr(os, "setsid"):
        os.setsid()
    command = sys.argv[1]
    os.execvp(command, sys.argv[1:])


if __name__ == "__main__":
    main()
