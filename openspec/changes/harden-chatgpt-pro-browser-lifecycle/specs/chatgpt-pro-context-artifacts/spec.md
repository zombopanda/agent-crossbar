## ADDED Requirements

### Requirement: Context preparation is deterministic and bounded

The system SHALL use only explicitly supplied generic `cwd`/`scope` context, walk paths deterministically, skip symlinks and generated directories, enforce total/file/chunk limits, and return a compact summary of included and omitted content.

#### Scenario: A bounded repository scope is requested

- **WHEN** `cwd` and a valid generic scope identify a repository context
- **THEN** the packer includes deterministic, budgeted text chunks and reports files scanned, included, omitted, and used characters

#### Scenario: An explicit context path is missing

- **WHEN** a requested context path or attachment does not exist
- **THEN** preparation fails before prompt submission with a stable validation error

#### Scenario: A symlink points outside the allowed context

- **WHEN** a traversal encounters a symlinked file or directory
- **THEN** the packer skips it and never includes the target contents

### Requirement: Context and attachments do not silently exceed safety budgets

The packer SHALL omit or reject content that exceeds configured file, chunk, total, binary, or attachment budgets and SHALL report omission reasons without including sensitive content in diagnostics.

#### Scenario: A large text file exceeds the inline budget

- **WHEN** a file is larger than the configured context budget
- **THEN** the packer includes only permitted chunks and records the omission/chunking reason

#### Scenario: A binary attachment is within upload limits

- **WHEN** the browser surface supports upload and a binary file is explicit or safely auto-attached within size limits
- **THEN** the runner attaches it and verifies that ChatGPT accepted it before Send

### Requirement: GUI diagnostics are secure and job-local

All ChatGPT Pro diagnostics and artifacts SHALL be written under the owning job's artifact directory, SHALL enforce realpath containment, SHALL reject symlinked or hardlinked artifact targets, and SHALL redact clipboard contents, credentials, raw context envelopes, and hidden reasoning.

#### Scenario: A failure captures browser evidence

- **WHEN** a GUI turn fails with an AX/page diagnostic
- **THEN** the evidence is registered as a job-local artifact with stage, kind, and bounded metadata and is returned through the existing job result artifact list

#### Scenario: An unsafe artifact path is supplied

- **WHEN** artifact metadata resolves outside the job directory or through a symlink/hardlink
- **THEN** the artifact is rejected or replaced with a safe job-local fallback and no outside file is read or overwritten

#### Scenario: A diagnostic contains sensitive transport data

- **WHEN** a UI snapshot or error string contains a context envelope, token-like value, clipboard content, or hidden reasoning marker
- **THEN** the persisted diagnostic contains only its redacted placeholder and bounded structural metadata
