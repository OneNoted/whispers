# CLI guide

Whispers intentionally has a small command surface. Run `whispers --help` for
the exact installed version.

| Command | Purpose |
| --- | --- |
| `whispers` or `whispers toggle` | Start listening, or finish the current dictation |
| `whispers cancel` | Cancel listening, finalization, or pending delivery |
| `whispers status` | Print `idle`, `listening`, `finalizing`, or `delivering` |
| `whispers setup` | Migrate config and install pinned runtimes/models |
| `whispers doctor` | Check config, runtime files, dynamic loading, and desktop tools |
| `whispers vocabulary list` | Show canonical terms and spoken aliases |
| `whispers vocabulary add` | Add aliases for a canonical term |
| `whispers vocabulary remove` | Remove a canonical term |

## Vocabulary

Vocabulary is one shared input to recognition and deterministic correction:

```sh
whispers vocabulary add serde_json \
  --alias "surdy json" \
  --alias "serde json"
whispers vocabulary list
whispers vocabulary remove serde_json
```

The store is `${XDG_DATA_HOME:-~/.local/share}/whispers/vocabulary.toml` and is
written atomically.

## Spoken structure and symbols

Common symbols are deterministic when spoken between terms:

```text
this slash that       → this/that
foo underscore bar    → foo_bar
first new line second → first\nsecond
```

Use `literal` to force a symbol by itself (`literal slash`). Lexical uses such as
“the word slash” are kept as words.

## Setup

```sh
whispers setup [--runtime-pack DIR] [--no-service]
```

Network setup verifies SHA-256 before atomically installing each asset. An
offline pack is copied without following symbolic links or special files.
