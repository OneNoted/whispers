# CLI guide

For the full generated help, run `whispers --help` or `whispers <command> --help`.

## Top-level commands

| Command | Purpose |
| --- | --- |
| `whispers` | Start or stop the default dictation loop |
| `whispers setup` | Guided setup for local, cloud, and experimental dictation paths |
| `whispers transcribe <file>` | Transcribe an audio file to stdout or a file |
| `whispers model` | Legacy `whisper_cpp` model commands |
| `whispers asr-model` | Manage ASR models across recommended and experimental backends |
| `whispers rewrite-model` | Manage local rewrite models |
| `whispers dictionary` | Manage deterministic replacements |
| `whispers app-rule` | Manage app-aware rewrite policy rules |
| `whispers glossary` | Manage technical glossary entries |
| `whispers cloud check` | Validate cloud config and connectivity |
| `whispers snippets` | Manage spoken snippets |
| `whispers rewrite-instructions-path` | Print the custom rewrite instructions file path |
| `whispers completions [shell]` | Print shell completions to stdout |

## Common flows

### Setup and dictation

```sh
whispers setup
whispers
```

### File transcription

```sh
whispers transcribe audio.wav
whispers transcribe meeting.m4a --output transcript.txt
whispers transcribe audio.wav --raw
```

### ASR model management

```sh
whispers asr-model list
whispers asr-model download large-v3-turbo
whispers asr-model select large-v3-turbo
```

`whispers model` still exists for legacy `whisper_cpp` flows, but `whispers asr-model` is the preferred interface.

### Rewrite model management

```sh
whispers rewrite-model list
whispers rewrite-model download qwen-3.5-4b-q4_k_m
whispers rewrite-model select qwen-3.5-4b-q4_k_m
```

### Personalization

```sh
whispers dictionary add "wisper flow" "Wispr Flow"
whispers snippets add signature "Best regards,\nNotes"
```

### Rewrite policy files

```sh
whispers app-rule path
whispers glossary path
whispers rewrite-instructions-path
```

### App-aware rewrite rules

```sh
whispers app-rule list
whispers app-rule add jira-browser \
  "Prefer issue IDs and preserve markdown punctuation" \
  --surface-kind browser \
  --browser-domain-contains atlassian.net
whispers app-rule remove jira-browser
```

### Technical glossary entries

```sh
whispers glossary list
whispers glossary add serde_json \
  --alias "surdy json" \
  --alias "serdy json" \
  --surface-kind editor
whispers glossary remove serde_json
```

### Cloud and shell integration

```sh
whispers cloud check
whispers completions zsh
```
