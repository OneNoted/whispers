# whispers

Fast speech-to-text dictation for Wayland with local-first ASR and optional cloud ASR/rewrite backends.
Press a key to start recording, press it again to transcribe and paste.

Local mode keeps all inference on your machine. Optional cloud modes can offload ASR, rewrite, or both when configured.

Inspired by [hyprwhspr](https://github.com/goodroot/hyprwhspr) by goodroot.

<img width="295" height="145" alt="image" src="https://github.com/user-attachments/assets/9005ad84-e09f-476a-ad2f-4aec8d6ad7ef" />



## How it works

1. Bind `whispers` to a key in your compositor
2. First press starts recording (OSD overlay shows audio visualization)
3. Second press stops recording, transcribes, and pastes via `Ctrl+Shift+V`

The two invocations communicate via PID file + `SIGUSR1` — no daemon, no IPC server.

## Post-processing modes

`whispers` now has three main dictation modes:

- `raw` keeps output close to the direct transcription result and is the default
- `advanced_local` enables the smart rewrite pipeline after transcription; `[rewrite].backend` chooses whether that rewrite runs locally or in the cloud
- `agentic_rewrite` uses the same local/cloud rewrite backends as `advanced_local`, but adds app-aware policy rules, technical glossary guidance, and a stricter conservative acceptance guard

The older heuristic cleanup path is still available as deprecated `legacy_basic` for existing configs that already use `[cleanup]`.
The local rewrite path is managed by `whispers` itself through an internal helper binary installed alongside the main executable, so there is no separate tool or daemon to install manually.
When a rewrite mode is enabled with `rewrite.backend = "local"`, `whispers` keeps a hidden rewrite worker warm for a short idle window so repeated dictation is much faster without becoming a permanent background daemon.
Managed rewrite models are the default path. If you point `rewrite.model_path` at your own GGUF, it should be a chat-capable model with an embedded template that `llama.cpp` can apply at runtime.
Deterministic personalization rules apply in all modes: dictionary replacements and spoken snippets. Custom rewrite instructions apply to both rewrite modes, and `agentic_rewrite` can additionally load app rules and glossary entries from separate TOML files.
Cloud ASR and cloud rewrite are both optional. Local remains the default.

For file transcription, `whispers transcribe --raw <file>` always prints the plain ASR transcript without any post-processing.

## Requirements

- Rust 1.85+ (edition 2024)
- Linux with Wayland compositor
- `wl-copy` (from `wl-clipboard`)
- `uinput` access (for virtual keyboard paste)
- NVIDIA GPU + CUDA toolkit (optional, for GPU acceleration)
- `python3` on `PATH` if you want to use the optional `faster-whisper` backend
- `python3.10`, `python3.11`, or `python3.12` on `PATH` if you want to use the experimental NeMo backends
- If no compatible GPU is available, set `transcription.use_gpu = false` in config

## Install

### From crates.io

```sh
# Default install: CPU build with Wayland OSD
cargo install whispers

# Enable CUDA acceleration explicitly
cargo install whispers --features cuda

# Build without the OSD overlay
cargo install whispers --no-default-features
```

### From git

```sh
# Default install: CPU build with Wayland OSD
cargo install --git https://github.com/OneNoted/whispers

# Enable CUDA acceleration explicitly
cargo install --git https://github.com/OneNoted/whispers --features cuda

# Build without the OSD overlay
cargo install --git https://github.com/OneNoted/whispers --no-default-features
```

### Setup

Run the interactive setup wizard to download a local ASR model, generate config, and optionally enable local or cloud advanced dictation. Recommended local models are shown first, and experimental backends like Parakeet are called out explicitly before you opt into them:

```sh
whispers setup
```

Normal runs keep output concise. Add `-v` when you want detailed diagnostic logs during setup, downloads, or dictation.

Use a custom config file for any command (including `setup` and `asr-model`):

```sh
whispers --config /path/to/config.toml setup
whispers --config /path/to/config.toml asr-model select tiny
```

Or manage ASR models manually:

```sh
whispers asr-model list
whispers asr-model download large-v3-turbo
whispers asr-model select large-v3-turbo
whispers asr-model download distil-large-v3.5
whispers asr-model select distil-large-v3.5
# Experimental NeMo path:
whispers asr-model download parakeet-tdt_ctc-1.1b
whispers asr-model select parakeet-tdt_ctc-1.1b

# Legacy whisper_cpp-only aliases still work for one release:
whispers model list
whispers model download large-v3-turbo
whispers model select large-v3-turbo

whispers rewrite-model list
whispers rewrite-model download qwen-3.5-4b-q4_k_m
whispers rewrite-model select qwen-3.5-4b-q4_k_m
whispers cloud check

whispers dictionary add "wisper flow" "Wispr Flow"
whispers dictionary list
whispers snippets add signature "Best regards,\nNotes"
whispers snippets list
whispers rewrite-instructions-path
whispers app-rule path
whispers app-rule add zed-rust "Preserve Rust identifiers." --app-id dev.zed.Zed --correction-policy balanced
whispers glossary path
whispers glossary add TypeScript --alias "type script" --surface-kind editor
```

That still remains a single install: `whispers` manages local ASR models, the optional local rewrite worker/model, and the optional cloud configuration from the same package. `faster-whisper` is bootstrapped into a hidden managed runtime when you download or prewarm that backend.

## Shell completions

Print completion scripts to `stdout`:

```sh
# auto-detect from $SHELL (falls back to parent process name)
whispers completions

# or specify manually
whispers completions zsh
```

Supported shells: `bash`, `zsh`, `fish`, `nushell`.

Example install paths:

```sh
# bash
mkdir -p ~/.local/share/bash-completion/completions
whispers completions bash > ~/.local/share/bash-completion/completions/whispers

# zsh
mkdir -p ~/.zfunc
whispers completions zsh > ~/.zfunc/_whispers

# fish
mkdir -p ~/.config/fish/completions
whispers completions fish > ~/.config/fish/completions/whispers.fish

# nushell
mkdir -p ~/.config/nushell/completions
whispers completions nushell > ~/.config/nushell/completions/whispers.nu
```

## Compositor keybinding

### Hyprland

```conf
bind = SUPER ALT, D, exec, whispers
```

### Sway

```conf
bindsym $mod+Alt+d exec whispers
```

## Configuration

Config lives at `~/.config/whispers/config.toml` by default. Generated automatically by `whispers setup`, or copy from `config.example.toml`:

```toml
[audio]
device = ""            # empty = system default
sample_rate = 16000

[transcription]
backend = "whisper_cpp"  # or "faster_whisper" / "nemo" / "cloud"
fallback = "configured_local"  # or "none"
local_backend = "whisper_cpp"
selected_model = "large-v3-turbo"
model_path = "~/.local/share/whispers/ggml-large-v3-turbo.bin"
language = "auto"      # or "en", "fr", "de", etc.
use_gpu = true         # set false to force CPU
flash_attn = true      # only used when use_gpu=true
idle_timeout_ms = 120000

[postprocess]
mode = "raw"           # or "advanced_local" / "agentic_rewrite"; deprecated: "legacy_basic"

[session]
enabled = true
max_entries = 3
max_age_ms = 8000
max_replace_graphemes = 400

[personalization]
dictionary_path = "~/.local/share/whispers/dictionary.toml"
snippets_path = "~/.local/share/whispers/snippets.toml"
snippet_trigger = "insert"

[rewrite]
backend = "local"      # or "cloud"
fallback = "local"     # or "none"
selected_model = "qwen-3.5-4b-q4_k_m"
model_path = ""        # optional manual GGUF path override
instructions_path = "~/.local/share/whispers/rewrite-instructions.txt"
profile = "auto"       # or "qwen", "generic", "llama_compat"
timeout_ms = 30000
idle_timeout_ms = 120000
max_output_chars = 1200
max_tokens = 256

[agentic_rewrite]
policy_path = "~/.local/share/whispers/app-rewrite-policy.toml"
glossary_path = "~/.local/share/whispers/technical-glossary.toml"
default_correction_policy = "balanced"

[cloud]
provider = "openai"    # or "openai_compatible"
base_url = ""          # required for openai_compatible
api_key = ""           # optional direct API key; leave empty to use api_key_env
api_key_env = "OPENAI_API_KEY"
connect_timeout_ms = 3000
request_timeout_ms = 15000

[cloud.transcription]
model = "gpt-4o-mini-transcribe"
language_mode = "inherit_local"  # or "force"
language = ""

[cloud.rewrite]
model = "gpt-4.1-mini"
temperature = 0.1
max_output_tokens = 256

[feedback]
enabled = true
start_sound = ""       # empty = bundled sound
stop_sound = ""
```

When `advanced_local` or `agentic_rewrite` is enabled, `whispers` also keeps a short-lived local session ledger in the runtime directory so immediate follow-up corrections like `scratch that` can safely replace the most recent dictation entry when focus has not changed. That session behavior is local either way; only the semantic rewrite stage may be cloud-backed.

## Cloud Modes

- `transcription.backend = "cloud"` uploads recorded audio to the configured provider for ASR.
- `rewrite.backend = "cloud"` uploads transcript/context JSON to the configured provider for semantic cleanup.
- `transcription.fallback = "configured_local"` keeps a local ASR fallback path.
- `rewrite.fallback = "local"` keeps a local rewrite fallback path.
- Use either `cloud.api_key_env` or `cloud.api_key`. `setup` accepts either an env var name or a pasted key.

Use `whispers cloud check` to validate cloud config, API key resolution, and basic provider connectivity.

## Managed ASR models

`whispers` currently ships managed local ASR entries across two backend families:

| Model | Backend | Scope | Notes |
|-------|---------|-------|-------|
| large-v3-turbo | whisper_cpp | Multilingual | Default path |
| large-v3 | whisper_cpp | Multilingual | Slower, higher accuracy |
| medium / small / base / tiny | whisper_cpp | Multilingual | Smaller/faster tradeoffs |
| *.en variants | whisper_cpp | English only | Smaller English Whisper options |
| distil-large-v3.5 | faster_whisper | English only | Fast English option |
| parakeet-tdt_ctc-1.1b | nemo | English only | Experimental NeMo ASR benchmark path |
| canary-qwen-2.5b | nemo | English only | Experimental NeMo ASR/LLM hybrid (currently blocked) |

`large-v3-turbo` remains the default multilingual local model. `distil-large-v3.5` is the speed-focused English option on the optional `faster-whisper` backend. `parakeet-tdt_ctc-1.1b` is kept as an experimental English-only NeMo backend for benchmarking against Whisper-family models, not as the default recommendation. Its first warm-up can be much slower than steady-state dictation, so judge it on warm use rather than the first cold start. `canary-qwen-2.5b` remains listed for evaluation, but the managed path is currently blocked by an upstream NeMo/PEFT initialization incompatibility. Cloud ASR models are configured under `[cloud.transcription]` instead of being downloaded locally.

## Whisper Models

| Model | Size | Speed | Notes |
|-------|------|-------|-------|
| large-v3-turbo | 1.6 GB | Fast | Best balance (recommended) |
| large-v3-turbo-q5_0 | 574 MB | Fast | Quantized, slightly less accurate |
| large-v3 | 3.1 GB | Slow | Most accurate |
| small / small.en | 488 MB | Very fast | Good for English-only |
| tiny / tiny.en | 78 MB | Instant | Least accurate |

Whisper.cpp models are downloaded from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp) and stored in `~/.local/share/whispers/`. The managed `faster-whisper` backend stores models and its Python runtime under the same XDG data directory.

## Managed rewrite models

When `rewrite.backend = "local"`, both rewrite modes use a second local model for post-processing. The managed local catalog currently includes:

| Model | Size | Notes |
|-------|------|-------|
| qwen-3.5-2b-q4_k_m | ~1.3 GB | Fallback for weaker hardware |
| qwen-3.5-4b-q4_k_m | ~2.9 GB | Recommended default |
| qwen-3.5-9b-q4_k_m | ~5.9 GB | Higher quality, heavier |

If you want to tinker, set `rewrite.model_path` to a custom GGUF file. When `rewrite.model_path` is set, it overrides the managed selection.
`rewrite.profile = "auto"` keeps the prompt/runtime model-aware without requiring manual tuning for managed models, and still falls back safely for custom GGUFs.
Custom rewrite models should include a chat template that `llama.cpp` can read from the GGUF metadata; otherwise rewrite prompting will fail fast instead of silently producing bad output.

## Personalization

Dictionary replacements apply deterministically in `raw`, `advanced_local`, and `agentic_rewrite`, with normalization for case and punctuation but no fuzzy matching. In the rewrite modes, dictionary replacements are applied before the rewrite model and again on the final output so exact names and product terms stay stable.

Spoken snippets also work in all modes. By default, saying `insert <snippet name>` expands the configured snippet text verbatim after post-processing finishes, so the rewrite model cannot paraphrase it. Change the trigger phrase with `personalization.snippet_trigger`.

Custom rewrite instructions live in a separate plain-text file referenced by `rewrite.instructions_path`. `whispers` appends that file to the built-in rewrite prompt for both rewrite modes while still enforcing the same final-text-only output contract. The file is optional, and a missing file is ignored.

`agentic_rewrite` also reads layered app rules from `agentic_rewrite.policy_path` and scoped glossary entries from `agentic_rewrite.glossary_path`. `whispers setup` creates commented starter files for both when you choose the agentic mode, and the minimal CRUD commands above are available for path/list/add/remove workflows.

## Faster Whisper

`faster-whisper` is optional and intended for users who want the fastest English dictation path. The current managed model for it is `distil-large-v3.5`.

Notes:
- English dictation is the intended use case
- if it fails at runtime and a local `large-v3-turbo` Whisper model is available, `whispers` falls back to `whisper_cpp`
- `transcription.idle_timeout_ms = 0` keeps the hidden ASR worker warm indefinitely

## Experimental NeMo backends

`parakeet-tdt_ctc-1.1b` is available as an experimental English-only ASR option on a managed NeMo backend. `canary-qwen-2.5b` remains under evaluation, but the managed path is currently blocked by an upstream initialization issue.

Notes:
- they are intended for benchmarking and experimentation, not as the default recommendation
- first warm-up can be much slower than steady-state dictation because the hidden worker and model need to come up
- the first use bootstraps a hidden managed Python runtime under the XDG data directory
- the runtime currently requires Python 3.10, 3.11, or 3.12 on `PATH`
- model downloads are stored as prepared NeMo model directories instead of ggml files
- if a NeMo backend fails at runtime and a local `large-v3-turbo` Whisper model is available, `whispers` falls back to `whisper_cpp`

## Privacy

- Local-only: no inference-time network traffic
- Cloud ASR: audio leaves the machine for transcription
- Cloud rewrite: transcript/context leaves the machine for rewrite
- Cloud ASR + rewrite: both leave the machine

## uinput permissions

whispers needs access to `/dev/uinput` for the virtual keyboard paste. Add your user to the `input` group:

```sh
sudo usermod -aG input $USER
```

Then log out and back in.

## Acknowledgements

This project is inspired by [hyprwhspr](https://github.com/goodroot/hyprwhspr) by [goodroot](https://github.com/goodroot), which provides native speech-to-text for Linux with support for multiple backends. whispers is a from-scratch Rust reimplementation focused on local-first dictation with minimal dependencies.

## License

[Mozilla Public License 2.0](LICENSE)
