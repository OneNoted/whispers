# whispers

Fast, local speech-to-text dictation for Wayland with optional advanced local rewrite cleanup.
Press a key to start recording, press it again to transcribe and paste.

Runs [whisper.cpp](https://github.com/ggerganov/whisper.cpp) locally — your audio never leaves your machine.

Inspired by [hyprwhspr](https://github.com/goodroot/hyprwhspr) by goodroot.

<img width="295" height="145" alt="image" src="https://github.com/user-attachments/assets/9005ad84-e09f-476a-ad2f-4aec8d6ad7ef" />



## How it works

1. Bind `whispers` to a key in your compositor
2. First press starts recording (OSD overlay shows audio visualization)
3. Second press stops recording, transcribes with Whisper, and pastes via `Ctrl+Shift+V`

The two invocations communicate via PID file + `SIGUSR1` — no daemon, no IPC server.

## Post-processing modes

`whispers` now has two main dictation modes:

- `raw` keeps output close to the Whisper transcript and is the default
- `advanced_local` runs a second local GGUF rewrite model after Whisper for smarter cleanup and self-correction

The rewrite path is managed by `whispers` itself through an internal helper binary installed alongside the main executable, so there is no separate tool or daemon to install manually.
Managed rewrite models are the default path. If you point `rewrite.model_path` at your own GGUF, it should be a chat-capable model with an embedded template that `llama.cpp` can apply at runtime.

For file transcription, `whispers transcribe --raw <file>` always prints the plain Whisper transcript without any post-processing.

## Requirements

- Rust 1.85+ (edition 2024)
- Linux with Wayland compositor
- `wl-copy` (from `wl-clipboard`)
- `uinput` access (for virtual keyboard paste)
- NVIDIA GPU + CUDA toolkit (optional, for GPU acceleration)
- If no compatible GPU is available, set `whisper.use_gpu = false` in config

## Install

### From source

```sh
# With CUDA (recommended if you have an NVIDIA GPU)
cargo install --git https://github.com/OneNoted/whispers

# Without CUDA
cargo install --git https://github.com/OneNoted/whispers --no-default-features --features osd

# Without OSD overlay
cargo install --git https://github.com/OneNoted/whispers --no-default-features --features cuda
```

### Setup

Run the interactive setup wizard to download a Whisper model, generate config, and optionally enable managed advanced dictation:

```sh
whispers setup
```

Use a custom config file for any command (including `setup` and `model`):

```sh
whispers --config /path/to/config.toml setup
whispers --config /path/to/config.toml model select tiny
```

Or manage models manually:

```sh
whispers model list          # show available models
whispers model download large-v3-turbo
whispers model select large-v3-turbo

whispers rewrite-model list
whispers rewrite-model download llama-3.2-3b-q4_k_m
whispers rewrite-model select llama-3.2-3b-q4_k_m
```

That still remains a single install: `whispers` manages both the Whisper model and the optional rewrite worker/model from the same binary package.

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

[whisper]
model_path = "~/.local/share/whispers/ggml-large-v3-turbo.bin"
language = "auto"      # or "en", "fr", "de", etc.
use_gpu = true         # set false to force CPU
flash_attn = true      # only used when use_gpu=true

[postprocess]
mode = "raw"           # or "advanced_local"

[rewrite]
selected_model = "llama-3.2-3b-q4_k_m"
model_path = ""        # optional manual GGUF path override
timeout_ms = 30000
max_output_chars = 1200
max_tokens = 256

[feedback]
enabled = true
start_sound = ""       # empty = bundled sound
stop_sound = ""
```

## Models

| Model | Size | Speed | Notes |
|-------|------|-------|-------|
| large-v3-turbo | 1.6 GB | Fast | Best balance (recommended) |
| large-v3-turbo-q5_0 | 574 MB | Fast | Quantized, slightly less accurate |
| large-v3 | 3.1 GB | Slow | Most accurate |
| small / small.en | 488 MB | Very fast | Good for English-only |
| tiny / tiny.en | 78 MB | Instant | Least accurate |

Models are downloaded from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp) and stored in `~/.local/share/whispers/`.

## Managed rewrite models

`advanced_local` uses a second local model for post-processing. The managed catalog currently includes:

| Model | Size | Notes |
|-------|------|-------|
| llama-3.2-1b-q4_k_m | ~0.8 GB | Fallback for weaker hardware |
| llama-3.2-3b-q4_k_m | ~2.0 GB | Recommended default |
| llama-3.1-8b-q4_k_m | ~4.9 GB | Higher quality, heavier |

If you want to tinker, set `rewrite.model_path` to a custom GGUF file. When `rewrite.model_path` is set, it overrides the managed selection.
Custom rewrite models should include a chat template that `llama.cpp` can read from the GGUF metadata; otherwise rewrite prompting will fail fast instead of silently producing bad output.

## uinput permissions

whispers needs access to `/dev/uinput` for the virtual keyboard paste. Add your user to the `input` group:

```sh
sudo usermod -aG input $USER
```

Then log out and back in.

## Acknowledgements

This project is inspired by [hyprwhspr](https://github.com/goodroot/hyprwhspr) by [goodroot](https://github.com/goodroot), which provides native speech-to-text for Linux with support for multiple backends. whispers is a from-scratch Rust reimplementation focused on local-only Whisper transcription with minimal dependencies.

## License

[Mozilla Public License 2.0](LICENSE)
