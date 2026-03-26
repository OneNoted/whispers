# whispers

Fast speech-to-text dictation for Wayland.

`whispers` is local-first by default, with optional cloud ASR and rewrite backends when you want them. The normal flow is simple: press a key to start recording, press it again to transcribe and paste.

## Install

```sh
# default install: CUDA + local rewrite + OSD
cargo install whispers

# no OSD
cargo install whispers --no-default-features --features cuda,local-rewrite

# no local rewrite
cargo install whispers --no-default-features --features cuda,osd

# CPU-only
cargo install whispers --no-default-features --features local-rewrite,osd
```

If you want the latest GitHub version instead of crates.io:

```sh
cargo install --git https://github.com/OneNoted/whispers
```

## Requirements

- Linux with Wayland
- `wl-copy`
- access to `/dev/uinput`
- Rust 1.85+
- CUDA toolkit for the default install; opt out with `--no-default-features` if you need a CPU-only build

If `/dev/uinput` is blocked, add your user to the `input` group and log back in:

```sh
sudo usermod -aG input $USER
```

## Quick Start

```sh
# generate config and download a model
whispers setup

# one-shot dictation
whispers
```

Default config path:

```text
~/.config/whispers/config.toml
```

Canonical example config:

- [config.example.toml](config.example.toml)

### Keybinding

Hyprland:

```conf
bind = SUPER ALT, D, exec, whispers
```

Sway:

```conf
bindsym $mod+Alt+d exec whispers
```

## Commands

```sh
# setup
whispers setup

# one-shot dictation
whispers
whispers transcribe audio.wav

# ASR models
whispers asr-model list
whispers asr-model download large-v3-turbo
whispers asr-model select large-v3-turbo

# rewrite models
whispers rewrite-model list
whispers rewrite-model download qwen-3.5-4b-q4_k_m
whispers rewrite-model select qwen-3.5-4b-q4_k_m

# personalization
whispers dictionary add "wisper flow" "Wispr Flow"
whispers snippets add signature "Best regards,\nNotes"

# cloud
whispers cloud check

# shell completions
whispers completions zsh
```

## Notes

- Local ASR is the default.
- The default install includes CUDA, local rewrite, and the OSD helper.
- `whispers` installs the helper rewrite worker for you when `local-rewrite` is enabled.
- Shell completions are printed to `stdout`.

## Troubleshooting

If the main `whispers` process ever gets stuck after playback when using local
`whisper_cpp`, enable the built-in hang diagnostics for the next repro:

```sh
WHISPERS_HANG_DEBUG=1 whispers
```

When that mode is enabled, `whispers` writes runtime status and hang bundles
under `${XDG_RUNTIME_DIR:-/tmp}/whispers/`:

- `main-status.json` shows the current dictation stage and recent stage metadata.
- `hang-<pid>-<stage>-<timestamp>.log` is emitted if `whisper_cpp` spends too
  long in model load or transcription.

Those bundles include the current status snapshot plus best-effort stack and
open-file diagnostics. If the hang reproduces, capture the newest `hang-*.log`
file along with `main-status.json`.

## License

Project code in this repository is licensed under the [MIT License](LICENSE).

Bundled third-party code under `vendor/whisper-rs-sys` carries upstream
license notices and file-level exceptions. See [NOTICE](NOTICE), the vendor
license files, and the relevant per-file headers under
`vendor/whisper-rs-sys/whisper.cpp`.
