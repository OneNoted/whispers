# whispers

Fast speech-to-text dictation for Wayland.

`whispers` is local-first by default, with optional cloud ASR and rewrite backends when you want them. The normal flow is simple: press a key to start recording, press it again to transcribe and paste.

## Install

```sh
# default install
cargo install whispers

# CUDA
cargo install whispers --features cuda

# local rewrite support
cargo install whispers --features local-rewrite

# CUDA + local rewrite
cargo install whispers --features cuda,local-rewrite

# no OSD
cargo install whispers --no-default-features
```

If you want the latest GitHub version instead of crates.io:

```sh
cargo install --git https://github.com/OneNoted/whispers --features cuda,local-rewrite
```

## Requirements

- Linux with Wayland
- `wl-copy`
- access to `/dev/uinput`
- Rust 1.85+
- CUDA toolkit if you enable the `cuda` feature

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
- Local rewrite is installed automatically with `--features local-rewrite`.
- `whispers` installs the helper rewrite worker for you when that feature is enabled.
- Shell completions are printed to `stdout`.

## License

Project code in this repository is licensed under the [MIT License](LICENSE).

Bundled third-party code under `vendor/whisper-rs-sys` is also MIT-licensed.
See [NOTICE](NOTICE) and the vendor license files for details.
