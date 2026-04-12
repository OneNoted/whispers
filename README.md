<h1 align="center">whispers</h1>

<p align="center">Fast local-first speech-to-text dictation for Wayland.<br />Press a key, speak, transcribe, and paste.</p>

<p align="center">
  <img alt="Release" src="https://img.shields.io/github/v/release/OneNoted/whispers?display_name=tag&style=flat-square" />
  <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/OneNoted/whispers/ci.yml?branch=main&label=ci&style=flat-square" />
  <img alt="License" src="https://img.shields.io/github/license/OneNoted/whispers?style=flat-square" />
  <img alt="Rust 1.85+" src="https://img.shields.io/badge/rust-1.85%2B-000000?style=flat-square&logo=rust" />
  <img alt="Wayland" src="https://img.shields.io/badge/linux-wayland-7c3aed?style=flat-square" />
</p>

<p align="center">
  <a href="#install"><strong>Install</strong></a>
  ·
  <a href="#quick-start"><strong>Quick start</strong></a>
  ·
  <a href="#commands"><strong>Commands</strong></a>
  ·
  <a href="#troubleshooting"><strong>Troubleshooting</strong></a>
  ·
  <a href="#releases"><strong>Releases</strong></a>
</p>

`whispers` keeps the default dictation path local, with optional cloud ASR and rewrite backends when you want them. The normal loop is simple: start recording with a keybinding, stop recording, and paste the transcript directly into the focused app.

## What it does

- Local-first speech-to-text for Wayland desktops.
- Optional local rewrite cleanup with a dedicated rewrite worker.
- Optional cloud ASR and rewrite backends when you want to trade locality for convenience.
- Model download, selection, and config management from the CLI.
- Wayland-native OSD support for a cleaner live dictation experience.

## Install

### Arch Linux (`paru`)

```sh
# prebuilt GitHub release bundle
paru -S whispers-bin

# latest main branch build
paru -S whispers-git
```

- `whispers-bin` installs the published Linux x86_64 release bundle.
- `whispers-git` builds the latest `main` branch from source.
- Both AUR packages currently ship the portable `local-rewrite,osd` feature set.

### Cargo

```sh
# crates.io with the default OSD-enabled install
cargo install whispers

# add local rewrite support
cargo install whispers --features local-rewrite

# add CUDA + local rewrite
cargo install whispers --features cuda,local-rewrite

# no OSD
cargo install whispers --no-default-features
```

If you want the latest GitHub version instead of crates.io:

```sh
cargo install --git https://github.com/OneNoted/whispers --features local-rewrite
```

## Requirements

- Linux with Wayland
- `wl-copy`
- access to `/dev/uinput`
- Rust 1.85+
- CUDA toolkit if you enable the `cuda` feature

If `/dev/uinput` is blocked, add your user to the `input` group and log back in:

```sh
sudo usermod -aG input "$USER"
```

## Quick start

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

- Local ASR is the default path.
- Local rewrite is enabled when you install with `--features local-rewrite` or use the current AUR packages.
- `whispers` installs the helper rewrite worker for you when that feature is enabled.
- Shell completions are printed to `stdout`.

## Troubleshooting

If the main `whispers` process ever gets stuck after playback when using local `whisper_cpp`, enable the built-in hang diagnostics for the next repro:

```sh
WHISPERS_HANG_DEBUG=1 whispers
```

When that mode is enabled, `whispers` writes runtime status and hang bundles under `${XDG_RUNTIME_DIR:-/tmp}/whispers/`:

- `main-status.json` shows the current dictation stage and recent stage metadata.
- `hang-<pid>-<stage>-<timestamp>.log` is emitted if `whisper_cpp` spends too long in model load or transcription.

Those bundles include the current status snapshot plus best-effort stack and open-file diagnostics. If the hang reproduces, capture the newest `hang-*.log` file along with `main-status.json`.

## Releases

Tagged releases publish a Linux x86_64 bundle with:

- `whispers`
- `whispers-osd`
- `whispers-rewrite-worker`
- Bash, Zsh, and Fish completions
- `README.md`, `config.example.toml`, `LICENSE`, and `NOTICE`

That bundle is what the `whispers-bin` AUR package installs.

## License

Project code in this repository is licensed under the [MIT License](LICENSE).

Bundled third-party code under `vendor/whisper-rs-sys` carries upstream license notices and file-level exceptions. See [NOTICE](NOTICE), the vendor license files, and the relevant per-file headers under `vendor/whisper-rs-sys/whisper.cpp`.
