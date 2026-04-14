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
  <a href="#docs"><strong>Docs</strong></a>
  ·
  <a href="#troubleshooting"><strong>Troubleshooting</strong></a>
  ·
  <a href="#releases"><strong>Releases</strong></a>
</p>

`whispers` keeps the default dictation path local, with optional cloud ASR and rewrite backends when you want them. The normal loop is simple: bind `whispers` to a key, press once to start recording, press again to stop, transcribe, and paste into the focused Wayland app.

## What it does

- Local-first speech-to-text for Wayland desktops.
- Optional local rewrite cleanup with a dedicated rewrite worker.
- Optional cloud ASR and rewrite backends when you want to trade locality for convenience.
- Model download, selection, and config management from the CLI.
- Wayland-native OSD support for a cleaner live dictation experience.

## Install

For the full package matrix, prerequisites, and post-install notes, see [docs/install.md](docs/install.md).

### Arch Linux (`paru`)

```sh
paru -S whispers-bin
# or: whispers-git / whispers-cuda-bin / whispers-cuda-git
```

### Cargo

```sh
cargo install whispers
```

`cargo install whispers` follows crates.io releases. The AUR `*-bin` packages follow published GitHub release bundles, and `*-git` packages track the repository `main` branch. If you need rewrite or CUDA features, or want install details before choosing a package, use [docs/install.md](docs/install.md).

## Quick start

```sh
# generate config and download a model
whispers setup

# start dictation (run again to stop, transcribe, and paste)
whispers
```

Default config path:

```text
~/.config/whispers/config.toml
```

Example compositor bindings:

Hyprland:

```conf
bind = SUPER ALT, D, exec, whispers
```

Sway:

```conf
bindsym $mod+Alt+d exec whispers
```

## Docs

- [Installation guide](docs/install.md) — package choices, prerequisites, config path, and feature notes.
- [CLI guide](docs/cli.md) — command groups, examples, and newer rewrite-policy commands.
- [Troubleshooting](docs/troubleshooting.md) — `wl-copy`, `/dev/uinput`, cloud checks, and hang diagnostics.
- [config.example.toml](config.example.toml) — the canonical config template.

## Troubleshooting

If `/dev/uinput` is blocked, run `whispers setup` and let it configure the dedicated `uinput` group and `udev` rule for you. If the main dictation process hangs around local `whisper_cpp` transcription, enable hang diagnostics for the next repro:

```sh
WHISPERS_HANG_DEBUG=1 whispers
```

For the full troubleshooting guide, including the emitted `main-status.json` and `hang-*.log` files, see [docs/troubleshooting.md](docs/troubleshooting.md).

## Releases

Tagged releases publish portable and CUDA-enabled Linux x86_64 bundles. The AUR `whispers-bin` and `whispers-cuda-bin` packages install those published release artifacts.

## License

Project code in this repository is licensed under the [MIT License](LICENSE).

Bundled third-party code under `vendor/whisper-rs-sys` carries upstream license notices and file-level exceptions. See [NOTICE](NOTICE), the vendor license files, and the relevant per-file headers under `vendor/whisper-rs-sys/whisper.cpp`.
