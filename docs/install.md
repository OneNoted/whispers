# Install

Whispers targets Linux x86_64 Wayland sessions. The Rust build is deliberately small;
inference engines and models are installed at runtime instead of compiled into
the application.

## System requirements

- PipeWire tools (`pw-cat` and `pw-play`)
- `wtype`
- `wl-copy` from `wl-clipboard`
- `curl`, `tar`, `bzip2`, and `sha256sum` for setup
- Quickshell for the status OSD
- `notify-send` for the optional fallback when Quickshell is unavailable
- Rust 1.85 or newer when building from source

The managed assets require about 4 GB of download bandwidth and somewhat more
installed space. Dictation itself does not require network access.

## Arch Linux

```sh
paru -S whispers-git
whispers setup
```

The AUR package builds only the control plane. Setup installs pinned inference
artifacts under `${XDG_DATA_HOME:-~/.local/share}/whispers`.

## From source

```sh
cargo build --release --locked
install -Dm755 target/release/whispers ~/.local/bin/whispers
whispers setup
```

There are no feature combinations or CUDA-specific application builds. The
managed llama.cpp runtime uses Vulkan acceleration when a compatible GPU is
available and retains its CPU backend as a fallback. GPU support remains a
downloaded runtime, not a Rust build dependency.

The binary installs a small Quickshell status OSD on first use. It remains a
separate process, so the interface adds no GUI toolkit to the Rust dependency
graph. The OSD follows dictation through listening, transcription, enhancement,
and delivery without taking keyboard focus.

## Setup choices

Normal network setup:

```sh
whispers setup
```

Install without enabling the user service:

```sh
whispers setup --no-service
```

For an offline machine, prepare a directory with this shape and pass it as a
runtime pack:

```text
pack/
├── models/
│   ├── nemotron-3.5-asr-streaming-0.6b-560ms-int8/
│   └── nemo-parakeet-tdt-0.6b-v3-int8/
├── rewrite/Qwen_Qwen3.5-4B-Q4_K_M.gguf
└── runtimes/
    ├── llama.cpp/b10258-vulkan/
    └── sherpa-onnx/current/
```

```sh
whispers setup --runtime-pack ./pack
```

Setup writes configuration version 2, imports the old dictionary and technical
glossary when present, and copies the old configuration to a timestamped backup.
It never deletes legacy models or configuration.

## Service and keybinding

Setup installs and starts a systemd user service when available. Toggle also
autostarts the daemon, so the systemd service is convenient rather than
mandatory.

Hyprland:

```conf
bind = SUPER ALT, D, exec, whispers
```

Niri:

```kdl
Mod+Alt+D { spawn "whispers"; }
```
