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

## Whisper.cpp backend

Recognition can run on the external whisper.cpp CLI instead of the managed
sherpa-onnx runtime. Select it during installation:

```sh
whispers setup --backend whisper
```

Setup sets `[recognition].runtime = "whisper-cli"`, skips the sherpa-onnx
downloads, and installs the pinned `ggml-large-v3-turbo.bin` into the data
directory. The equivalent configuration is:

```toml
[recognition]
runtime = "whisper-cli"
fallback_model = "ggml-large-v3-turbo.bin"
```

`fallback_model` is resolved relative to the data directory
(`${XDG_DATA_HOME:-~/.local/share}/whispers`); use an absolute path only for a
model stored elsewhere. The CLI transcribes the finished recording, so this mode
has no partial previews.

Setup requires `whisper-cli` in PATH. The distro `whisper-cpp` package ships only
the CPU `ggml` backend, which takes seconds per sentence with larger models.
Install a GPU backend as well. On Arch Linux:

```sh
sudo pacman -S whisper-cpp ggml-vulkan
```

`ggml-vulkan` provides `/usr/lib/ggml/libggml-vulkan.so`, which `whisper-cli`
loads automatically. Verify on one short recording:

```sh
whisper-cli -m /path/to/ggml-large-v3-turbo.bin -f sample.wav -nt
```

The log must contain `using Vulkan0 backend`. Without a GPU backend the same
model is roughly an order of magnitude slower.

## Setup choices

Normal network setup:

```sh
whispers setup
```

Pick the recognition backend during installation:

```sh
whispers setup --backend whisper
```

`--backend sherpa` is the default and installs the managed runtime. Choosing
whisper requires `whisper-cli` in PATH and installs the GGML model instead; see
[Whisper.cpp backend](#whispercpp-backend).

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

`whispers setup --backend whisper --runtime-pack` uses an existing model and
fails when it is missing, because runtime packs do not carry GGML models.

Setup writes configuration version 2, imports the old dictionary and technical
glossary when present, and copies the old configuration to a timestamped backup.
It never deletes legacy models or configuration.

## Service and keybinding

Setup installs and starts a systemd user service when a Wayland session is
available. At login, systemd skips an early start without display access; the
first toggle then autostarts the daemon inside the compositor session. The
systemd service is convenient rather than mandatory.

Hyprland:

```conf
bind = SUPER ALT, D, exec, whispers
```

Niri:

```kdl
Mod+Alt+D { spawn "whispers"; }
```
