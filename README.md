<h1 align="center">whispers</h1>

<p align="center">Private intended-text dictation for Wayland.<br />Press once, speak naturally, press again.</p>

<p align="center">
  <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/OneNoted/whispers/ci.yml?branch=main&label=ci&style=flat-square" />
  <img alt="License" src="https://img.shields.io/github/license/OneNoted/whispers?style=flat-square" />
  <img alt="Rust 1.85+" src="https://img.shields.io/badge/rust-1.85%2B-000000?style=flat-square&logo=rust" />
</p>

Whispers is one local dictation pipeline, not a collection of modes. It records
from PipeWire, shows a low-latency Nemotron preview, then transcribes the full
recording with Parakeet before applying your vocabulary, spoken symbols, and a
conservative local Qwen cleanup. The result is typed into the window where
dictation began.

A compact bottom-center OSD shows whether Whispers is listening, transcribing,
enhancing, or delivering. It runs out of process through Quickshell, keeping GUI
toolkits out of the Rust build.

Saying “this slash that” produces `this/that`. Saying “the word slash” keeps the
word. `literal slash` is the explicit escape hatch.

Clear spoken number lists use digits. Saying “one, two, three, put a hyphen
between those” produces `1-2-3`; `comma`, `slash`, and `space` work as explicit
separators too.

## Why the rewrite is smaller

The Rust application is a control plane with seven direct dependencies. It does
not compile or vendor whisper.cpp, llama.cpp, CUDA, an async runtime, an audio
stack, or a GUI toolkit. `whispers setup` downloads checksum-pinned sherpa-onnx,
Nemotron, Parakeet, llama.cpp, and Qwen artifacts. Inference runtimes can
therefore be
upgraded independently without turning every application build into an ML
toolchain build.

There is one binary and one interaction:

```sh
whispers setup
whispers        # start listening
whispers        # finish, infer intended text, and deliver
```

No audio or transcript is retained. Setup is the only operation that needs the
network; dictation and enhancement run locally.

## Install

Linux x86_64, Wayland, PipeWire, `wtype`, `wl-copy`, `curl`, `tar`, `bzip2`, and
`sha256sum` are required. On Arch Linux:

```sh
paru -S whispers-git
whispers setup
```

Or build from source:

```sh
cargo install --path . --locked
whispers setup
```

Setup downloads roughly 4 GB and preserves any version 1 configuration as a
timestamped backup before importing useful settings and vocabulary.

Bind `whispers` to one compositor shortcut. For Hyprland:

```conf
bind = SUPER ALT, D, exec, whispers
```

For Niri:

```kdl
Mod+Alt+D { spawn "whispers"; }
```

## Commands

```text
whispers                         Toggle dictation
whispers cancel                  Cancel recording or finalization
whispers status                  Print the service state
whispers setup                   Install runtimes and migrate configuration
whispers doctor                  Diagnose the local runtime
whispers vocabulary list
whispers vocabulary add TERM --alias SPOKEN
whispers vocabulary remove TERM
```

See [installation](docs/install.md), the [CLI guide](docs/cli.md), and
[architecture](docs/architecture.md), plus
[troubleshooting](docs/troubleshooting.md), for details. The generated
configuration lives at `~/.config/whispers/config.toml`; the canonical shape is
in [config.example.toml](config.example.toml).

## License

Whispers is licensed under the [MIT License](LICENSE). Downloaded inference
runtimes and models retain their respective upstream licenses and are not
redistributed in this repository.
