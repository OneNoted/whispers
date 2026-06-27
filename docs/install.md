# Install

`whispers` targets Linux Wayland desktops and supports a few different installation paths depending on whether you want stable release bundles, the latest `main` branch, local rewrite, CUDA, or Vulkan.

## Requirements

- Linux with Wayland
- `wl-copy` (usually provided by `wl-clipboard`)
- Access to `/dev/uinput` for paste injection
- Rust 1.85+ for Cargo installs and source builds
- CUDA toolkit only when you build with the `cuda` feature yourself
- Vulkan loader plus a GPU driver when you use a Vulkan-enabled build
- Vulkan headers when you build with the `vulkan` feature yourself

If `/dev/uinput` is not ready, run `whispers setup`. It can configure the dedicated `uinput` group and matching `udev` rule automatically.

On Arch with AMD graphics, the runtime Vulkan driver package is usually `vulkan-radeon`. `vulkan-tools` is useful for confirming the driver with `vulkaninfo`, but it is not required by `whispers`.

## Arch Linux (`paru`)

### Portable / non-CUDA

```sh
# published GitHub release bundle
paru -S whispers-bin

# latest main branch build
paru -S whispers-git
```

### CUDA-enabled

```sh
# published GitHub release bundle with CUDA support
paru -S whispers-cuda-bin

# latest main branch build with CUDA support
paru -S whispers-cuda-git
```

### Vulkan-enabled

```sh
# latest main branch build with Vulkan support
paru -S whispers-vulkan-git
```

The `whispers-vulkan-bin` package should only be published after the matching GitHub release contains a `whispers-vulkan-*` tarball.

### Package matrix

| Package | Source | Feature set |
| --- | --- | --- |
| `whispers-bin` | Published GitHub release bundle | `local-rewrite,osd` |
| `whispers-git` | Latest `main` branch build | `local-rewrite,osd` |
| `whispers-cuda-bin` | Published GitHub CUDA release bundle | `cuda,local-rewrite,osd` |
| `whispers-cuda-git` | Latest `main` branch build with CUDA | `cuda,local-rewrite,osd` |
| `whispers-vulkan-git` | Latest `main` branch build with Vulkan | `vulkan,local-rewrite,osd` |

## Cargo

### crates.io

```sh
# default install (OSD enabled)
cargo install whispers

# add local rewrite support
cargo install whispers --features local-rewrite

# add CUDA + local rewrite
cargo install whispers --features cuda,local-rewrite

# add Vulkan + local rewrite
cargo install whispers --features vulkan,local-rewrite

# no OSD
cargo install whispers --no-default-features
```

### GitHub source install

If you want the current repository version instead of the latest crates.io publish:

```sh
cargo install --git https://github.com/OneNoted/whispers --features local-rewrite
```

For a source install with Vulkan support:

```sh
cargo install --git https://github.com/OneNoted/whispers --features vulkan,local-rewrite
```

## Local `whisper_cpp` tuning

Vulkan and CUDA builds use `[transcription].use_gpu = true` to request whisper.cpp GPU acceleration. CPU thread count is controlled separately:

```toml
[transcription]
use_gpu = true
threads = 0
```

`threads = 0` means auto and is capped to 8 logical CPUs. Set a positive value if you want to force a specific whisper.cpp worker count.

## After install

Generate a config, download an ASR model, and walk through optional rewrite/cloud setup:

```sh
whispers setup
```

Default config path:

```text
~/.config/whispers/config.toml
```

Canonical example config:

- [config.example.toml](../config.example.toml)

## Keybinding examples

Hyprland:

```conf
bind = SUPER ALT, D, exec, whispers
```

Sway:

```conf
bindsym $mod+Alt+d exec whispers
```

The first invocation starts recording. The next invocation stops recording, transcribes, and pastes into the currently focused app.

## Release bundles

Tagged releases publish portable, CUDA-enabled, and Vulkan-enabled Linux x86_64 tarballs. The AUR `*-bin` packages install those published release artifacts rather than building from source on your machine.
