# Install

`whispers` targets Linux Wayland desktops and supports a few different installation paths depending on whether you want stable release bundles, the latest `main` branch, local rewrite, or CUDA.

## Requirements

- Linux with Wayland
- `wl-copy` (usually provided by `wl-clipboard`)
- Access to `/dev/uinput` for paste injection
- Rust 1.85+ for Cargo installs and source builds
- CUDA toolkit only when you build with the `cuda` feature yourself

If `/dev/uinput` is not ready, run `whispers setup`. It can configure the dedicated `uinput` group and matching `udev` rule automatically.

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

### Package matrix

| Package | Source | Feature set |
| --- | --- | --- |
| `whispers-bin` | Published GitHub release bundle | `local-rewrite,osd` |
| `whispers-git` | Latest `main` branch build | `local-rewrite,osd` |
| `whispers-cuda-bin` | Published GitHub CUDA release bundle | `cuda,local-rewrite,osd` |
| `whispers-cuda-git` | Latest `main` branch build with CUDA | `cuda,local-rewrite,osd` |

## Cargo

### crates.io

```sh
# default install (OSD enabled)
cargo install whispers

# add local rewrite support
cargo install whispers --features local-rewrite

# add CUDA + local rewrite
cargo install whispers --features cuda,local-rewrite

# no OSD
cargo install whispers --no-default-features
```

### GitHub source install

If you want the current repository version instead of the latest crates.io publish:

```sh
cargo install --git https://github.com/OneNoted/whispers --features local-rewrite
```

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

## Shell completions

The Arch packages and release bundles install completions in each shell's native
system location. Fish discovers the installed
`/usr/share/fish/vendor_completions.d/whispers.fish` automatically; open a new
fish session after installing or upgrading the package.

Cargo deliberately has no post-install hook, so `cargo install` cannot safely
write into a shell configuration directory. For a Cargo install, fish needs this
one-time user-level installation:

```fish
mkdir -p ~/.config/fish/completions
whispers completions fish > ~/.config/fish/completions/whispers.fish
```

Fish loads that file automatically in future sessions. The same command can
print scripts for Bash, Zsh, and Nushell when you need to install them manually.

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

Tagged releases publish portable and CUDA-enabled Linux x86_64 tarballs. The AUR `*-bin` packages install those published release artifacts rather than building from source on your machine.
