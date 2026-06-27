# AUR packaging

This repository keeps the maintained AUR sources for:

- `whispers-bin`: installs the portable GitHub release bundle
- `whispers-git`: builds the latest `main` branch without CUDA
- `whispers-cuda-bin`: installs the CUDA-enabled GitHub release bundle
- `whispers-cuda-git`: builds the latest `main` branch with CUDA
- `whispers-vulkan-bin`: installs the Vulkan-enabled GitHub release bundle
- `whispers-vulkan-git`: builds the latest `main` branch with Vulkan

Current feature sets:

- `whispers-bin` / `whispers-git`: `local-rewrite,osd`
- `whispers-cuda-bin` / `whispers-cuda-git`: `cuda,local-rewrite,osd`
- `whispers-vulkan-bin` / `whispers-vulkan-git`: `vulkan,local-rewrite,osd`

## Refreshing metadata

Whenever you edit a `PKGBUILD`, regenerate its `.SRCINFO` from the package
directory:

```sh
makepkg --printsrcinfo > .SRCINFO
```

## Updating `*-bin` packages

1. Cut a GitHub release for `vX.Y.Z`.
2. Wait for the release workflow to upload the matching tarball and `.sha256`
   asset.
3. Update `pkgver` and `sha256sums` in the matching `*-bin/PKGBUILD` from the
   published release asset.
4. Regenerate the matching `*-bin/.SRCINFO`.

Do not update the `*-bin` AUR packages from `main` alone. They install tagged
release bundles, so user-visible README behavior in `main` only reaches the
matching `*-bin` package after a new GitHub release is published.

`whispers-vulkan-bin` requires a matching
`whispers-vulkan-X.Y.Z-x86_64-unknown-linux-gnu.tar.gz` release asset.

## Publishing to the AUR

Each AUR package base lives in its own Git repository. Push the contents of the
matching directory to:

- `ssh://aur@aur.archlinux.org/whispers-bin.git`
- `ssh://aur@aur.archlinux.org/whispers-git.git`
- `ssh://aur@aur.archlinux.org/whispers-cuda-bin.git`
- `ssh://aur@aur.archlinux.org/whispers-cuda-git.git`
- `ssh://aur@aur.archlinux.org/whispers-vulkan-bin.git`
- `ssh://aur@aur.archlinux.org/whispers-vulkan-git.git`
