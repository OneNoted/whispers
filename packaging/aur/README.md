# AUR packaging

This repository keeps the maintained AUR sources for:

- `whispers-bin`: installs the portable GitHub release bundle
- `whispers-git`: builds the latest `main` branch without CUDA
- `whispers-cuda-bin`: installs the CUDA-enabled GitHub release bundle
- `whispers-cuda-git`: builds the latest `main` branch with CUDA

Current feature sets:

- `whispers-bin` / `whispers-git`: `local-rewrite,osd`
- `whispers-cuda-bin` / `whispers-cuda-git`: `cuda,local-rewrite,osd`

## Refreshing metadata

Whenever you edit a `PKGBUILD`, regenerate its `.SRCINFO` from the package
directory:

```sh
makepkg --printsrcinfo > .SRCINFO
```

## Updating `whispers-bin` / `whispers-cuda-bin`

1. Cut a GitHub release for `vX.Y.Z`.
2. Update `pkgver` and `sha256sums` in the matching `*-bin/PKGBUILD`.
3. Regenerate the matching `*-bin/.SRCINFO`.

## Publishing to the AUR

Each AUR package base lives in its own Git repository. Push the contents of the
matching directory to:

- `ssh://aur@aur.archlinux.org/whispers-bin.git`
- `ssh://aur@aur.archlinux.org/whispers-git.git`
- `ssh://aur@aur.archlinux.org/whispers-cuda-bin.git`
- `ssh://aur@aur.archlinux.org/whispers-cuda-git.git`
