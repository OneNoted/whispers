# AUR packaging

This repository keeps the maintained AUR sources for:

- `whispers-bin`: installs the GitHub release bundle
- `whispers-git`: builds the latest `main` branch from source

Both packages currently ship the portable `local-rewrite,osd` feature set.

## Refreshing metadata

Whenever you edit a `PKGBUILD`, regenerate its `.SRCINFO` from the package
directory:

```sh
makepkg --printsrcinfo > .SRCINFO
```

## Updating `whispers-bin`

1. Cut a GitHub release for `vX.Y.Z`.
2. Update `pkgver` and `sha256sums` in `packaging/aur/whispers-bin/PKGBUILD`.
3. Regenerate `packaging/aur/whispers-bin/.SRCINFO`.

## Publishing to the AUR

Each AUR package base lives in its own Git repository. Push the contents of the
matching directory to:

- `ssh://aur@aur.archlinux.org/whispers-bin.git`
- `ssh://aur@aur.archlinux.org/whispers-git.git`
