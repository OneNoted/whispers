# AUR packaging

The rewrite maintains one package: `whispers-git`. It builds the small Rust
control plane and downloads the pinned inference runtimes only when the user
runs `whispers setup`; the package build no longer compiles or vendors ML
engines.

After editing `PKGBUILD`, regenerate `.SRCINFO` from its directory:

```sh
makepkg --printsrcinfo > .SRCINFO
```

Publish the directory to
`ssh://aur@aur.archlinux.org/whispers-git.git` after the corresponding change is
recorded in version control.
