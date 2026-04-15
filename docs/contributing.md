# Contributor workflow

Before you push or open a PR, run the local CI workflow so you catch the same failures that GitHub Actions would catch later.

## Install the pre-push hook

Run this once in your local clone:

```sh
scripts/install-git-hooks.sh
```

That configures:

- `core.hooksPath=.githooks`
- `.githooks/pre-push` to run `scripts/local-ci.sh`

## Run the full local CI workflow manually

```sh
scripts/local-ci.sh
```

By default it mirrors the repo's main CI workflow as closely as practical:

- `cargo fmt --all -- --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test`
- `cargo check --no-default-features`
- `cargo check --no-default-features --features osd`
- `cargo check --no-default-features --features local-rewrite`
- `cargo package --locked --allow-dirty`
- CUDA feature checks when `nvcc` is available
- `scripts/build-release-bundle.sh`

## Useful overrides

You can skip expensive local-only steps with environment variables:

```sh
WHISPERS_LOCAL_CI_SKIP_CUDA=1 scripts/local-ci.sh
WHISPERS_LOCAL_CI_SKIP_PACKAGE=1 scripts/local-ci.sh
WHISPERS_LOCAL_CI_SKIP_RELEASE_BUNDLE=1 scripts/local-ci.sh
```

These are meant for local iteration only; the default workflow should stay as close to CI as possible before you open a PR.
