#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

skip_cuda="${WHISPERS_LOCAL_CI_SKIP_CUDA:-0}"
skip_vulkan="${WHISPERS_LOCAL_CI_SKIP_VULKAN:-0}"
skip_package="${WHISPERS_LOCAL_CI_SKIP_PACKAGE:-0}"
skip_release_bundle="${WHISPERS_LOCAL_CI_SKIP_RELEASE_BUNDLE:-0}"

run_step() {
  local label="$1"
  shift
  printf '\n==> %s\n' "$label"
  "$@"
}

run_step "Check formatting" cargo fmt --all -- --check
run_step "Clippy (default features)" cargo clippy --all-targets -- -D warnings
run_step "Test (default features)" cargo test

run_step "Check no default features" cargo check --no-default-features
run_step "Check osd feature only" cargo check --no-default-features --features osd
run_step "Check local rewrite feature only" cargo check --no-default-features --features local-rewrite

if [[ "$skip_package" != "1" ]]; then
  run_step "Package crate" cargo package --locked --allow-dirty
else
  printf '\n==> Skipping cargo package (--allow-dirty) because WHISPERS_LOCAL_CI_SKIP_PACKAGE=1\n'
fi

if [[ "$skip_cuda" == "1" ]]; then
  printf '\n==> Skipping CUDA checks because WHISPERS_LOCAL_CI_SKIP_CUDA=1\n'
elif command -v nvcc >/dev/null 2>&1; then
  run_step "Check cuda feature only" cargo check --no-default-features --features cuda
  run_step "Check cuda + local rewrite features" cargo check --no-default-features --features cuda,local-rewrite
else
  printf '\n==> Skipping CUDA checks because nvcc is not available on PATH\n'
fi

if [[ "$skip_vulkan" == "1" ]]; then
  printf '\n==> Skipping Vulkan checks because WHISPERS_LOCAL_CI_SKIP_VULKAN=1\n'
elif command -v pkg-config >/dev/null 2>&1 \
  && pkg-config --exists vulkan \
  && command -v glslc >/dev/null 2>&1; then
  run_step "Check vulkan feature only" cargo check --no-default-features --features vulkan
  run_step "Check vulkan + local rewrite features" cargo check --no-default-features --features vulkan,local-rewrite
else
  printf '\n==> Skipping Vulkan checks because Vulkan development files or glslc are not available\n'
fi

if [[ "$skip_release_bundle" != "1" ]]; then
  run_step "Build release bundle" scripts/build-release-bundle.sh
else
  printf '\n==> Skipping release bundle because WHISPERS_LOCAL_CI_SKIP_RELEASE_BUNDLE=1\n'
fi

printf '\nAll local CI checks passed.\n'
