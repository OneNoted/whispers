#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

direct_dependencies="$(cargo tree --depth 1 --prefix none | tail -n +2 | sort -u | wc -l)"
resolved_packages="$(cargo tree --prefix none | sort -u | wc -l)"

test "$direct_dependencies" -le 10 || {
  printf 'direct dependency budget exceeded: %s > 10\n' "$direct_dependencies" >&2
  exit 1
}
test "$resolved_packages" -le 40 || {
  printf 'resolved package budget exceeded: %s > 40\n' "$resolved_packages" >&2
  exit 1
}
test ! -d vendor || {
  printf 'vendored source trees are forbidden\n' >&2
  exit 1
}
if cargo tree --prefix none | grep -Eq '^(tokio|reqwest|clap|cpal|rodio|whisper-rs|llama-cpp-2) '; then
  printf 'a retired heavyweight dependency returned\n' >&2
  exit 1
fi

printf 'build budget: %s direct dependencies, %s resolved packages\n' \
  "$direct_dependencies" "$resolved_packages"
