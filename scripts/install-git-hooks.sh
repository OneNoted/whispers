#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

git config core.hooksPath .githooks
chmod +x .githooks/pre-push

printf 'Configured git hooks path: %s\n' "$(git config --get core.hooksPath)"
printf 'Installed pre-push hook: %s/.githooks/pre-push\n' "$repo_root"
