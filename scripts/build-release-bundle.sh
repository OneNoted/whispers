#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

version="${1:-$(awk -F '"' '$1 ~ /^version *=/ { print $2; exit }' Cargo.toml)}"
host_triple="$(rustc -vV | sed -n 's/^host: //p')"
target_triple="${WHISPERS_RELEASE_TARGET:-$host_triple}"
dist_dir="${WHISPERS_RELEASE_DIST_DIR:-$repo_root/dist}"
bundle_name="whispers-${version}-${target_triple}"
bundle_root="$dist_dir/$bundle_name"
tarball="$dist_dir/$bundle_name.tar.gz"
checksum="$tarball.sha256"
target_dir="${CARGO_TARGET_DIR:-target}/$target_triple/release"

mkdir -p "$dist_dir"
rm -rf -- "$bundle_root"
rm -f -- "$tarball" "$checksum"

cargo build --release --locked --target "$target_triple"
test -x "$target_dir/whispers"

stage_dir="$(mktemp -d)"
trap 'rm -rf -- "$stage_dir"' EXIT

install -Dm755 "$target_dir/whispers" "$stage_dir/$bundle_name/bin/whispers"
install -Dm644 packaging/systemd/whispers.service \
  "$stage_dir/$bundle_name/share/systemd/user/whispers.service"
install -Dm644 sounds/start.wav "$stage_dir/$bundle_name/share/whispers/sounds/start.wav"
install -Dm644 sounds/stop.wav "$stage_dir/$bundle_name/share/whispers/sounds/stop.wav"
install -Dm644 README.md "$stage_dir/$bundle_name/share/doc/whispers/README.md"
install -Dm644 docs/architecture.md "$stage_dir/$bundle_name/share/doc/whispers/architecture.md"
install -Dm644 docs/install.md "$stage_dir/$bundle_name/share/doc/whispers/install.md"
install -Dm644 docs/cli.md "$stage_dir/$bundle_name/share/doc/whispers/cli.md"
install -Dm644 docs/troubleshooting.md \
  "$stage_dir/$bundle_name/share/doc/whispers/troubleshooting.md"
install -Dm644 config.example.toml \
  "$stage_dir/$bundle_name/share/doc/whispers/config.example.toml"
install -Dm644 LICENSE "$stage_dir/$bundle_name/share/licenses/whispers/LICENSE"
install -Dm644 NOTICE "$stage_dir/$bundle_name/share/licenses/whispers/NOTICE"

tar --sort=name --mtime="@${SOURCE_DATE_EPOCH:-0}" --owner=0 --group=0 \
  --numeric-owner -C "$stage_dir" -czf "$tarball" "$bundle_name"
(
  cd "$dist_dir"
  sha256sum "$(basename "$tarball")" > "$(basename "$checksum")"
)

printf 'created %s\ncreated %s\n' "$tarball" "$checksum"
