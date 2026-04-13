#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

version="${1:-$(awk -F '"' '$1 ~ /^version *=/ { print $2; exit }' Cargo.toml)}"
profile="${WHISPERS_RELEASE_PROFILE:-release}"
features="${WHISPERS_RELEASE_FEATURES:-local-rewrite,osd}"
target_triple="${WHISPERS_RELEASE_TARGET:-x86_64-unknown-linux-gnu}"
bundle_prefix="${WHISPERS_RELEASE_BUNDLE_PREFIX:-whispers}"
dist_dir="${WHISPERS_RELEASE_DIST_DIR:-$repo_root/dist}"
bundle_name="${bundle_prefix}-${version}-${target_triple}"
bundle_root="$dist_dir/$bundle_name"
tarball_path="$dist_dir/${bundle_name}.tar.gz"
sha_path="$dist_dir/${bundle_name}.tar.gz.sha256"
commit_sha="$(git rev-parse --short=12 HEAD)"
commit_epoch="$(git log -1 --format=%ct HEAD)"
target_dir="target/${target_triple}/${profile}"

mkdir -p "$dist_dir"
rm -rf "$bundle_root" "$tarball_path" "$sha_path"

cargo build \
  --profile "$profile" \
  --locked \
  --target "$target_triple" \
  --no-default-features \
  --features "$features"

for binary in whispers whispers-osd whispers-rewrite-worker; do
  if [[ ! -x "${target_dir}/${binary}" ]]; then
    echo "expected ${target_dir}/${binary} to exist after build" >&2
    exit 1
  fi
done

stage_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$stage_dir"
}
trap cleanup EXIT

mkdir -p \
  "$stage_dir/$bundle_name/bin" \
  "$stage_dir/$bundle_name/share/bash-completion/completions" \
  "$stage_dir/$bundle_name/share/zsh/site-functions" \
  "$stage_dir/$bundle_name/share/fish/vendor_completions.d" \
  "$stage_dir/$bundle_name/share/doc/whispers" \
  "$stage_dir/$bundle_name/share/licenses/whispers"

install -Dm755 "${target_dir}/whispers" \
  "$stage_dir/$bundle_name/bin/whispers"
install -Dm755 "${target_dir}/whispers-osd" \
  "$stage_dir/$bundle_name/bin/whispers-osd"
install -Dm755 "${target_dir}/whispers-rewrite-worker" \
  "$stage_dir/$bundle_name/bin/whispers-rewrite-worker"

"${target_dir}/whispers" completions bash \
  > "$stage_dir/$bundle_name/share/bash-completion/completions/whispers"
"${target_dir}/whispers" completions zsh \
  > "$stage_dir/$bundle_name/share/zsh/site-functions/_whispers"
"${target_dir}/whispers" completions fish \
  > "$stage_dir/$bundle_name/share/fish/vendor_completions.d/whispers.fish"

install -Dm644 README.md \
  "$stage_dir/$bundle_name/share/doc/whispers/README.md"
install -Dm644 config.example.toml \
  "$stage_dir/$bundle_name/share/doc/whispers/config.example.toml"
install -Dm644 LICENSE \
  "$stage_dir/$bundle_name/share/licenses/whispers/LICENSE"
install -Dm644 NOTICE \
  "$stage_dir/$bundle_name/share/licenses/whispers/NOTICE"

cat > "$stage_dir/$bundle_name/share/doc/whispers/RELEASE-BUNDLE.txt" <<EOF
version=${version}
target=${target_triple}
features=${features}
commit=${commit_sha}
EOF

tar \
  --sort=name \
  --mtime="@${commit_epoch}" \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  -C "$stage_dir" \
  -czf "$tarball_path" \
  "$bundle_name"

(
  cd "$dist_dir"
  sha256sum "$(basename "$tarball_path")" > "$(basename "$sha_path")"
)

printf 'created %s\n' "$tarball_path"
printf 'created %s\n' "$sha_path"
