#!/usr/bin/env bash
# Installs only under this checkout; leaves shell profiles and system packages alone.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export RUSTUP_HOME="$PWD/.tools/rustup"
export CARGO_HOME="$PWD/.tools/cargo"
export PATH="$CARGO_HOME/bin:$PATH"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  printf '%s\n' 'This bootstrap is pinned for the MiniDC Linux x86_64 host.' >&2
  exit 1
fi
if [[ ! -x "$CARGO_HOME/bin/rustup" ]]; then
  mkdir -p .tools/bootstrap
  curl --fail --location --silent --show-error \
    https://static.rust-lang.org/rustup/archive/1.29.1/x86_64-unknown-linux-gnu/rustup-init \
    -o .tools/bootstrap/rustup-init
  printf '%s\n' 'dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71  .tools/bootstrap/rustup-init' | sha256sum --check --status
  chmod u+x .tools/bootstrap/rustup-init
  .tools/bootstrap/rustup-init -y --no-modify-path --profile minimal --default-toolchain 1.90.0
fi
rustup toolchain install 1.90.0 --profile minimal --component rustfmt --component clippy
cargo --version
rustc --version
