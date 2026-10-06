#!/usr/bin/env bash
# Builds the native PyO3 extension into the importable package directory.
# Python package: engine/python/pa3_engine  (extension is gitignored)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export RUSTUP_HOME="$PWD/.tools/rustup"
export CARGO_HOME="$PWD/.tools/cargo"
export PATH="$CARGO_HOME/bin:$PATH"
cargo build --release --locked --features python
cp -f target/release/libpa3_engine.so engine/python/pa3_engine/pa3_engine.so
printf '%s\n' "built engine/python/pa3_engine/pa3_engine.so"
