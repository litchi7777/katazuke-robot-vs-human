#!/usr/bin/env bash
# Build the pinned RobStride parameter-notification fix without replacing pip packages.
set -euo pipefail

rebot_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
rebot_source="$rebot_root/third_party/motorbridge"
rebot_patch="$rebot_root/patches/motorbridge-0.5.6-param-notify.patch"
rebot_pin=c652ce420e7da1008fa1864cc7f47d7d7c57fdb6
rebot_fixed_pin=a27d1a60d3eb6f8d106005d1534b267eb334829b

if [[ ! -d "$rebot_source/.git" && ! -f "$rebot_source/.git" ]]; then
    git clone --depth 1 --branch v0.5.6 https://github.com/motorbridge/motorbridge.git "$rebot_source"
fi
rebot_head="$(git -C "$rebot_source" rev-parse HEAD)"
if [[ "$rebot_head" != "$rebot_pin" && "$rebot_head" != "$rebot_fixed_pin" ]]; then
    printf 'motorbridge HEAD differs from the pinned commit; existing checkout was preserved.\n' >&2
    exit 1
fi
if git -C "$rebot_source" apply --reverse --check "$rebot_patch" 2>/dev/null; then
    printf 'motorbridge notification patch already applied.\n'
elif git -C "$rebot_source" apply --check "$rebot_patch"; then
    git -C "$rebot_source" apply "$rebot_patch"
else
    printf 'Patch cannot be applied cleanly; existing source was preserved.\n' >&2
    exit 1
fi

if [[ -x "$rebot_root/.toolchains/cargo/bin/cargo" ]]; then
    export CARGO_HOME="$rebot_root/.toolchains/cargo"
    export RUSTUP_HOME="$rebot_root/.toolchains/rustup"
    export PATH="$CARGO_HOME/bin:$PATH"
fi
if ! command -v cargo >/dev/null; then
    printf 'Rust/cargo is required. Install Rust using https://rustup.rs/ and run this script again.\n' >&2
    exit 1
fi

cargo test --locked --manifest-path "$rebot_source/Cargo.toml" \
    -p motor_core -p motor_vendor_robstride -p motor_abi
cargo build --locked --manifest-path "$rebot_source/Cargo.toml" -p motor_abi --release

REBOT_BUILD_ROOT="$rebot_root" REBOT_UPSTREAM_COMMIT="$rebot_pin" python3 - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path(os.environ["REBOT_BUILD_ROOT"])
source = root / "third_party/motorbridge"
library = source / "target/release/libmotor_abi.so"
metadata = {
    "upstream_commit": os.environ["REBOT_UPSTREAM_COMMIT"],
    "source_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
    "patch_sha256": hashlib.sha256((root / "patches/motorbridge-0.5.6-param-notify.patch").read_bytes()).hexdigest(),
    "library": str(library),
    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
    "hardware_validation": "not performed by this build script",
}
manifest = library.with_name("rebot_build.json")
manifest.write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
PY
