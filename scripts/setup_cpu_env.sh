#!/bin/bash
set -euo pipefail
# Rebuild .pip_cpu2/site (pure-Python peft + pyyaml) for the x86 CPU Apptainer image.
# The CPU image itself (e.g. audinet/SAE-scrubbing/apptainer/containers/pipeline-cpu.sif)
# already provides torch>=2.6 and transformers; this only adds the two wheels needed by
# the LoRA baseline. No GPU, no pip install into the (read-only) image.

SITE="$(cd "$(dirname "$0")/.." && pwd)/.pip_cpu2/site"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$SITE"

wheel_url() {
    python3 - "$1" "$2" <<'PY'
import json, sys, urllib.request
pkg, ver = sys.argv[1], sys.argv[2]
d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json"))
for u in d["releases"].get(ver, []):
    fn = u["filename"]
    if fn.endswith(".whl") and ("py3-none-any" in fn or "cp313" in fn):
        print(u["url"]); sys.exit(0)
raise SystemExit(f"no wheel for {pkg}=={ver}")
PY
}

curl -sSL -o "$TMP/peft.whl" "$(wheel_url peft 0.22.0)"
curl -sSL -o "$TMP/pyyaml.whl" "$(wheel_url pyyaml 6.0.3)"
unzip -oq "$TMP/peft.whl" -d "$SITE"
unzip -oq "$TMP/pyyaml.whl" -d "$SITE"
echo "Unpacked peft + pyyaml into $SITE"