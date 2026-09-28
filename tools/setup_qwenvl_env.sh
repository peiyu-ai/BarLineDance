#!/usr/bin/env bash
#
# Fetch Qwen2.5-VL from the team's OSS model store into third_party/QwenVL
# (gitignored: a runtime dependency, not project source).
#
# Why OSS and not HuggingFace: from this network hf.co and its mirror sustain
# ~0.2 MB/s and modelscope.cn refuses connections outright, so a 16.5 GB repo
# is a ten-hour download.  The team's OSS bucket is the intended home for large
# models anyway; this script just makes retrieving them one command.
#
# Credentials come from the STS token mounts the pod already carries.  Only
# the writable STS mount can write; any of them can read, and this script only
# reads.  Tokens rotate, so they are read at run time and never cached here.
#
# Qwen2.5-VL captions dance segments for the paper's M3 in-group re-clustering
# (the "w/ LLM" row of Tab. 2).  It is a *substitution* for the paper's
# Gemini-2.5-Pro and must be recorded as one wherever its output is used, in
# the same way S3D standing in for I3D is recorded.
#
# Usage:
#   bash tools/setup_qwenvl_env.sh [MODEL_NAME]
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Qwen3-VL-30B-A3B is the default because of what this task needs, not because
# it is the largest available: it is a mixture of experts with ~3B active
# parameters, so it runs at roughly 8B-dense speed while describing video at
# 30B quality.  Captioning ~95k segments makes throughput a first-class
# constraint, and the 72B dense alternative is ~137 GB and an order of
# magnitude slower for quality that still has to be *measured*, not assumed.
MODEL="${1:-Qwen3-VL-30B-A3B-Instruct}"
DEST="${REPO_ROOT}/third_party/QwenVL/${MODEL}"
OSS_PREFIX="${OSS_ASSET_PREFIX:-oss://example-bucket/example/prefix/}models/${MODEL}/"
ENDPOINT="${OSS_ENDPOINT:?set OSS_ENDPOINT}"
STS_DIRS=(${OSS_STS_DIRS:-})

# The processor configs matter as much as the weights: Qwen2.5-VL cannot build
# its vision preprocessing without them, and a weights-only copy fails at load
# time with an error that points at the model rather than the missing file.
REQUIRED=(
  model.safetensors.index.json
  config.json
  generation_config.json
  preprocessor_config.json
  chat_template.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
)

command -v ossutil64 >/dev/null || { echo "error: ossutil64 not on PATH" >&2; exit 1; }

CONFIG="$(mktemp)"
trap 'rm -f "${CONFIG}"' EXIT

write_config() {
  python3 - "$1" "${CONFIG}" "${ENDPOINT}" <<'PY'
import json
import sys

token_dir, config_path, endpoint = sys.argv[1], sys.argv[2], sys.argv[3]
with open(token_dir + "/..data/token", encoding="utf-8") as handle:
    token = json.load(handle)
with open(config_path, "w", encoding="utf-8") as handle:
    handle.write(
        "[Credentials]\nlanguage=EN\nendpoint={}\naccessKeyID={}\n"
        "accessKeySecret={}\nstsToken={}\n".format(
            endpoint, token["access_key_id"], token["access_key_secret"],
            token["security_token"])
    )
PY
}

echo "== find a credential that can read the model store =="
CREDENTIAL=""
for dir in "${STS_DIRS[@]}"; do
  [[ -r "${dir}/..data/token" ]] || continue
  write_config "${dir}"
  if ossutil64 --config-file "${CONFIG}" ls "${OSS_PREFIX}" >/dev/null 2>&1; then
    CREDENTIAL="${dir}"
    echo "   using ${dir}"
    break
  fi
done
if [[ -z "${CREDENTIAL}" ]]; then
  echo "error: no STS token could list ${OSS_PREFIX}" >&2
  echo "  the model may not be uploaded yet, or the tokens have expired" >&2
  exit 1
fi

echo "== sync =="
mkdir -p "${DEST}"
ossutil64 --config-file "${CONFIG}" sync "${OSS_PREFIX}" "${DEST}/" --update

# The upstream mirror stores each model under a revision directory ("main/"),
# so the weights land one level below DEST.  Resolve the root by finding
# config.json rather than assuming either layout, because guessing wrong fails
# later at load time with an error that points at the model, not the path.
MODEL_ROOT="${DEST}"
if [[ ! -s "${DEST}/config.json" ]]; then
  found="$(find "${DEST}" -mindepth 2 -maxdepth 2 -name config.json -printf '%h\n' | head -1)"
  if [[ -n "${found}" ]]; then
    MODEL_ROOT="${found}"
    echo "   model root resolved to ${MODEL_ROOT#${REPO_ROOT}/}"
  fi
fi

echo "== verify =="
missing=0
for name in "${REQUIRED[@]}"; do
  if [[ ! -s "${MODEL_ROOT}/${name}" ]]; then
    echo "   MISSING ${name}"
    missing=$((missing + 1))
  fi
done
if (( missing )); then
  echo "error: ${missing} required file(s) absent; the OSS copy is incomplete" >&2
  exit 1
fi

# Shard sizes are checked against the index, because a truncated shard is the
# failure this repo has already hit once: three "present" files that were a few
# megabytes each, counted as complete because only their existence was tested.
python3 - "${MODEL_ROOT}" <<'PY'
import json
import pathlib
import sys

dest = pathlib.Path(sys.argv[1])
index = json.loads((dest / "model.safetensors.index.json").read_text())
expected_total = index["metadata"]["total_size"]
shards = sorted(set(index["weight_map"].values()))
actual = sum((dest / shard).stat().st_size for shard in shards)
print("   shards: {}  bytes: {:.2f} GB (index says {:.2f} GB)".format(
    len(shards), actual / 2**30, expected_total / 2**30))
# safetensors files carry an 8-byte header plus JSON metadata, so the on-disk
# total runs slightly above the tensor total the index reports.
if actual < expected_total:
    raise SystemExit("error: shards total less than the index declares; truncated copy")
print("   size check passed")
PY

echo
echo "Done: ${MODEL_ROOT}"
echo "Pass this path as --model."
echo "Remember: this VLM substitutes for the paper's Gemini-2.5-Pro in M3."
echo "Record that substitution wherever its captions are used."
