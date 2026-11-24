# Shared settings for start.sh, stop.sh and scripts/*.sh. A setting's value comes from the first of these that sets it:
#   1. the environment: `PORT=9000 ./start.sh`, `PREPARE=1 ./start.sh restart`
#   2. ./.env (KEY=value lines, read, never run)
# Settings marked PROVISIONAL follow the engine build in patches/ and change when it is re-measured.
if [[ -f .env ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < .env
fi
unset _line _key _value

# The checkpoint, mounted read-only at /models/ink. Default: upstream thinkingmachines/Inkling-2-Mini at a pinned
# revision, fetched into MODEL_DIR by scripts/fetch.sh when it is missing. RERISKED=1 (alias DERISKED=1): the local
# Inkling-2-Mini-Derisked folder (Blackfrost's derivative, never downloaded). SHARD_SHA256 is the target shard's
# (the hub's LFS SHA-256 upstream, the derivative README's otherwise); both share the byte-identical MTP module.
RERISKED="${RERISKED:-${DERISKED:-0}}"
[[ "$RERISKED" =~ ^[01]$ ]] || { echo "RERISKED is 0 or 1, not $RERISKED" >&2; exit 1; }
if [[ "$RERISKED" == 1 ]]; then
  MODEL_DIR="${MODEL_DIR:-$HOME/models/hf/Inkling-2-Mini-Derisked}"
  MODEL_LABEL="${MODEL_LABEL:-Inkling-2-Mini-Derisked}"
  SHARD_SHA256="${SHARD_SHA256:-b1156b5d53ddabf927a8e3f209eebc5da610d2b90ed8c14a6a945b78080bef21}"
  MODEL_FETCH=0
else
  MODEL_DIR="${MODEL_DIR:-$HOME/models/hf/Inkling-2-Mini}"
  MODEL_LABEL="${MODEL_LABEL:-Inkling-2-Mini}"
  SHARD_SHA256="${SHARD_SHA256:-65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1}"
  MODEL_FETCH="${MODEL_FETCH:-1}"   # scripts/prepare.sh runs scripts/fetch.sh when the folder is incomplete
fi
SHARD_NAME="model-00001-of-00001.safetensors"
MTP_NAME="mtp.safetensors"
MTP_SHA256="${MTP_SHA256:-7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2}"
MODEL_MOUNT=/models/ink
QUANT_LABEL="ModelOpt NVFP4 experts, MXFP8 layers 20-23, bf16 MTP"
CKPT_GIB=20
# What loading takes of MemAvailable: the engine reported "model 22.82 GiB" with no drafter (2026-11-22). The DSpark
# drafter adds its 0.6 GiB of weights, a 0.1 GiB folded head and its scratch; the vision tower (VISION=1) 0.6 GiB. PROVISIONAL.
WEIGHTS_GIB="${WEIGHTS_GIB:-25}"

# The engine: TensorFold's Zig engine at Jürgen Schmied's Inkling 2 Mini CUDA head (TensorFold PR #4441, "C4"),
# plus this recipe's patches/*.patch. The commit is reachable as refs/pull/4441/head on ashhart/TensorFold and as
# the branch inkling2-mini-c4 on jschmied/TensorFold; the fork is fetched because a PR ref is not a branch.
TF_REPO="${TF_REPO:-https://github.com/jschmied/TensorFold.git}"
TF_REF="${TF_REF:-dc2ca8a1785df6ee5c8489ee7cd18e29cf7b4568}"   # PR #4441 head (C4), on main 1587c8caee
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
# libtfgrammar.so (xgrammar: structured output, required tool calls) is built from the patched tree over xgrammar at
# TensorFold's pinned release, checked by commit. Patch 0013 turns xgrammar's own compile cache off (a v0.2.8 bug).
XGRAMMAR_REPO="${XGRAMMAR_REPO:-https://github.com/mlc-ai/xgrammar}"
XGRAMMAR_TAG=v0.2.8
XGRAMMAR_REF=97787376faee5ed8466cfad57c99855e4ce2f6aa
# -Dnative-fp-decode: the FP4/FP8 expert and projection kernels (qmmf, prompt16, fp8_experts) decode their codes
# with PTX ISA 9.2's native cvt instructions on sm_121a (nvcc 13.2+). NATIVE_FP_DECODE=0 builds the generic decode.
NATIVE_FP_DECODE="${NATIVE_FP_DECODE:-1}"
TF_TAG="c4-${TF_REF:0:7}"
IMAGE="${IMAGE:-tensorfold-inkling2:${TF_TAG}}"
image_hash() {
  ( cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    cat patches/*.patch 2>/dev/null
    sed -n "/<<'DOCKERFILE'/,/^DOCKERFILE\$/p" scripts/prepare.sh
    echo "$TF_REPO $TF_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE xgrammar-$XGRAMMAR_TAG-$XGRAMMAR_REF native-fp-$NATIVE_FP_DECODE"
  ) | sha256sum | cut -c1-12
}
CONTAINER_NAME="${CONTAINER_NAME:-inkling2-mini-tf}"

SERVED_NAME="${SERVED_NAME:-$MODEL_LABEL}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
CONTEXT="${CONTEXT:-262144}"
# 8 full windows at fp8 KV for both checkpoints. (The Derisked kit's vLLM command leaves its KV cache at "auto", bf16:
# KV_DTYPE=bf16 PARALLEL=4 serves that; fp8 is this deployment's choice.)
PARALLEL="${PARALLEL:-8}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
# DRAFTS=1 loads the checkpoint's own DSpark drafter (mtp.safetensors) with --drafter; 0 passes --no-drafts.
DRAFTS="${DRAFTS:-1}"
# DRAFTER=dflash2 drafts with incoai/Inkling-2-Mini-DFlash2 (patch 0018) from DFLASH2_DIR instead of DSpark.
# start.sh fetches it there when it is missing (scripts/fetch_dflash2.sh, ~1 GB).
DRAFTER="${DRAFTER:-dspark}"
DFLASH2_DIR="${DFLASH2_DIR:-$HOME/models/hf/Inkling-2-Mini-DFlash2}"
DFLASH2_MOUNT=/models/dflash2
DFLASH2_REPO=incoai/Inkling-2-Mini-DFlash2
DFLASH2_REVISION=4f9088ea66c21f2b99ed657d1941c306b03e84d2
DFLASH2_CONFIG_SHA256=d094e435eb939e01fca6297f637c5af35e3da4b552f8e1ab0a181cd278a1cefa
DFLASH2_SHA256="${DFLASH2_SHA256:-217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee}"
# VISION=1 loads the checkpoint's inkling2_vision tower (--vision, ~0.6 GiB of bf16 weights) for image inputs (PNG,
# JPEG, WebP). VISION_URLS=1 also lets image_url fetch http(s) URLs (data: URLs always work). VISION_MAX_IMAGES and
# VISION_IMAGE_TOKENS (an image's token budget, engine default 4096) are passed when set.
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-}"
VISION_IMAGE_TOKENS="${VISION_IMAGE_TOKENS:-}"
# The chat template's reasoning_effort: none, minimal, low, medium, high (its default, 0.9), xhigh, max, or a number in
# [0, 0.99]. "none" still reasons a little (the model card: minimum effort does not fully disable reasoning).
THINKING="${THINKING:-1}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
# generation_config.json: do_sample, temperature 1.0; the card names no top_p or top_k.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-0}"
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-9}"

# The target's attention caches: bf16 (exact, the default) or fp8 (`--kv-cache fp8`: e4m3 codes with a scale a position and
# key head, about half the bytes; lossy, so a reply can differ from the bf16 one). The drafter's own caches stay bf16.
KV_DTYPE="${KV_DTYPE:-fp8}"
# FP8 attention projections (patch 0012; lossy, the engine's default is off): half the attention weight bytes, decode
# about +22% at one stream. Teacher-forced top-1 agreement with bf16 98.26% (fp8 KV alone 98.52%).
# TF_INKLING2_ATTN_FP8=0 serves the exact bf16 projections.
export TF_INKLING2_ATTN_FP8="${TF_INKLING2_ATTN_FP8:-1}"
# Tensor-core prefill experts (patch 0015; lossy, the engine's default is off): prompt rows are quantized once a layer
# to NVFP4 (layers 0-19) or MXFP8 (20-23) for block-scaled MMA. Prefill +5% at 8k to +41% at 120k tokens; a 734-problem
# accuracy set scored 732 either way, teacher-forced top-1 agreement with the exact path 96.3%.
# TF_INKLING2_PREFILL_TC=0 serves the exact bf16 prefill.
export TF_INKLING2_PREFILL_TC="${TF_INKLING2_PREFILL_TC:-1}"
# The kit's vLLM serves reasoning in a `reasoning` field (its inkling reasoning parser); TensorFold names it
# reasoning_content. For Derisked both are sent (TF_REASONING_FIELD: reasoning_content, reasoning or both).
if [[ "$RERISKED" == 1 ]]; then export TF_REASONING_FIELD="${TF_REASONING_FIELD:-both}"; fi

# One stream's cache at a CONTEXT-token window. Per token: the 8 full visits' K and V (bf16: 8 x 1024 x 2 x 2 = 32,768 bytes;
# fp8: 8 x 2 x (1024 + 8 scales x 4) = 16,896 bytes); the 40 sliding visits keep fixed 1,536-row rings and 48 visits keep conv
# states (~0.24 GiB together a stream). The engine reported 1.24 GiB a stream at 32,768 tokens, which this reproduces.
# DSpark's global layer adds 4,096 bytes a token; DFlash2's rings are fixed (0.04 GiB a stream). PROVISIONAL.
kv_gib() {
  local ctx=${1:-$CONTEXT} per=32768
  [[ "$KV_DTYPE" == fp8 ]] && per=16896
  [[ "$ctx" =~ ^[0-9]+$ ]] || ctx=$CONTEXT
  [[ "$DRAFTS" == 1 && "$DRAFTER" == dspark ]] && per=$((per + 4096))
  echo $(( (per * ctx + 268435456 + 1073741823) / 1073741824 ))
}
# MemAvailable a start needs: weights, every stream's window, scratch slack and the floor. PROVISIONAL.
mem_need_gib() {
  local ctx=${1:-$CONTEXT}
  echo $(( WEIGHTS_GIB + PARALLEL * $(kv_gib "$ctx") + 4 + MEM_FLOOR_GIB ))
}
MEM_NEED_GIB="${MEM_NEED_GIB:-}"
MEM_FLOOR_GIB="${MEM_FLOOR_GIB:-$TENSORFOLD_MEMORY_RESERVE_GIB}"

KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-inkling2}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/inkling2-mini-tf}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/tensorfold-inkling2/logs}"
IMAGE_FREE_GB="${IMAGE_FREE_GB:-40}"
PREPARED_MARKER="$STATE_DIR/prepared"
VERIFIED_MARKER="$STATE_DIR/verified"

_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

# size and mtime of the two weight files: a change makes prepare.sh hash them again
weights_stamp() {
  local f out=""
  for f in "$MODEL_DIR/$SHARD_NAME" "$MODEL_DIR/$MTP_NAME"; do
    out+="$(stat -c '%s %Y' "$f" 2>/dev/null || echo missing) "
  done
  echo "$out"
}
prepared_state() {
  local label
  label=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing)
  echo "model=$MODEL_DIR $(weights_stamp)image=$label patches=$(image_hash)"
}
