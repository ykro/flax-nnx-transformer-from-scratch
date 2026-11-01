#!/usr/bin/env bash
# Ephemeral Cloud TPU v5e (v5litepod-1) / Vertex AI L4 training runner for Project: ai-experiments-487722
# Provisions an ephemeral TPU VM, runs the Flax NNX Transformer verification suite, and guarantees teardown.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-ai-experiments-487722}"
ZONE="${ZONE:-us-central1-a}"
TPU_NAME="flax-nnx-transformer-$(date +%s)"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v5litepod-1}"
RUNTIME_VERSION="${RUNTIME_VERSION:-v2-alpha-tpuv5-lite}"

gcloud_cmd() {
  CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }datacloud.antigravity" \
    gcloud "$@"
}

cleanup() {
  echo "[cleanup] Deleting ephemeral TPU VM ${TPU_NAME} in ${ZONE}..."
  gcloud_cmd compute tpus tpu-vm delete "${TPU_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --quiet || true
  echo "[cleanup] Teardown complete."
}
trap cleanup EXIT INT TERM

echo "[setup] Creating ephemeral Cloud TPU VM ${TPU_NAME} (${ACCELERATOR_TYPE}) in ${PROJECT_ID}/${ZONE}..."
gcloud_cmd compute tpus tpu-vm create "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --accelerator-type="${ACCELERATOR_TYPE}" \
  --version="${RUNTIME_VERSION}" \
  --spot

echo "[sync] Copying Flax NNX Transformer repository to ${TPU_NAME}..."
gcloud_cmd compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --command="mkdir -p ~/flax-nnx-transformer/benchmarks"
gcloud_cmd compute tpus tpu-vm scp --recurse \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  ./experiments ./verify_transformer.py ./pyproject.toml \
  "${TPU_NAME}:~/flax-nnx-transformer/"

echo "[exec] Installing dependencies and running all 5 experiments + Outside Judge on TPU..."
gcloud_cmd compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --command="cd ~/flax-nnx-transformer && pip install -q 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html flax optax torch numpy && python3 verify_transformer.py"

echo "[sync] Fetching benchmark artifacts from ${TPU_NAME}..."
gcloud_cmd compute tpus tpu-vm scp --recurse \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  "${TPU_NAME}:~/flax-nnx-transformer/benchmarks/*" \
  ./benchmarks/
