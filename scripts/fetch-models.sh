#!/usr/bin/env bash
# Downloads every model Minastir uses into <data>/models (default ./data/models), skipping files
# that are already there. About 3.5 GB in total.
#
# The server downloads most of these by itself the first time it needs them; running this up
# front just avoids waiting mid-session. The pose model, the describer and Segment Anything are
# only fetched here.
#
#   scripts/fetch-models.sh            # everything
#   scripts/fetch-models.sh --core     # only what calibration and the live engine need
set -euo pipefail

DATA="${MINASTIR_DATA:-./data}"
M="$DATA/models"
mkdir -p "$M"
HF=https://huggingface.co

get() { # <url> <dest>
  if [[ -s "$2" ]]; then echo "  have  $2"; return; fi
  echo "  fetch $2"
  mkdir -p "$(dirname "$2")"
  curl -fL --retry 3 --progress-bar -o "$2.part" "$1"
  mv "$2.part" "$2"
}

echo "Core models → $M"
get "$HF/Ruicheng/moge-2-vitl-normal-onnx/resolve/main/model.onnx"                          "$M/moge2_vitl_normal.onnx"
get "$HF/Xenova/segformer-b5-finetuned-ade-640-640/resolve/main/onnx/model.onnx"            "$M/segformer_b5_ade.onnx"
get "$HF/onnx-community/dfine_x_obj2coco-ONNX/resolve/main/onnx/model.onnx"                 "$M/dfine_x_obj2coco.onnx"
get "$HF/onnx-community/dinov2-small/resolve/main/onnx/model.onnx"                          "$M/dinov2_small.onnx"
get "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v3.0/raco_aliked_lightglue_pipeline_k2048.onnx" \
                                                                                             "$M/aliked_lightglue_k2048.onnx"
for f in model.onnx model.onnx_data; do
  get "$HF/onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX/resolve/main/onnx/$f"        "$M/dinov3_vitb16/$f"
done
get "$HF/onnx-community/grounding-dino-tiny-ONNX/resolve/main/onnx/model.onnx"              "$M/grounding_dino_tiny/model.onnx"
get "$HF/onnx-community/grounding-dino-tiny-ONNX/resolve/main/tokenizer.json"               "$M/grounding_dino_tiny/tokenizer.json"

[[ "${1:-}" == "--core" ]] && exit 0

echo "Optional models"
# Body keypoints (tells children from adults by height).
if [[ ! -s "$M/rtmpose_m_body7.onnx" ]]; then
  tmp=$(mktemp -d)
  get "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip" "$tmp/rtmpose.zip"
  unzip -q "$tmp/rtmpose.zip" -d "$tmp/x"
  mv "$(find "$tmp/x" -name '*.onnx' | head -1)" "$M/rtmpose_m_body7.onnx"
  rm -rf "$tmp"
else
  echo "  have  $M/rtmpose_m_body7.onnx"
fi

# Segment Anything: traces roofs on the site photo when you click one (falls back to colour).
if [[ ! -s "$M/sam_vit_b.encoder.onnx" ]]; then
  tmp=$(mktemp -d)
  get "$HF/vietanhdev/segment-anything-onnx-models/resolve/main/sam_vit_b_01ec64.zip" "$tmp/sam.zip"
  unzip -q "$tmp/sam.zip" -d "$tmp/x"
  mv "$(find "$tmp/x" -name '*encoder*.onnx' | head -1)" "$M/sam_vit_b.encoder.onnx"
  mv "$(find "$tmp/x" -name '*decoder*.onnx' | head -1)" "$M/sam_vit_b.decoder.onnx"
  rm -rf "$tmp"
else
  echo "  have  $M/sam_vit_b.encoder.onnx"
fi

# Describer: a small vision-language model that names what a tracked object is (~2 GB).
Q="$HF/onnx-community/Qwen3-VL-2B-Instruct-ONNX/resolve/main"
for f in tokenizer.json \
         onnx/vision_encoder_q4f16.onnx onnx/vision_encoder_q4f16.onnx_data \
         onnx/embed_tokens_q4f16.onnx onnx/embed_tokens_q4f16.onnx_data \
         onnx/decoder_model_merged_q4f16.onnx onnx/decoder_model_merged_q4f16.onnx_data; do
  get "$Q/$f" "$M/qwen3vl_2b/$f"
done
echo "Done."
