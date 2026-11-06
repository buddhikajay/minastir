"""Export FastReID's VeRi-776 vehicle re-identification model (SBS, ResNet50-IBN, Apache-2.0)
to ONNX for Minastir's appearance fingerprints.

  cd tools/model-export && .venv/bin/python export_vehicle_reid.py

Needs: vendor/fast-reid (git clone https://github.com/JDAI-CV/fast-reid) and
weights/veri_sbs_R50-ibn.pth (FastReID release v0.1.1).

Input:  images [N, 3, 256, 256], RGB 0–255 (the model normalises internally).
Output: features [N, 2048] (not normalised; L2-normalise before cosine similarity).
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor", "fast-reid"))

from fastreid.config import get_cfg  # noqa: E402
from fastreid.modeling.meta_arch import build_model  # noqa: E402
from fastreid.utils.checkpoint import Checkpointer  # noqa: E402

cfg = get_cfg()
cfg.merge_from_file(os.path.join(HERE, "vendor", "fast-reid", "configs", "VeRi", "sbs_R50-ibn.yml"))
cfg.MODEL.BACKBONE.PRETRAIN = False  # weights come from the checkpoint below
cfg.MODEL.DEVICE = "cpu"
cfg.freeze()

model = build_model(cfg)
Checkpointer(model).load(os.path.join(HERE, "weights", "veri_sbs_R50-ibn.pth"))
model.eval()

os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
out = os.path.join(HERE, "out", "vehicle_reid_veri_r50ibn.onnx")
dummy = torch.randint(0, 255, (2, 3, 256, 256)).float()
with torch.no_grad():
    ref = model(dummy)
    torch.onnx.export(
        model,
        dummy,
        out,
        input_names=["images"],
        output_names=["features"],
        dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
print("exported", out, "features", tuple(ref.shape))
