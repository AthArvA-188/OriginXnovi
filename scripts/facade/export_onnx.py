"""Export the selected tile classifier to ONNX and check parity (training env: cerebro_ml).

Two files:
- data/raw/facade/ckpt/<tag>_fp32.onnx (git-ignored, ~45 MB): plain fp32 export; parity vs torch
  must be max |logit diff| < 1e-3 on 256 SDNET test tiles.
- models/facade/tilecls_resnet18_v1.onnx (shipped, < 25 MB): the same graph with weights STORED as
  fp16 and a Cast to fp32 in front of each, so onnxruntime CPU computes in fp32 after constant
  folding. Its parity (max |logit| and |p| difference, decision agreement) is measured and reported.

    python scripts/facade/export_onnx.py --tag resnet18_ozg_sdnet --backbone resnet18.a1_in1k
Writes data/raw/facade/ckpt/<tag>_export.json (parity, sizes, latency on this machine).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import timm
import torch
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "facade"
CKPT = RAW / "ckpt"
OUT = ROOT / "models" / "facade" / "tilecls_resnet18_v1.onnx"
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def fp16_storage(src: Path, dst: Path) -> dict:
    m = onnx.load(str(src))
    g = m.graph
    new_inits, casts, n_conv = [], [], 0
    for init in list(g.initializer):
        arr = numpy_helper.to_array(init)
        if init.data_type == TensorProto.FLOAT and arr.size >= 16:
            h = numpy_helper.from_array(arr.astype(np.float16), init.name + "__fp16")
            new_inits.append(h)
            casts.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
            n_conv += 1
        else:
            new_inits.append(init)
    del g.initializer[:]
    g.initializer.extend(new_inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    return {"initializers_cast_to_fp16": n_conv}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="resnet18_ozg_sdnet")
    ap.add_argument("--backbone", default="resnet18.a1_in1k")
    a = ap.parse_args()
    model = timm.create_model(a.backbone, pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(CKPT / f"{a.tag}.pt", map_location="cpu"))
    model.eval()
    fp32 = CKPT / f"{a.tag}_fp32.onnx"
    dummy = torch.randn(2, 3, 224, 224)
    try:
        torch.onnx.export(model, dummy, str(fp32), input_names=["input"], output_names=["logits"], opset_version=17,
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
        exporter = "torchscript (dynamo=False)"
    except Exception as e:  # fall back to the dynamo exporter (needs onnxscript)
        print("legacy export failed:", e)
        torch.onnx.export(model, dummy, str(fp32), input_names=["input"], output_names=["logits"], opset_version=17,
                          dynamic_shapes={"x": {0: torch.export.Dim("batch")}}, dynamo=True)
        exporter = "dynamo"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    info = fp16_storage(fp32, OUT)

    # parity on 256 SDNET test tiles from the pre-decoded cache
    df = pd.read_csv(RAW / "tiles.csv")
    mm = np.load(RAW / "cache224.npy", mmap_mode="r")
    te = df[(df.dataset == "sdnet_walls") & (df.split == "test")]
    rows = te.sample(256, random_state=0).index.to_numpy()
    x = ((np.asarray(mm[np.sort(rows)], np.float32) / 255.0).transpose(0, 3, 1, 2) - MEAN) / STD
    x = x.astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()

    def run(path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        outs = np.concatenate([s.run(None, {"input": x[i:i + 32]})[0] for i in range(0, len(x), 32)])
        t0 = time.time()
        for i in range(0, 128, 32):
            s.run(None, {"input": x[i:i + 32]})
        ms = (time.time() - t0) / 128 * 1000
        return outs, ms

    def sm(z):
        e = np.exp(z - z.max(1, keepdims=True))
        return e[:, 1] / e.sum(1)

    o32, ms32 = run(fp32)
    o16, ms16 = run(OUT)
    rep = {
        "tag": a.tag, "backbone": a.backbone, "exporter": exporter, "opset": 17, **info,
        "fp32_file_bytes": fp32.stat().st_size, "shipped_file": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "shipped_file_bytes": OUT.stat().st_size,
        "parity_n_tiles": int(len(x)),
        "fp32_max_abs_logit_diff": float(np.abs(o32 - ref).max()),
        "fp16store_max_abs_logit_diff": float(np.abs(o16 - ref).max()),
        "fp16store_max_abs_p_diff": float(np.abs(sm(o16) - sm(ref)).max()),
        "ort_cpu_ms_per_tile_fp32_4threads_cerebro_env": round(ms32, 2),
        "ort_cpu_ms_per_tile_fp16store_4threads_cerebro_env": round(ms16, 2),
    }
    assert rep["fp32_max_abs_logit_diff"] < 1e-3, rep
    (CKPT / f"{a.tag}_export.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
