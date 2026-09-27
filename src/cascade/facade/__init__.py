"""Exterior facade screening: tile crack classifier (ONNX), heatmap, top-K tiles, grader glue.

AI screening for a human inspector, not a QEWI/FISP finding. Submodules:
- tiles: window geometry and preprocessing (numpy + PIL only; importable in the training env)
- heatmap: onnxruntime tile classifier, probability grid, overlay, top-K with NMS
- review: glue to the existing VLM grader (facade_ll11 rubric) and cascade.measure
"""
