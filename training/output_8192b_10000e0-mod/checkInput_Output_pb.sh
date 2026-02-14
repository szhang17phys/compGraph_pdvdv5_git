#!/usr/bin/env bash

python <<'PYEOF'
from tensorflow.core.protobuf import saved_model_pb2

pb = saved_model_pb2.SavedModel()
with open("saved_model.pb", "rb") as f:
    pb.ParseFromString(f.read())

# Select MetaGraph: prefer 'serve'
mg = None
for m in pb.meta_graphs:
    if "serve" in m.meta_info_def.tags:
        mg = m
        break
if mg is None:
    mg = pb.meta_graphs[0]

sigs = mg.signature_def
sig = sigs["serving_default"] if "serving_default" in sigs else next(iter(sigs.values()))

def shape_to_str(ts):
    if not ts.tensor_shape.dim:
        return "()"
    dims = []
    for d in ts.tensor_shape.dim:
        dims.append(str(d.size) if d.size != -1 else "None")
    return "(" + ", ".join(dims) + ")"

print("\n=== INPUTS (SignatureDef) ===")
for k, v in sig.inputs.items():
    print(f"{k:20s} shape={shape_to_str(v)}, dtype={v.dtype}, name={v.name}")

print("\n=== OUTPUTS (SignatureDef) ===")
for k, v in sig.outputs.items():
    print(f"{k:20s} shape={shape_to_str(v)}, dtype={v.dtype}, name={v.name}")
PYEOF
