python - <<'EOF'
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2

pb = saved_model_pb2.SavedModel()
with open("saved_model.pb", "rb") as f:
    pb.ParseFromString(f.read())

print("SavedModel schema version:", pb.saved_model_schema_version)
for m in pb.meta_graphs:
    print("Tags:", list(m.meta_info_def.tags))
    print("TF version in MetaGraph:", m.meta_info_def.tensorflow_version)
    print("TF git version:", m.meta_info_def.tensorflow_git_version)
EOF
