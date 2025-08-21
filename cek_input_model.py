import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

saved_model_path = "saved_model"

# Load SavedModel langsung
loaded = tf.saved_model.load(saved_model_path)

# Tampilkan signature
signature = loaded.signatures["serving_default"]

print("========== INPUT SIGNATURE ==========")
for name, tensor_spec in signature.structured_input_signature[1].items():
    print(f"Nama Input: {name}")
    print(f"  Shape    : {tensor_spec.shape}")
    print(f"  Dtype    : {tensor_spec.dtype}")
print("=====================================")
