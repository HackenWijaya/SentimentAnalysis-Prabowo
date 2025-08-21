import tensorflow as tf
from transformers import AutoTokenizer

# Load tokenizer dan model SavedModel
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
tfsmlayer = tf.saved_model.load("saved_model")

# Tokenisasi input
text = "Saya senang sekali"
inputs = tokenizer(text, return_tensors="tf", max_length=128, truncation=True, padding="max_length")

# Panggil model dengan dict (bukan positional args!)
result = tfsmlayer({
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
    "token_type_ids": tf.zeros_like(inputs["input_ids"])
})

# Lihat hasil logits
print(result["logits"])
