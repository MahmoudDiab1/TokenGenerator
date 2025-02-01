from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1. Load a tokenizer and a model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer("Hi everyone, I'm Diab, and I'm studying Generative AI", return_tensors = "pt")
print(inputs["input_ids"])
