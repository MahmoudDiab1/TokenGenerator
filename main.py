import pandas
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1. Load a tokenizer and a model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer("Hi everyone, I'm Diab, and I'm studying Generative AI", return_tensors = "pt")
print(inputs["input_ids"])

# Step 2. Examine the tokenization¶

def show_tokenization(input):
   token_tuples = [(id, tokenizer.decode(id))for id in inputs["input_ids"][0]]
   table = pandas.DataFrame(token_tuples, columns=("id", "token"))
   return table

table = show_tokenization(inputs)
print(table)