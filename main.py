import pandas
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1. Load a tokenizer and a model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = AutoModelForCausalLM.from_pretrained("gpt2")
inputs = tokenizer("Hi everyone, I'm Diab, and I'm studying Generative AI", return_tensors = "pt")
print(inputs["input_ids"])

# Step 2. Examine the tokenization¶
def show_tokenization(input):
   token_tuples = [(id, tokenizer.decode(id))for id in inputs["input_ids"][0]]
   table = pandas.DataFrame(token_tuples, columns=("id", "token"))
   return table

table = show_tokenization(inputs)
print(table)


# Step 3. Calculate the probability of the next token
with torch.no_grad():
    logites = model(**inputs).logits[:,-1,:]
    props = torch.nn.functional.softmax(logites[0], dim=-1)


def show_next_token_choices(probabilities, top_n=5):
    return pandas.DataFrame(
        [
            (id, tokenizer.decode(id), p.item())
            for id, p in enumerate(probabilities)
            if p.item()
        ],
        columns=["id", "token", "p"],
    ).sort_values("p", ascending=False)[:top_n]


next_words = show_next_token_choices(props)
print(next_words)