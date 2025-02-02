import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

text = "Hello everyone! my name is Diab and I'm learning generative"
print(f" Input text: {text}")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
inputs = tokenizer(text, return_tensors = "pt")

def show_next_n_tokens(probs, top_n):
    dataFrame = pd.DataFrame(
        [(id, tokenizer.decode([id]), p.item()) for id, p in enumerate(probs) if p.item() > 0],
        columns=("id", "token", "probability")
    ).sort_values("probability", ascending=False)[:top_n]
    return dataFrame

with torch.no_grad():
    logits = model(**inputs).logits[:,-1,:]
    probabilities = torch.nn.functional.softmax(logits[0], dim=-1)

next_n_tokens_table = show_next_n_tokens(probabilities, 5)
next_token_id = torch.argmax(probabilities).item() # token with max score
next_token_value = tokenizer.decode(next_token_id)
generated_result = text+next_token_value
print(f" Generated result: {generated_result}")
output_by_generated_func = model.generate(**inputs, max_length = 30, pad_token_id = tokenizer.eos_token_id)
generated_result = tokenizer.decode(output_by_generated_func[0])
print(f" Generated result(generated func): {generated_result}")