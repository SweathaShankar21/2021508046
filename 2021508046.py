from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2") model = GPT2LMHeadModel.from_pretrained("gpt2")
def generate_response(prompt, max_length=50):
input_ids = tokenizer.encode(prompt, return_tensors="pt") response_ids = model.generate(input_ids, max_length=max_length,
num_return_sequences=1)
response = tokenizer.decode(response_ids[0], skip_special_tokens=True) return response
# Example usage while True:
user_input = input("You: ")
if user_input.lower() == "exit": break
response = generate_response(user_input) print("ChatGPT:", response)
