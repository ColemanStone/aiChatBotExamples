import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM

# Load pre-trained model and tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cpu")  # Use CPU
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def generate_response(prompt, model, tokenizer, max_length=10):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(device)

    outputs = model.generate(
        inputs, attention_mask=attention_mask, max_length=max_length,
        num_return_sequences=1, pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Chatbot: Hello! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Chatbot: Goodbye!")
        break

    response = generate_response(user_input, model, tokenizer)
    print(f"Chatbot: {response}")