print("importing...")
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
print("setting up...")
# Set up the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
print("reading data...")
# Load the training data
with open('traindata.txt', 'r', encoding='utf-8') as f:
    train_data = f.read()
print("encoding...")
# Encode the training data
encodings = tokenizer.encode_plus(train_data, return_tensors='pt')
print("setting up parameters (setting up adam 1/3)...")
# Set up the training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
print("setting up parameters (setting up cuda/mps 2/3)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'mps') # Comment out this line for CPU
print("setting up parameters (piping to cuda/mps 3/3)...")
model.to(device)
print("initializing training...")
# Fine-tune the model
model.train()
print("starting training...")
for i in range(100):
    print("starting epoch " + str(i) + "...")
    input_ids = encodings['input_ids'].to(device)
    labels = input_ids.to(device)
    outputs = model(input_ids, labels=labels)
    loss = outputs[0]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("finished epoch " + str(i) + "...")

# Save the fine-tuned model to disk
print("saving model...")
model.save_pretrained('finetuned_gpt2')
print("saved model...")

