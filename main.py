# mps-cuda-gpt2-finetune by @fakerybakery on GitHub - github.com/fakerybakery/mps-cuda-gpt2-finetune
# copyright 2023 mrfakename - all rights reserved
# commercial use allowed, but please consider adding my username to your model card if you publish your model. this is not required but would be appreciated :) thank you
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Set up the tokenizer and model
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load the training data
with open('traindata.txt', 'r', encoding='utf-8') as f:
    train_data = f.read()

# Split the training data into smaller chunks
max_length = model.config.n_positions
train_chunks = [train_data[i:i+max_length] for i in range(0, len(train_data), max_length)]

# Set up the training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model.to(device)

# Fine-tune the model on each training chunk
model.train()
for i, chunk in enumerate(train_chunks):
    print(f"Processing chunk {i+1}/{len(train_chunks)}...")
    encodings = tokenizer.encode_plus(chunk, return_tensors='pt', max_length=max_length)
    input_ids = encodings['input_ids'].to(device)
    labels = input_ids.clone().detach()
    labels[labels == tokenizer.pad_token_id] = -100

    outputs = model(input_ids, labels=labels)
    loss = outputs[0]
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print(f"Processed chunk {i+1}/{len(train_chunks)}. Loss: {loss.item()}")

# Save the fine-tuned model to disk
model.save_pretrained('finetuned_gpt2')
