# by @fakerybakery on GitHub - github.com/fakerybakery/hf-transformers-finetune
# copyright 2023 mrfakename - all rights reserved
# commercial use allowed, but please consider adding my username to your model card if you publish your model. this is not required but would be appreciated :) thank you
# Grab data from http://www.manythings.org/anki/
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class TranslationDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []
        self.tokenizer = tokenizer

        # Read the TSV file
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Parse the TSV file and tokenize the input and target sentences
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                source_text = parts[0]
                target_text = parts[1]
                self.data.append((source_text, target_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source_text, target_text = self.data[index]
        source_tokens = self.tokenizer.encode(source_text, padding='max_length', truncation=True, max_length=512)
        target_tokens = self.tokenizer.encode(target_text, padding='max_length', truncation=True, max_length=512)
        return torch.tensor(source_tokens), torch.tensor(target_tokens)

def train_translation_model(train_file, model_name, output_dir, num_epochs=10, batch_size=4, max_length=512):
    # Load the T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Create the translation dataset
    train_dataset = TranslationDataset(train_file, tokenizer)

    # Create the data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Set the device to GPU if available
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)

    # Set the optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0

        # Set up TQDM for progress tracking
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for batch in progress_bar:
            source_tokens, target_tokens = batch
            source_tokens = source_tokens.to(device)
            target_tokens = target_tokens.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=source_tokens, labels=target_tokens)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update TQDM progress bar description with current loss
            progress_bar.set_postfix({"Loss": loss.item()})

        # Print the average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")

        # Save the trained model after each epoch
        model.save_pretrained(output_dir + '_epoch' + str(epoch + 1))

    # Save the final trained model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# Usage example
train_file = 'data.txt'  # Path to your TSV file
model_name = 't5-base'  # Pretrained T5 model
output_dir = 'trained_model'  # Output directory for the trained model

train_translation_model(train_file, model_name, output_dir, num_epochs=10, batch_size=4, max_length=512)
