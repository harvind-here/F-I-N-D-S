import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
import pandas as pd
import logging


# Check GPU availability and set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU instead.")

# Load dataset
data = pd.read_csv('WELFake_Dataset.csv')

# Preprocessing
data['text'] = data['title'] + " " + data['text']  # Combine Title and Text columns
data['text'] = data['text'].str.lower()  # Clean the text

# Ensure the labels are 0 for fake and 1 for real
data['label'] = data['label'].apply(lambda x: 0 if x == 'fake' else 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Convert all elements to strings
X_train = X_train.astype(str)
X_test = X_test.astype(str)

# Tokenization
print("Initializing BERT model for training...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)  # Reduced max_length
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)  # Reduced max_length

# Dataset preparation
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FakeNewsDataset(train_encodings, y_train.tolist())
test_dataset = FakeNewsDataset(test_encodings, y_test.tolist())

# Model training
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,  # Reduced epochs
    per_device_train_batch_size=16,  # Increased batch size
    per_device_eval_batch_size=16,  # Increased batch size
    warmup_steps=100,  # Reduced warmup steps
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy="steps",  # Use eval_strategy instead of evaluation_strategy
    save_strategy="steps",  # Ensure the save strategy matches the evaluation strategy
    save_steps=200,
    eval_steps=200,
    logging_steps=100,
    load_best_model_at_end=True,
    fp16=True,
    dataloader_pin_memory=True
)

# Initialize Trainer with logging enabled
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=None,  # Disable default metrics calculation
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]  # Reduced patience
)

# Enable logging
logging.basicConfig(level=logging.INFO)

# Train the model
print("Training BERT model...")
trainer.train()

# Access training logs
train_logs = trainer.state.log_history  # This will contain the logs generated during training

# Evaluate the model
print("Evaluating the model...")
results = trainer.evaluate()
print(results)

# Save the model and tokenizer to your directory
model.save_pretrained('add your directory path here')
tokenizer.save_pretrained('add your directory path here')

# Load and use the model for inference
model = BertForSequenceClassification.from_pretrained('add your directory path here').to(device)
tokenizer = BertTokenizer.from_pretrained('add your directory path here')

# Example of making a prediction
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

# Test the prediction
print(predict("sample news here"))
