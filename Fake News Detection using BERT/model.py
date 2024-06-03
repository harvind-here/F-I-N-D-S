import torch
from extraction import extract_text_and_images
from transformers import BertTokenizer, BertForSequenceClassification

# Check GPU availability and set device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the model and tokenizer from the saved directory
model = BertForSequenceClassification.from_pretrained('Your directory here').to(device)
tokenizer = BertTokenizer.from_pretrained('your directory here')

# Function to predict whether the news is fake or real
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return "Fake News" if predictions.item() == 0 else "Real News"

# Test the prediction function
article_content = extract_text_and_images("paste news article url")
prediction = predict(article_content)
print(f"The prediction for the given text is: {prediction}")
