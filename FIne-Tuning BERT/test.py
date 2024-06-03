import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Check GPU availability and set device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the model and tokenizer from the saved directory
model = BertForSequenceClassification.from_pretrained('your directory path here').to(device)
tokenizer = BertTokenizer.from_pretrained('your directory path here')

# Function to predict whether the news is fake or real
def predict(text):
    model.eval()  # Set the model to evaluation mode
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    probability = torch.softmax(logits, dim=-1)
    return "Fake News" if predictions.item() == 0 else "Real News", probability

# Test the prediction function with a list of sample texts
sample_texts = [
    "UNBELIEVABLE! OBAMA’S ATTORNEY GENERAL SAYS MOST CHARLOTTE RIOTERS WERE “PEACEFUL” PROTESTERS…In Her...Now, most of the demonstrators gathered last night were exercising their constitutional and protect...",
    "Breaking news! This information is not true and is intended to mislead people.",
    "A dozen politically active pastors came here for a private dinner Friday night to hear a conversion ...",
    "Aliens have landed on Earth, according to some fake sources."
]

for text in sample_texts:
    prediction, probability = predict(text)
    print(f"Text: {text}\nPrediction: {prediction}\nProbability: {probability}\n")
