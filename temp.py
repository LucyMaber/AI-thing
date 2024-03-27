import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

class DistilBertClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DistilBertClassifier, self).__init__()
        
        # Load DistilBERT model and tokenizer
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Classification head
        self.fc = nn.Linear(self.distilbert.config.hidden_size, num_classes)
    
    def update_num_classes(self, num_classes):
        pass
    
    def forward(self, input_text):
        # Tokenize input text
        input_ids = self.tokenizer(input_text, return_tensors='pt')['input_ids']
        
        # Get DistilBERT model output
        output = self.distilbert(input_ids)
        
        # Extract the [CLS] token representation (last hidden state)
        cls_token_rep = output.last_hidden_state[:, 0, :]
        
        # Pass the [CLS] token representation through the classification head
        logits = self.fc(cls_token_rep)
        
        return logits

# Example usage
num_classes = 10  # Specify the number of classes for your classification task
model = DistilBertClassifier(num_classes)

# Dummy input text
input_text = "This is an example sentence for classification."

# Forward pass
logits = model(input_text)

# Print the logits
print(logits)
