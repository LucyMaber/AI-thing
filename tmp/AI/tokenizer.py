import os
import random
from transformers import AutoTokenizer
import torch
import torch.optim as optim
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import sys, traceback


class DistilBertClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DistilBertClassifier, self).__init__()

        # Load DistilBERT model and tokenizer
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        # Classification head
        self.fc = nn.Linear(self.distilbert.config.hidden_size, num_classes)

    def update_num_classes(self, num_classes):
        self.fc = nn.Linear(self.distilbert.config.hidden_size, num_classes)

    def forward(self, input_text):
        # Tokenize input text
        input_ids = self.tokenizer(
            input_text, padding=True, truncation=True, return_tensors="pt")["input_ids"]

        # Get DistilBERT model output
        output = self.distilbert(input_ids)

        # Extract the [CLS] token representation (last hidden state)
        cls_token_rep = output.last_hidden_state[:, 0, :]

        # Pass the [CLS] token representation through the classification head
        logits = self.fc(cls_token_rep)

        return logits


# Choose a pre-trained model name or path
model_name = "distilbert-base-uncased"


def list2Dict(lst):
    return {
        "id2label": {lst[i]: i for i in range(len(lst))},
        "label2id": {i: lst[i] for i in range(len(lst))},
    }

import os
import torch

def make_classifier(labels, file_path):
    # Test if file exists
    print("file_path", file_path, os.path.isfile(file_path))
    if os.path.isfile(file_path):
        try:
            checkpoint = torch.load(file_path)
            model = DistilBertClassifier(len(labels))  # Assuming your DistilBertClassifier has an appropriate constructor
            model.load_state_dict(checkpoint, strict=False) 
            return model
        except Exception as e:
            print("Error loading model",file_path,e)
            return DistilBertClassifier(len(labels))
    num_labels = len(labels)
    model = DistilBertClassifier(num_labels)
    return model



# def make_classifier(labels):
#     c = list2Dict(labels)
#     num_labels = len(labels)
#     model = DistilBertClassifier(num_labels)
#     return model


tokenizer = AutoTokenizer.from_pretrained(model_name)
def decode_indices(values, indices):
    decoded_tensor = [0] * len(values)
    for index in indices:
        value_index = values.index(index)
        decoded_tensor[value_index] = 1
    return torch.tensor([decoded_tensor], dtype=torch.float32)

def train_model_classifier(
    model: DistilBertClassifier,
    texts: str,
    labels: [str],
    master_label: [str],
    learning_rate=5e-5,
):
    if type(texts) == str:
        pass
    else:
        print("texts is not a string", type(texts))
    try:
        num_params_to_train = 30
        # Get a list of all parameters in the model
        all_params = list(model.parameters())

        # Shuffle the parameters randomly
        random.shuffle(all_params)

        # Freeze the parameters except for the selected ones
        for param in all_params[num_params_to_train:]:
            param.requires_grad = False

        # Enable gradients for the selected parameters
        for param in all_params[:num_params_to_train]:
            param.requires_grad = True
        
        model.update_num_classes(len(master_label))
        # Convert labels to numerical format
        label_ids = decode_indices(master_label,labels)
    
        # Forward pass
        outputs = model(texts)  # Pass input_ids

        # Define CrossEntropyLoss
        loss_function = nn.CrossEntropyLoss()
         
        # Calculate the loss
        loss = loss_function(outputs, label_ids)  # Pass outputs directly

        # Assuming you have an optimizer defined
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("train_model_classifier loss:", loss.item())

    except Exception as e:
        #  line number of error
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = traceback.extract_tb(exc_tb)[-1][2]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("train_model_classifier", e)
