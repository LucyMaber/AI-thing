import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import traceback
import sys

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class LikesEstimator(nn.Module):
    def __init__(self, text_transformer, input_dim, hidden_dim, output_dim, num_layers):
        super(LikesEstimator, self).__init__()
        self.text_transformer = text_transformer
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()  # Activation function

    def forward(self, time_step, follows, followers, text_input):
        # Assuming text_input is already processed
        text_output = self.text_transformer(text_input)
        x = torch.cat(
            (time_step, follows, followers, text_output.last_hidden_state[:, 0]), dim=1
        )
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x


def make_estimator(file_path, hidden_dim=256, output_dim=10, num_layers=10):
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    if os.path.isfile(file_path):
        try:
            state_dict, = torch.load(file_path)
            print("checkpoint", state_dict,)
            likes_estimator = LikesEstimator(
                bert_model,
                input_dim=3 + 768,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
            )  # Assuming appropriate constructor
            likes_estimator.load_state_dict(state_dict['model_state_dict'], strict=False)
            return likes_estimator
        except Exception as e:
            #  line number of error
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = traceback.extract_tb(exc_tb)[-1][2]
            print(exc_type, fname, exc_tb.tb_lineno)
            print("make_estimator", e)
        # Initialize LikesEstimator model
    likes_estimator = LikesEstimator(
        bert_model,
        input_dim=3 + 768,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
    )
    return likes_estimator


# Define your data preprocessing function
def preprocess_estimator_data(
    time_steps, num_favourites, num_followers, text_inputs, output_targets
):
    # Tokenize text_inputs using BERT tokenizer
    encoded_inputs = tokenizer(
        text_inputs, padding=True, truncation=True, return_tensors="pt"
    )
    text_input_ids = encoded_inputs["input_ids"]
    text_attention_masks = encoded_inputs["attention_mask"]
    # Convert to PyTorch tensors
    time_steps_tensor = torch.tensor(time_steps, dtype=torch.float32)
    num_followers_tensor = torch.tensor(num_followers, dtype=torch.float32)
    num_favourites_tensor = torch.tensor(num_favourites, dtype=torch.float32)
    output_targets = torch.tensor(output_targets, dtype=torch.float32)
    return (
        time_steps_tensor,
        num_followers_tensor,
        num_favourites_tensor,
        text_input_ids,
        text_attention_masks,
        output_targets,
    )


def train_model_likes_estimator(model: LikesEstimator, batch, learning_rate=5e-5):
    try:
        # Extract batch data
        time_steps, num_followers, num_favourites, text_input_ids, _, targets = batch

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

        # Forward pass
        outputs = model(time_steps, num_followers, num_favourites, text_input_ids)

        # Define CrossEntropyLoss
        loss_function = nn.CrossEntropyLoss()

        # Convert targets to tensor
        targets = targets.long()

        # Calculate the loss
        loss = loss_function(outputs, targets)

        # Assuming you have an optimizer defined
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("train_model_estimator loss:", loss.item())

    except Exception as e:
        #  line number of error
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = traceback.extract_tb(exc_tb)[-1][2]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("train_model_estimator", e)
