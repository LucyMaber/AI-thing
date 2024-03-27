from transformers import XLNetConfig, XLNetModel

# Initializing a XLNet configuration
configuration = XLNetConfig()

# Initializing a model (with random weights) from the configuration
model = XLNetModel(configuration)

# Accessing the model configuration
configuration = model.config