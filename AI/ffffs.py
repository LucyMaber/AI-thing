from transformers import AlbertConfig, AlbertModel

albert_base_configuration = AlbertConfig(
    hidden_size=768,
    num_attention_heads=12,
    intermediate_size=3072,
)
model = AlbertModel(albert_base_configuration)
