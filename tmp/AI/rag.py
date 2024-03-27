import torch
from transformers import LlamaModel, LlamaConfig
from transformers import LlamaTokenizerFast
from transformers import AutoTokenizer, LlamaForCausalLM

prompt = """
On January 30, 2024 PPI’s representatives Ohad Shem Tov and Ohad Bar Siman Tov attended the United Nations Headquarters in New York for the Partnership Forum of the Economic and Social Council (ECOSOC). The forum addressed the critical theme of “Reinforcing the 2030 Agenda and eradicating poverty in times of multiple crises: the effective delivery of sustainable, resilient, and innovative solutions.”

https://ecosoc.un.org/en/events/2024/ecosoc-partnership-forum

The 2024 ECOSOC Partnership Forum placed a special emphasis on the following Sustainable Development Goals (SDGs): SDG 1 (No Poverty), SDG2 (Zero Hunger), SDG13 (Climate Action), SDG16 (Peace and Justice), and SDG17 (Partnership for the Goals). The forum aimed to exchange new ideas to advance the 2030 Agenda. Ohad and Ohad had the opportunity to actively participate in sessions, interacting with global leaders. They tried to make a speech within the main ECOSOC hall, but unfortunately they ran out of time before reaching their turn.

 

Despite their ill fated attempt to speak in the main hall, they did succeed in presenting at a side event of the conference. Ohad and Ohad spoke about the ongoing situation in Israel, where they both come from. Side events provide a platform for more intimate discussions and give speakers more time to discuss their ideas. Statements made at the main hall are limited to just a few minutes.

 

The 2024 ECOSOC Partnership Forum will be instrumental in shaping the July 2024 High Level Political Forum (HLPF), where Ohad, Ohad, and other PPI representatives should attend in the summer. The next main event for PPI at UNHQ will be the Youth Forum in April. We are making arrangements now to have the PPI Board member Owen Richardson make a statement at this conference.

https://ecosoc.un.org/en/events/2024/2024-youth-forum

Having Pirate Parties International representatives actively participate in this global dialogue is extremely important, because we are making ourselves known to the world and are able to interact with other NGOs. Ohad and Ohad improved the narrative around effective partnerships between Pirates and other ideological movements. We thank them for their participation.

We also call on other Pirates around the world to attend UN activities in NY, Geneva, Vienna, and around the world. It is not always so easy to get accepted into events, so please coordinate with us well in advance of your intended visit.
"""


class LLMModel():
    def __init__(self, config):
        device = torch.device("cpu")
        configuration = LlamaConfig(
            vocab_size=32000,
            hidden_size=256*4,   # Reduced hidden size
            intermediate_size=5504,  # Reduced intermediate size
            num_hidden_layers=32, 
            num_attention_heads=32,
            max_position_embeddings=2048,
        )
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
        self.model = LlamaModel(configuration)
        self.model.to(device)
        self.model.eval()
    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


