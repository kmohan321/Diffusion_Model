from transformers import CLIPTokenizer, CLIPTextModel
from transformers import AutoTokenizer,AutoModel
import torch
import torch.nn as nn

class CLIPTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state

class BERT(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device
        model_name = "huawei-noah/TinyBERT_General_4L_312D"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

        for param in self.model.parameters():
            param.requires_grad = False
    def forward(self,texts):
        inputs = self.tokenizer(texts,padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state