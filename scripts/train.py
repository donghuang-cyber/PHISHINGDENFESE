import torch
from transformers import AlbertForSequenceClassification, AlbertTokenizer

class FGM(object):
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.1, emb_name='albert.embeddings.word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad.sign() * norm
                    param.data.add_(r_at)

    def restore(self, emb_name='albert.embeddings.word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def load_albert_model():
    model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)
    return model

def load_tokenizer():
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    return tokenizer
