import json
import torch
from transformers import AlbertTokenizer

def load_data_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def preprocess_data(data):
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    input_ids = []
    attention_masks = []
    labels = []
    for item in data:
        text = item['text']
        label = item['label']
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoded.input_ids)
        attention_masks.append(encoded.attention_mask)
        labels.append(label)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels
