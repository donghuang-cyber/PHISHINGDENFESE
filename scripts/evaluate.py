import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from albert_model import load_albert_model, load_tokenizer
from utils import load_data_from_json, preprocess_data
import time

# Load model and tokenizer
tokenizer = load_tokenizer()
model = load_albert_model()
model.to('cuda')

def evaluate_on_test_set(test_json_file):
    test_data = load_data_from_json(test_json_file)
    test_input_ids, test_attention_masks, test_labels = preprocess_data(test_data)

    test_input_ids = test_input_ids.to('cuda')
    test_attention_masks = test_attention_masks.to('cuda')
    test_labels = test_labels.to('cuda')

    model.eval()
    all_preds, all_labels = [], []
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(len(test_input_ids)):
            inputs = test_input_ids[i].unsqueeze(0)
            attention_masks = test_attention_masks[i].unsqueeze(0)
            labels = test_labels[i].unsqueeze(0)

            outputs = model(inputs, attention_mask=attention_masks)
            _, predicted = torch.max(outputs.logits, 1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    end_time = time.time()
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    test_accuracy = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds)
    test_recall = recall_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds)

    print(f'Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}')
    print(f'Time taken for testing: {end_time - start_time:.2f} seconds')
