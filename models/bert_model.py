import torch
from transformers import BertTokenizer, BertModel

class BERTModel:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
    
    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state

if __name__ == "__main__":
    bert_model = BERTModel()
    sample_text = "Patient has a history of hypertension and diabetes."
    embeddings = bert_model.encode(sample_text)
    print(embeddings.shape)  # Example output shape
