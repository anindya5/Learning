from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, tokenizer_name=model_name, max_length=max_length):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def load_data(self, file_path):
        """Loads and splits the dataset."""
        import pandas as pd
        data = pd.read_csv(file_path)
        texts = data['text']
        labels = data['label']
        return train_test_split(texts, labels, test_size=0.2, random_state=42)

    def preprocess_data(self, texts):
        """Tokenizes and encodes texts using BERT tokenizer."""
        encodings = self.tokenizer(
            list(texts), truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
        )
        return encodings