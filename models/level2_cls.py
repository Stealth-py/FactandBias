from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from typing import List
import torch
import torch.nn as nn
import numpy as np
import os
import transformers
from sklearn.svm import SVC


class ModelInferenceForEmbeddings:
    def __init__(self, model_path: str, tokenizer_path: str, quantize: bool, use_gpu: bool) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() and use_gpu else torch.device("cpu")
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        if quantize:
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model = self.model.to(self.device).eval()

    def get_embeddings(self, batch: List[str]) -> np.array:
        with torch.no_grad():
            inputs = self.tokenizer(batch, truncation=True, padding=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
        return outputs.pooler_output

class Classifier:
    def __init__(self, cls_name: str = "svc") -> None:
        self.cls = None
        if cls_name == "svc":
            self.cls = SVC()
    
    def train(self, X, y):
        self.cls.fit(X, y)
    
    def predict(self, X):
        return self.cls.predict(X)
    
    def score(self, X, y):
        return self.cls.score(X, y)
