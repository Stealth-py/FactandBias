from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List
import torch
import torch.nn as nn
import numpy as np
import os
import transformers


class ModelInference:
    def __init__(self, model_path: str, tokenizer_path: str, quantize: bool, use_gpu: bool) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() and use_gpu else torch.device("cpu")
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        if quantize:
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model = self.model.to(self.device).eval()

    def predict(self, batch: List[str]) -> np.array:
        with torch.no_grad():
            inputs = self.tokenizer(batch, truncation=True, padding=True, return_tensors="pt").to(self.device)
            #labels = torch.tensor([1]).unsqueeze(0)
            outputs = self.model(**inputs)
            res = nn.Softmax(dim = -1)(outputs.logits).cpu().numpy().tolist()
        return res
