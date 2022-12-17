import torch
import pandas as pd
from torch.utils.data import Dataset


class TypeDataset(Dataset):
    label2idx = {"사실형": 0, "추론형": 1, "대화형": 2, "예측형": 3}

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.features = self._create_features(data=self.data)
        self.labels = self._create_labels(data=self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.features.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def _create_features(self, data):
        features = self.tokenizer(
            data.문장.tolist(), padding=True, truncation=True, return_tensors="pt"
        )
        return features

    def _create_labels(self, data):
        labels = []
        for i in data.유형.tolist():
            labels.append(self.label2idx[i])
        return labels

class EmotionDataset(Dataset):
    label2idx = {"긍정": 0, "부정": 1, "미정": 2}

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.features = self._create_features(data=self.data)
        self.labels = self._create_labels(data=self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.features.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def _create_features(self, data):
        features = self.tokenizer(
            data.문장.tolist(), padding=True, truncation=True, return_tensors="pt"
        )
        return features

    def _create_labels(self, data):
        labels = []
        for i in data.극성.tolist():
            labels.append(self.label2idx[i])
        return labels

class TimeDataset(Dataset):
    label2idx = {"과거": 0, "현재": 1, "미래": 2}

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.features = self._create_features(data=self.data)
        self.labels = self._create_labels(data=self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.features.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def _create_features(self, data):
        features = self.tokenizer(
            data.문장.tolist(), padding=True, truncation=True, return_tensors="pt"
        )
        return features

    def _create_labels(self, data):
        labels = []
        for i in data.시제.tolist():
            labels.append(self.label2idx[i])
        return labels

class ConfidenceDataset(Dataset):
    label2idx = {"확실": 0, "불확실": 1}

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.features = self._create_features(data=self.data)
        self.labels = self._create_labels(data=self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.features.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def _create_features(self, data):
        features = self.tokenizer(
            data.문장.tolist(), padding=True, truncation=True, return_tensors="pt"
        )
        return features

    def _create_labels(self, data):
        labels = []
        for i in data.확실성.tolist():
            labels.append(self.label2idx[i])
        return labels

    
class TestDataset(Dataset):

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.features = self._create_features(data=self.data)
        self.labels = self._create_labels(data=self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.features.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def _create_features(self, data):
        features = self.tokenizer(
            data.문장.tolist(), padding=True, truncation=True, return_tensors="pt"
        )
        return features

    def _create_labels(self, data):
        labels = []
        for i in range(len(data)):
            labels.append(0)
        return labels