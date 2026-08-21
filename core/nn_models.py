import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, DistilBertModel, DistilBertTokenizerFast
from sklearn.base import BaseEstimator, TransformerMixin

class RNNForCategoryClassification(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        rnn_units: int,
        dropout_rate: float,
        max_sequence_length: int,
        recurrent_dropout_rate: float = 0.0,
        pad_token_id: int = 0,
        num_classes: int = 4,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=rnn_units,
            batch_first=True,
            bidirectional=True,
            dropout=recurrent_dropout_rate
        )

        self.fc = nn.Linear(2 * rnn_units, num_classes)
        self.pad_token_id = pad_token_id
        self.max_sequence_length = max_sequence_length

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (batch_size, seq_len)
        x = self.embedding(X) # (batch, seq, emb)
        x = self.dropout(x)

        output, _ = self.rnn(x) # (batch, seq, 2 * rnn_units)

        mask = (X != self.pad_token_id).unsqueeze(-1) # (batch, seq, 1)
        
        masked_output = output * mask.float()
        sum_output = masked_output.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        mean_output = sum_output / lengths

        logits = self.fc(mean_output) # (batch, num_classes)
        return logits


class CNNForCategoryClassification(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        kernel_size: int,
        dropout_rate: float,
        max_sequence_length: int,
        pad_token_id: int = 0,
        num_classes: int = 4,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=pad_token_id
        )
        self.dropout = nn.Dropout(dropout_rate)
        
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size
        )

        self.relu = nn.ReLU()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Linear(num_filters, num_classes)

        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (batch_size, seq_len)
        x = self.embedding(X)               # (batch, seq_len, emb_dim)
        x = self.dropout(x)
        
        x = x.permute(0, 2, 1)              # (batch, emb_dim, seq_len)
        x = self.relu(self.conv(x))         # (batch, num_filters, new_seq_len)
        x = self.global_max_pool(x)         # (batch, num_filters, 1)
        x = x.squeeze(-1)                   # (batch, num_filters)
        
        logits = self.fc(x)                 # (batch, num_classes)
        return logits


class LSTMForCategoryClassification(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        lstm_units: int,
        dropout_rate: float,
        recurrent_dropout_rate: float = 0.0,
        max_sequence_length: int = 512,
        pad_token_id: int = 0,
        num_classes: int = 4,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_token_id
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True,
            dropout=recurrent_dropout_rate
        )

        self.fc = nn.Linear(2 * lstm_units, num_classes)
        self.pad_token_id = pad_token_id
        self.max_sequence_length = max_sequence_length

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (batch_size, seq_len)
        x = self.embedding(X) # (batch, seq, emb)
        x = self.dropout(x)

        output, _ = self.lstm(x) # (batch, seq, 2 * lstm_units)

        mask = (X != self.pad_token_id).unsqueeze(-1) # (batch, seq, 1)
        masked_output = output * mask.float()
        sum_output = masked_output.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        mean_output = sum_output / lengths # (batch, 2 * lstm_units)

        logits = self.fc(mean_output) # (batch, num_classes)
        return logits


class GloveCNNForCategoryClassification(nn.Module):
    def __init__(
        self,
        glove_matrix: np.ndarray,
        num_filters: int,
        kernel_size: int,
        num_classes: int = 4,
        freeze_embeddings: bool = True,
        dropout_rate: float = 0.3,
        pad_token_id: int = 0
    ):
        super().__init__()

        vocab_size, embedding_dim = glove_matrix.shape

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.embedding.weight = nn.Parameter(
            torch.tensor(glove_matrix, dtype=torch.float32)
        )
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.dropout = nn.Dropout(dropout_rate)
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size
        )
        self.relu = nn.ReLU()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (batch_size, seq_len)
        x = self.embedding(X) # (batch, seq_len, emb_dim)
        x = self.dropout(x)
        
        x = x.transpose(1, 2) # (batch, emb_dim, seq_len)
        x = self.relu(self.conv(x)) # (batch, num_filters, new_len)
        x = self.global_max_pool(x) # (batch, num_filters, 1)
        x = x.squeeze(2) # (batch, num_filters)
        
        logits = self.fc(x) # (batch, num_classes)
        return logits


class DistilBertForCategoryClassification(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 4,
        dropout_rate: float = 0.3,
        freeze_bert: bool = False,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.pad_token_id = pad_token_id
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        hidden_size = self.bert.config.dim # 768
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        attention_mask = (X != self.pad_token_id).long()
        
        outputs = self.bert(
            input_ids=X,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :] # [CLS]
        cls_output = self.dropout(cls_output)
        logits = self.fc(cls_output)
        
        return logits


class BertTokenizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="distilbert-base-uncased", max_len=128):
        self.model_name = model_name
        self.max_len = max_len
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            texts = X.tolist()
        else:
            texts = list(X)
        
        texts = [str(text) for text in texts]

        encodings = self.tokenizer(
            texts,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        
        return np.array(encodings["input_ids"], dtype=np.int64)