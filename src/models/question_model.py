_author_ = "Rohit Patil"
import torch.nn as nn
from transformers import BertForSequenceClassification


class BertClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=num_labels,
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )

    def forward(self, ids, mask, token_type_ids):
        o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        return o2
