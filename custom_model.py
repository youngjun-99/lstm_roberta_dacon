import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss

from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel 
from transformers.modeling_outputs import SequenceClassifierOutput

from loss import FocalLoss, softXEnt

class RobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config)
        classifier_dropout = config.hidden_dropout_prob
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        lstm_output, (last_hidden, c) = self.lstm(sequence_output)
        cat_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        logits = self.classifier(cat_hidden)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                if len(labels.size()) == len(logits.size()):
                    loss = softXEnt(logits.view(-1, self.num_labels), labels)
                else:
                    loss_fct = FocalLoss(gamma=2) #data imbalance
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )