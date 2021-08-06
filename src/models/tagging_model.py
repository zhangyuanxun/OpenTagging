import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import LayerNorm
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertConfig,
    BertPreTrainedModel,
)
from .layers import Attention
from .crf import CRF
#from .new_crf import CRF


class Tagging(BertPreTrainedModel):
    def __init__(self, config, label_list, device):
        super(Tagging, self).__init__(config)

        # get bert model
        self.bert = BertModel(config)
        self.config = config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.context_lstm = nn.LSTM(input_size=self.config.hidden_size,
                                    hidden_size=self.config.hidden_size // 2,
                                    batch_first=True,
                                    bidirectional=True)
        self.attribute_lstm = nn.LSTM(input_size=self.config.hidden_size,
                                      hidden_size=self.config.hidden_size // 2,
                                      batch_first=True,
                                      bidirectional=True)

        self.ln = LayerNorm(self.config.hidden_size * 2)
        self.attention = Attention()
        self.classifier = nn.Linear(self.config.hidden_size * 2, len(label_list))
        label2id = {k: i for i, k in enumerate(label_list)}
        # self.crf = CRF(num_tags=len(label_list), tag2id=label2id, batch_first=True)
        self.crf = CRF(tagset_size=len(label_list), tag_dictionary=label2id, device=device, is_bert=True)
        self.init_weights()

    def forward(self, context_input_ids=None, context_input_mask=None, context_type_ids=None,
                context_input_len=None, attribute_input_ids=None, attribute_input_mask=None,
                attribute_type_ids=None, attribute_input_len=None, label_ids=None):

        bert_output_context = self.bert(context_input_ids, context_type_ids, context_input_mask)
        bert_output_attribute = self.bert(attribute_input_ids, attribute_type_ids, attribute_input_mask)

        # passing bert to lstm
        context_output, _ = self.context_lstm(bert_output_context[0])
        _, attribute_hidden = self.attribute_lstm(bert_output_attribute[0])

        # using last hidden state of attribute lstm
        attribute_output = torch.cat([attribute_hidden[0][-2], attribute_hidden[0][-1]], dim=-1)

        # get attention output
        attention_output = self.attention(context_output, attribute_output)
        outputs = torch.cat([context_output, attention_output], dim=-1)

        outputs = self.ln(outputs)
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        loss = None
        if label_ids is not None:
            # loss = self.crf.calculate_loss(emissions=logits, tags=label_ids, mask=context_input_mask)
            loss = self.crf.calculate_loss(logits, tag_list=label_ids, lengths=context_input_len)
        return {'loss': loss, 'logits': logits}

