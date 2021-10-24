import torch

from transformers import (
    AutoTokenizer,
)
from transformers.models.bert.modeling_bert import (
    BertConfig,
)
from config import *
from models.tagging_model import Tagging
from transformers import WEIGHTS_NAME
import nltk
from torch.utils.data import DataLoader
from utils import *


class InferModel:
    def __init__(self, device, bert_model, label_list, model_dir, max_seq_length, max_attr_length, batch_size):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.config = BertConfig.from_pretrained(bert_model)
        self.max_seq_length = max_seq_length
        self.max_attr_length = max_attr_length
        self.label2id = {k: i for i, k in enumerate(label_list)}
        self.id2label = {i: k for i, k in enumerate(label_list)}
        self.tagging_model = Tagging.from_pretrained(bert_model, config=self.config,
                                                label_list=label_list, device=self.device)
        self.tagging_model.load_state_dict(torch.load(os.path.join(model_dir, WEIGHTS_NAME), map_location="cpu"))
        self.tagging_model.eval()
        self.tagging_model.to(self.device)
        self.batch_size = batch_size

    def convert_features(self, context, attribute):
        # split into sentences
        context_sentences = nltk.sent_tokenize(context)
        features = []

        for context in context_sentences:
            context_tokens = nltk.word_tokenize(context)
            context_features = self.tokenizer(context_tokens, is_split_into_words=True,
                                              max_length=self.max_seq_length, padding="max_length", truncation=True)

            attribute_tokens = nltk.word_tokenize(attribute)
            attribute_features = self.tokenizer(attribute_tokens, is_split_into_words=True,
                                                max_length=self.max_attr_length, padding="max_length", truncation=True)

            features.append({'context_input_ids': context_features['input_ids'],
                             'context_input_mask': context_features['attention_mask'],
                             'context_type_ids': context_features['token_type_ids'],
                             'context_input_len': self.max_seq_length,
                             'attribute_input_ids': attribute_features['input_ids'],
                             'attribute_input_mask': attribute_features['attention_mask'],
                             'attribute_type_ids': attribute_features['token_type_ids'],
                             'attribute_input_len': self.max_attr_length})

        def collate_fn(batch):
            def convert_to_tensor(key):
                if isinstance(key, str):
                    tensors = [torch.tensor(o[1][key], dtype=torch.long) for o in batch]
                else:
                    tensors = [torch.tensor(o, dtype=torch.long) for o in key]

                return torch.stack(tensors)

            ret = dict(context_input_ids=convert_to_tensor('context_input_ids'),
                       context_input_mask=convert_to_tensor('context_input_mask'),
                       context_type_ids=convert_to_tensor('context_type_ids'),
                       context_input_len=convert_to_tensor('context_input_len'),
                       attribute_input_ids=convert_to_tensor('attribute_input_ids'),
                       attribute_input_mask=convert_to_tensor('attribute_input_mask'),
                       attribute_type_ids=convert_to_tensor('attribute_type_ids'),
                       attribute_input_len=convert_to_tensor('attribute_input_len'))

            return ret

        dataloader = DataLoader(list(enumerate(features)), batch_size=self.batch_size, collate_fn=collate_fn)
        return dataloader

    def predict(self, context, attribute):
        dataloader = self.convert_features(context, attribute)
        results = dict()

        # for inference, there is only one item in the dataloader with idx = 0

        batch = next(iter(dataloader))
        inputs = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            output = self.tagging_model(**inputs)
            preds, _ = self.tagging_model.crf.obtain_labels(output['logits'], self.id2label,
                                                            inputs['context_input_len'])

        context_input_ids = inputs['context_input_ids']
        idx = 0
        pred_labels = get_entities(preds[idx], self.id2label, 'bio')
        context_tokens = self.tokenizer.convert_ids_to_tokens(context_input_ids[idx])
        results['context'] = self.tokenizer.convert_tokens_to_string(context_tokens)
        results['tokens'] = context_tokens
        results['attribute'] = attribute
        labels = []
        if pred_labels:
            for j in range(len(pred_labels)):
                pred_tokens = self.tokenizer.convert_tokens_to_string((context_tokens[pred_labels[j][1]: pred_labels[j][2] + 1]))
                labels.append({'value': pred_tokens, 'position': [pred_labels[j][1], pred_labels[j][2]]})

        results['labels'] = labels
        return results

        # for batch in dataloader:
        #     inputs = {k: v.to(self.device) for k, v in batch.items()}
        #
        #     with torch.no_grad():
        #         output = self.tagging_model(**inputs)
        #         preds, _ = self.tagging_model.crf.obtain_labels(output['logits'], self.id2label,
        #                                                         inputs['context_input_len'])
        #
        #     context_input_ids = inputs['context_input_ids']
        #     for i in range(len(preds)):
        #         pred_labels = get_entities(preds[i], self.id2label, 'bio')
        #         if not pred_labels:
        #             continue
        #
        #         context_tokens = self.tokenizer.convert_ids_to_tokens(context_input_ids[i])
        #
        #         for j in range(len(pred_labels)):
        #             pred_tokens = self.tokenizer.convert_tokens_to_string((context_tokens[pred_labels[j][1]: pred_labels[j][2] + 1]))
        #             results.append({'value': pred_tokens, 'position': [pred_labels[j][1], pred_labels[j][2]],
        #                             'tokens': context_tokens,
        #                             'context': self.tokenizer.convert_tokens_to_string(context_tokens)})
        #
        #     return results


model = InferModel(device=DEVICE, bert_model=BERT_MODEl,
                   label_list=LABEL_LIST, model_dir=MODEL_DIR,
                   max_seq_length=MAX_SEQ_LEN, max_attr_length=MAX_ATTR_LEN,
                   batch_size=BATCH_SIZE)


def get_model():
    return model
