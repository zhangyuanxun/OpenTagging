import torch
import os
from torch.utils.data import TensorDataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from data_processor import TaggingProcessor, InputFeatures
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler


def convert_examples_to_features(examples, label_list, tokenizer, max_seq_length, max_attr_length):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for (idx, example) in enumerate(examples):
        if idx % 10000 == 0:
            print("Converting examples to features: {} of {}".format(idx, len(examples)))

        context_features = tokenizer(example.context, is_split_into_words=True,
                                     max_length=max_seq_length, padding="max_length", truncation=True)
        label_ids = []

        # align label with word token
        word_ids = context_features.word_ids(batch_index=0)
        for i in range(len(word_ids)):
            wid = word_ids[i]
            iid = context_features['input_ids'][i]
            if iid == tokenizer.cls_token_id:
                label_ids.append(label_map['[CLS]'])
            elif iid == tokenizer.sep_token_id:
                label_ids.append(label_map['[SEP]'])
            elif wid is None:
                label_ids.append(label_map['O'])
            else:
                # deal with sub-words problem
                if i > 0 and wid == word_ids[i - 1] and (label_ids[-1] == label_map['B-a'] or
                                                         label_ids[-1] == label_map['I-a']):
                    label_ids.append(label_map['I-a'])
                else:
                    label_ids.append(label_map[example.labels[wid]])

        context_features['label_ids'] = label_ids
        attribute_features = tokenizer(example.attributes, is_split_into_words=True,
                                       max_length=max_attr_length, padding="max_length", truncation=True)

        assert len(context_features['input_ids']) == max_seq_length
        assert len(context_features['attention_mask']) == max_seq_length
        assert len(context_features['label_ids']) == max_seq_length
        assert len(context_features['token_type_ids']) == max_seq_length
        assert len(attribute_features['input_ids']) == max_attr_length
        assert len(attribute_features['attention_mask']) == max_attr_length
        assert len(attribute_features['token_type_ids']) == max_attr_length

        features.append(InputFeatures(context_input_ids=context_features['input_ids'],
                                      context_input_mask=context_features['attention_mask'],
                                      context_input_len=max_seq_length,
                                      context_type_ids=context_features['token_type_ids'],
                                      attribute_input_ids=attribute_features['input_ids'],
                                      attribute_input_mask=attribute_features['attention_mask'],
                                      attribute_input_len=max_attr_length,
                                      attribute_type_ids=attribute_features['token_type_ids'],
                                      label_ids=context_features['label_ids']
                                      ))

    return features


def load_examples(args, tokenizer, data_type):
    if args.local_rank not in (-1, 0) and data_type == "train":
        torch.distributed.barrier()

    processor = TaggingProcessor()
    if data_type == 'train' and args.debug:
        examples = processor.get_debug_examples(args.data_dir)
    elif data_type == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif data_type == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    elif data_type == 'test' and args.debug:
        examples = processor.get_debug_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    label_list = processor.get_labels()

    print("Creating features from the dataset...")
    features = convert_examples_to_features(examples, label_list, tokenizer, args.max_seq_length, args.max_attr_length)

    if args.local_rank == 0 and data_type == "train":
        torch.distributed.barrier()

    def collate_fn(batch):
        def convert_to_tensor(key):
            if isinstance(key, str):
                tensors = [torch.tensor(getattr(o[1], key), dtype=torch.long) for o in batch]
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
                   attribute_input_len=convert_to_tensor('attribute_input_len'),
                   label_ids=convert_to_tensor('label_ids'))

        return ret

    if data_type == "train":
        sampler = RandomSampler(features) if args.local_rank == -1 else DistributedSampler(features)
        dataloader = DataLoader(list(enumerate(features)), sampler=sampler, batch_size=args.train_batch_size,
                                collate_fn=collate_fn)
    else:
        dataloader = DataLoader(list(enumerate(features)), batch_size=args.eval_batch_size, collate_fn=collate_fn)

    return dataloader, label_list
