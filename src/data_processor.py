import os
import json
import copy
import nltk
import re


class InputExample(object):
    def __init__(self, guid, context, attributes, labels):
        self.guid = guid
        self.context = context
        self.attributes = attributes
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(self, context_input_ids, context_input_mask, context_input_len,
                 attribute_input_ids, attribute_input_mask, attribute_input_len,
                 context_type_ids, attribute_type_ids, label_ids):
        self.context_input_ids = context_input_ids
        self.context_input_mask = context_input_mask
        self.context_input_len = context_input_len
        self.context_type_ids = context_type_ids
        self.attribute_input_ids = attribute_input_ids
        self.attribute_input_mask = attribute_input_mask
        self.attribute_input_len = attribute_input_len
        self.attribute_type_ids = attribute_type_ids
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class TaggingProcessor(object):
    def get_debug_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "opentags.debug")), 'train')

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "opentags.train")), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "opentags.dev")), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "opentags.test")), 'test')

    def get_labels(self):
        return ['B', 'I', 'O']

    def _word_tokenize(self, sentence):
        return nltk.word_tokenize(sentence)

    def _get_value_offset(self, context_words, value):
        value_words = self._word_tokenize(value)
        value_no_space = ''.join(value_words)

        for i in range(len(context_words) - len(value_words) + 1):
            windows = context_words[i: i + len(value_words)]
            if value_words == windows:
                return i, i + len(value_words) - 1

            windows = ''.join(windows)
            if value_no_space in windows:
                return i, i + len(value_words) - 1

        raise Exception('Attribute Value Parsing Error, context: {}, '
                        'value: {}, sentence: {}, value tokens: {}'.format(context_words, value,
                                                                           " ".join(context_words), value_words))

    def _read_data(self, input_file):
        data = []
        with open(input_file, 'r') as fp:
            for line in fp.readlines():
                line = line.strip().split('\01')

                context = line[0].strip()
                attribute = line[1].strip()
                value = line[2].strip()

                context_words = self._word_tokenize(context)
                attribute_words = self._word_tokenize(attribute)
                labels = ['O'] * len(context_words)

                if value != 'NULL':
                    assert value in context
                    i, j = self._get_value_offset(context_words, value)
                    if i == j:
                        labels[i] = 'B'
                    else:
                        labels[i] = 'B'
                        labels[i + 1: j] = ['I'] * (j - i)

                data.append({"context": context_words, "labels": labels, 'attributes': attribute_words})

        return data

    def _create_examples(self, data, data_type):
        examples = []
        for (i, line) in enumerate(data):
            guid = "%s-%s" % (data_type, i)
            context = line['context']
            attribute = line['attributes']
            labels = line['labels']
            examples.append(InputExample(guid=guid, context=context, attributes=attribute, labels=labels))
        return examples

