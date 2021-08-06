import os
import torch

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT_PATH, "../outputs")
BERT_MODEl = 'bert-base-uncased'
PORT_NUMBER = 8005
MAX_SEQ_LEN = 32
MAX_ATTR_LEN = 8
BATCH_SIZE = 16
LABEL_LIST = ['B-a', 'I-a', 'O', '[CLS]', '[SEP]']
DEVICE = torch.device("cpu")