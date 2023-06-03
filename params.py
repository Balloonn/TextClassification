import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE_PATH = os.path.join(PROJECT_DIR, 'data/train.csv')
TEST_FILE_PATH = os.path.join(PROJECT_DIR, 'data/test.csv')
LABELS_PK_PATH = os.path.join(PROJECT_DIR, 'data_pk/labels.pk')
CHARS_PK_PATH = os.path.join(PROJECT_DIR, 'data_pk/chars.pk')
PRETRAINED_VECTOR_PATH = os.path.join(PROJECT_DIR, 'PretrainedVector/sgns.wiki.char.bz2')

CHAR_NUM = 5500
PAD = '<PAD>'
UNK = '<UNK>'
PAD_NO = 0
UNK_NO = 1
START_NO = UNK_NO + 1
SEQ_LEN = 256

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 10
