import torch
from utils import load_pk_file
from gensim.models import KeyedVectors
from params import PRETRAINED_VECTOR_PATH, CHAR_NUM


label_dict, char_dict = load_pk_file()

model = KeyedVectors.load_word2vec_format(PRETRAINED_VECTOR_PATH,
                                          binary=False,
                                          encoding='utf-8',
                                          unicode_errors="ignore")
pretrained_vector = torch.zeros(CHAR_NUM + 4, 300).float()

for char, index in char_dict.items():
    if char in model.key_to_index:
        vector = model.get_vector(char)
        pretrained_vector[index, :] = torch.from_numpy(vector)
