import torch
from gensim.models import KeyedVectors

from config.params import NUM_WORDS
from utils.text_featuring import load_file

label_dict, char_dict = load_file()

model = KeyedVectors.load_word2vec_format('./Pretrain_Vector/sgns.wiki.char.bz2',
                                          binary=False,
                                          encoding="utf-8",
                                          unicode_errors="ignore")
# gensim load word2vec
pretrained_vector = torch.zeros(NUM_WORDS + 4, 300).float()
# print(model.index2word)

for char, index in char_dict.items():
    if char in model.index_to_key:
        vector = model.get_vector(char)
        # print(vector)
        pretrained_vector[index, :] = torch.from_numpy(vector)
