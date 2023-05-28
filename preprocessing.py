import pandas as pd
from collections import defaultdict, Counter
from operator import itemgetter
from pickle_file_operator import PickleFileOperator
from params import TRAIN_FILE_PATH, CHAR_NUM, CHARS_PK_PATH, LABELS_PK_PATH
from random import shuffle

class FilePreprocessing(object):
    def __init__(self, n):
        self._n = n  # keep top n words in vocab

    def read_train_file(self):
        train_pd = pd.read_csv(TRAIN_FILE_PATH)
        label_list = train_pd['label'].unique().tolist()

        # count word frequency
        character_dict = defaultdict(int)
        for content in train_pd['content']:
            for key, value in Counter(content).items():
                character_dict[key] += value
        # # 不排序
        # sort_char_list = [(k, v) for k, v in character_dict.items()]
        # shuffle(sort_char_list)
        # 排序
        sort_char_list = sorted(character_dict.items(), key=itemgetter(1), reverse=True)
        print('total {} words in character_dict'.format(len(sort_char_list)))
        print('top 5 characters: ', sort_char_list[:5])

        top_n_chars = [_[0] for _ in sort_char_list[:self._n]]
        return label_list, top_n_chars

    def run(self):
        label_list, top_n_chars = self.read_train_file()
        PickleFileOperator(data=label_list, file_path=LABELS_PK_PATH).save()
        PickleFileOperator(data=top_n_chars, file_path=CHARS_PK_PATH).save()


if __name__ == '__main__':
    processor = FilePreprocessing(CHAR_NUM)
    processor.run()

    labels = PickleFileOperator(file_path=LABELS_PK_PATH).read()
    chars = PickleFileOperator(file_path=CHARS_PK_PATH).read()
    print(labels)
    print(chars)
