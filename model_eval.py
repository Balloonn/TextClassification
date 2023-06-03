import torch as T
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from utils import *
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

model = T.load('sgns_cls.pth')

label_dict, char_dict = load_pk_file()
label_dict_rev = {v: k for k, v in label_dict.items()}


def eval(text):
    labels, contents = ['汽车'], [text]
    samples, y_true = text_feature(contents, labels, char_dict, label_dict)
    x = T.from_numpy(np.array(samples)).long()
    y_pred = model(x, None)
    y_numpy = y_pred.detach().numpy()
    predict_list = np.argmax(y_numpy, axis=1).tolist()
    return label_dict_rev[predict_list[0]]


if __name__ == '__main__':
    test_df = pd.read_csv('data/test.csv')
    true_label = []
    pred_label = []
    for index, row in test_df.iterrows():
        print(index)
        true_label.append(row['label'])
        pred_label.append(eval(row['content']))

    print(classification_report(true_label, pred_label, digits=4))
    # 绘制混淆矩阵
    label_names = list(label_dict.keys())
    C = confusion_matrix(true_label, pred_label, labels=label_names)

    plt.matshow(C, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    num_local = np.array(range(len(label_names)))
    plt.xticks(num_local, label_names, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, label_names)  # 将标签印在y轴坐标上
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    # plt.show()
    plt.savefig("./image/confusion_matrix.png")
