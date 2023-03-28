import numpy as np
from tqdm import tqdm  # 第三方进度条库，看到运行进度条

from test import *


def extract(string, t):
    user_list = []
    word_list = []
    for i in string.split():
        split_words = str(i).split('/')  # check //m
        word, tag = split_words[0], split_words[-1]
        word_list.append(word)
        if tag == t:
            user_list.append(word)
    return user_list, word_list


def build_train(word_list, user_list):
    with open("BIO_train.txt", "a", encoding="utf-8", newline="") as f:
        for value in word_list:
            if (len(value) == 1):
                if value in user_list:
                    f.write(value + " " + "B")
                else:
                    f.write(value + " " + "O")
                f.write("\r\n")
            else:
                if value in user_list:
                    f.write(value[0] + " " + "B")
                else:
                    f.write(value[0] + " " + "O")
                f.write("\r\n")
                for i in range(len(value))[1:]:
                    if value in user_list:
                        f.write(value[i] + " " + "I")
                    else:
                        f.write(value[i] + " " + "O")
                    f.write("\r\n")


class HMM_Model:
    def __init__(self):
        # 标记-id
        self.tag2id = {'B': 0,
                       'I': 1,
                       'O': 2}
        # id-标记
        self.id2tag = dict(zip(self.tag2id.values(), self.tag2id.keys()))
        # 表示所有可能的标签个数N
        self.num_tag = len(self.tag2id)
        # 所有字符的Unicode编码个数 x16
        self.num_char = 65535
        # 转移概率矩阵,N*N，相邻tag间转移的概率
        self.A = np.zeros((self.num_tag, self.num_tag))
        # 输出概率矩阵,N*M，但此时某tag的概率
        self.B = np.zeros((self.num_tag, self.num_char))
        # 初始隐状态概率,N
        self.pi = np.zeros(self.num_tag)
        # 无穷小量
        self.epsilon = 1e-100

    def train(self, corpus_path):
        with open(corpus_path, mode='r', encoding='utf-8') as f:
            # 读取训练数据
            lines = f.readlines()
        BIO_true = []
        Obs = ""
        print('开始训练数据：')
        for i in tqdm(range(len(lines))):
            if len(lines[i]) == 1:
                # 空行，即只有一个换行符，跳过
                continue
            else:
                # split()的时候，多个空格当成一个空格。“字 tag”
                cut_char, cut_tag = lines[i].split()
                BIO_true.append(cut_tag)
                Obs += cut_char
                # ord是python内置函数
                # ord(c)返回字符c对应的十进制整数。B[id][字对应的十进制整数]
                self.B[self.tag2id[cut_tag]][ord(cut_char)] += 1
                if len(lines[i - 1]) == 1:
                    # 如果上一个数据是空格
                    # 即当前为一句话的开头
                    # 即初始状态
                    self.pi[self.tag2id[cut_tag]] += 1
                    continue
                pre_char, pre_tag = lines[i - 1].split()
                self.A[self.tag2id[pre_tag]][self.tag2id[cut_tag]] += 1
        # 为矩阵中所有是0的元素赋值为epsilon
        self.pi[self.pi == 0] = self.epsilon
        # 防止数据下溢,对数据进行对数归一化
        self.pi = np.log(self.pi) - np.log(np.sum(self.pi))
        self.A[self.A == 0] = self.epsilon
        # axis=1将每一行的元素相加，keepdims=True保持其二维性
        self.A = np.log(self.A) - np.log(np.sum(self.A, axis=1, keepdims=True))
        self.B[self.B == 0] = self.epsilon
        self.B = np.log(self.B) - np.log(np.sum(self.B, axis=1, keepdims=True))
        print('训练完毕！')
        return BIO_true, Obs

    def viterbi(self, Obs):
        # 获得观测序列的文本长度
        T = len(Obs)
        # T*N
        delta = np.zeros((T, self.num_tag))
        # T*N
        psi = np.zeros((T, self.num_tag))
        # ord是python内置函数
        # ord(c)返回字符c对应的十进制整数
        # 初始化
        delta[0] = self.pi[:] + self.B[:, ord(Obs[0])]
        # range（）左闭右开
        for i in range(1, T):
            # arr.reshape(4,-1) 将arr变成4行的格式，列数自动计算的(c=4, d=16/4=4)
            temp = delta[i - 1].reshape(self.num_tag, -1) + self.A
            # 按列取最大值
            delta[i] = np.max(temp, axis=0)
            # 得到delta值
            delta[i] = delta[i, :] + self.B[:, ord(Obs[i])]
            # 取出元素最大值对应的索引
            psi[i] = np.argmax(temp, axis=0)
        # 最优路径回溯
        path = np.zeros(T)
        path[T - 1] = np.argmax(delta[T - 1])
        for i in range(T - 2, -1, -1):
            path[i] = int(psi[i + 1][int(path[i + 1])])
        return path

    def predict(self, Obs):
        T = len(Obs)
        path = self.viterbi(Obs)
        BIO_pred = []
        for i in range(T):
            BIO_pred.append(self.id2tag[path[i]])
        return BIO_pred

def ner_hmm(tag):
    model = HMM_Model()


    with open("dataset/199801.txt", "r", encoding="utf-8", newline="") as f:
        train = f.read()
    user_list, word_list = extract(train, tag)
    build_train(word_list, user_list)

    BIO_true, Obs = model.train("BIO_train_per.txt")
    # 识别人名、地名、组织机构名
    BIO_pred = model.predict(Obs)
    return cal_f1(cal_precision(BIO_pred, BIO_true), cal_recall(BIO_pred, BIO_true))


