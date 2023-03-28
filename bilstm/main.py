from data import build_corpus
from utils import extend_maps
from evaluate import bilstm_train

def main():
    """训练模型，评估结果"""

    # 读取数据
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    print("数据读取完毕")
    # 训练评估BI-LSTM模型
    print("正在训练双向LSTM模型...")
    # LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id) # 获取标注序列和观测值序列的索引
    bilstm_train((train_word_lists, train_tag_lists),    # 训练模型
                 (dev_word_lists, dev_tag_lists),
                 bilstm_word2id, bilstm_tag2id )
    print("训练完毕")

if __name__ == "__main__":
    main()
