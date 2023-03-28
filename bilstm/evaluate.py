import time

from models.bilstm import BILSTM_Model
from utils import save_model

def bilstm_train(train_data, dev_data, word2id, tag2id):

    train_word_lists, train_tag_lists = train_data  # 训练集观测值序列和标注序列
    dev_word_lists, dev_tag_lists = dev_data        # 验证集观测值序列和标注序列

    start = time.time()
    vocab_size = len(word2id)   # 观测值类别数
    out_size = len(tag2id)      # 标注值类别数
    bilstm_model = BILSTM_Model(vocab_size, out_size)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)
    model_name = "bilstm"
    save_model(bilstm_model, "./ckpts/"+model_name+".pkl")
    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))

