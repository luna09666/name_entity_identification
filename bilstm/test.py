from utils import load_model, extend_maps
from data import build_corpus
from evaluating import Metrics

BiLSTM_MODEL_PATH = './ckpts/bilstm0.pkl'

REMOVE_O = False  # 在评估的时候是否去除O标记

def main():
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # bilstm模型
    print("加载并评估bilstm模型...")
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id)
    bilstm_model = load_model(BiLSTM_MODEL_PATH)
    # lstm_pred是预测结果，target_tag_list是实际结果
    lstm_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                    bilstm_word2id, bilstm_tag2id)
    metrics = Metrics(target_tag_list, lstm_pred, remove_O=REMOVE_O)
    metrics.report_scores()
    # metrics.report_confusion_matrix()
    print("测试完毕")

if __name__ == "__main__":
    main()
