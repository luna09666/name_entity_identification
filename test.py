import os

from sklearn.metrics import f1_score


def test(seq_list_no_stop, dataset, result, tag):
    true = []
    pred = []
    for i in seq_list_no_stop:
        if i in dataset[tag]:
            true.append(1)
        else:
            true.append(0)
        if i in result:
            pred.append(1)
        else:
            pred.append(0)
    return f1_score(true, pred)


def cal_precision(pred, true):
    # 准确率 = 正确预测的样本数量/预测中所有样本的数量
    l = len(true)
    sum_correct = 0
    sum_all = 0
    for i in range(l):
        if pred[i] == true[i] and ("B" in pred[i] or "I" in pred[i]):
            sum_correct += 1
        if "B" in pred[i] or "I" in pred[i]:
            sum_all += 1
    return sum_correct / sum_all


def cal_recall(pred, true):
    # 召回率 = 正确预测的样本数量/标准结果中所有样本的数量
    l = len(true)
    sum_correct = 0
    sum_all = 0
    for i in range(l):
        if pred[i] == true[i] and ("B" in pred[i] or "I" in pred[i]):
            sum_correct += 1
        if "B" in true[i] or "I" in true[i]:
            sum_all += 1
    return sum_correct / sum_all

def cal_f1(p, r):
    return p * 2 * r / (p + r)