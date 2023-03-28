import re
from typing import List
from sklearn.metrics import f1_score
import jieba

import ner
import ner_hmm
from ner_hanlp import *


def read_boson(path: str) -> List:
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    return txt


boson_file = "dataset/BosonNLP_NER_6C.txt"
data = read_boson(boson_file)


def convert_data_to_ground_truth(data: List) -> None:
    org_name = re.findall(r"{org_name:(.+?)}", data)
    product_name = re.findall(r"{product_name:(.+?)}", data)
    time = re.findall(r"{time:(.+?)}", data)
    location = re.findall(r"{location:(.+?)}", data)
    company_name = re.findall(r"{company_name:(.+?)}", data)
    person_name = re.findall(r"{person_name:(.+?)}", data)
    return {'org_name': org_name, 'product_name': product_name, 'time': time, 'location': location,
            'company_name': company_name, 'person_name': person_name}


dataset = convert_data_to_ground_truth(data)
'''
with open("standard.txt", "a", encoding="utf-8", newline="") as f:
    for key, value in dataset.items():
        for i in value:
            f.write(i + '\n')
'''


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def movestopwords(sentence):
    stopwords = stopwordslist('dataset/stopwords.txt')
    santi_words = [x for x in sentence if len(x) > 1 and x not in stopwords]
    return santi_words


jieba.load_userdict("standard.txt")
seq_list = jieba.lcut(data)
seq_list_no_stop = movestopwords(seq_list)  # 分词提纯结果



# print("f1_ner_time:", ner.ner(seq_list_no_stop, ner.point_time, dataset, 'time'))
# print("f1_ner_org:", ner.ner(seq_list_no_stop, ner.point_org, dataset, 'org_name'))

# print("f1_hmm_org:", ner_hmm.ner_hmm('nt'))
# print("f1_hmm_per:", ner_hmm.ner_hmm('nr'))

# print("f1_hanlp_hmm_per:", ner_hanlp(seq_list_no_stop, dataset, 'person_name', tagger1, 'nr'))
# print("f1_hanlp_hmm_org:", ner_hanlp(seq_list_no_stop, dataset, 'org_name', tagger1, 'nt'))
# print("f1_hanlp_perc_per:", ner_hanlp(seq_list_no_stop, dataset, 'person_name', tagger2, 'nr'))