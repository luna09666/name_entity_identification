from os.path import join
from codecs import open


# 读取数据，获得word_list和tag_list
def build_corpus(filename, make_vocab=True, data_dir="./ResumeNER"):
    """读取数据"""
    word_lists = []
    tag_lists = []
    with open(join(data_dir, filename+".char.bmes"), 'r', encoding='utf-8') as file:
        word_list = []
        tag_list = []
        for line in file: # 将每个空行识别为一个段落的结尾，每个段落用一个数组储存
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists) # 给每个观测值附加word_lists的索引
        tag2id = build_map(tag_lists) # 给每个标注值附加tag_lists的索引
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps
