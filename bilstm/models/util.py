import torch
import torch.nn.functional as F

# ******** LSTM模型 工具函数*************

def tensorized(batch, maps): # 构造batch的索引数组和大小数组
    PAD = maps.get('<pad>')
    UNK = maps.get('<unk>')

    max_len = len(batch[0])
    batch_size = len(batch)

    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch): # i = i, l = batch[i]
        for j, e in enumerate(l): # j = j, e = batch[i][j]
            batch_tensor[i][j] = maps.get(e, UNK) # 不存在则返回UNK
    # batch各个元素的长度
    lengths = [len(l) for l in batch]

    return batch_tensor, lengths


def sort_by_lengths(word_lists, tag_lists):
    pairs = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]
    # pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    word_lists, tag_lists = list(zip(*pairs))

    return word_lists, tag_lists, indices


def cal_loss(logits, targets, tag2id):
    """计算损失
    参数:
        logits: [B, L, out_size]
        targets: [B, L] # 标注数组
        lengths: [B]
    """
    PAD = tag2id.get('<pad>')

    mask = (targets != PAD)  # [B, L]
    targets = targets[mask]
    out_size = logits.size(2)
    logits = logits.masked_select(mask.unsqueeze(2).expand(-1, -1, out_size)).contiguous().view(-1, out_size)
    loss = F.cross_entropy(logits, targets)

    return loss
