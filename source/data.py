from os.path import join

import torch
from torch.utils.data import Dataset

from source.utils import load_sents

class GigaDataset(Dataset):
    def __init__(self, path, split):
        """
        args:
        path: path to dataset
        split: train/val/test
        """
        assert split in ['train', 'val', 'test']
        self.path = path
        self.src = load_sents(join(path, split + "_article.txt"))
        self.tgt = load_sents(join(path, split + "_title.txt"))
        assert len(self.src) == len(self.tgt)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.tgt[index]

def prepro_batch(max_src_len, max_tgt_len, word2id, batch):

    def sort_key(src_tgt):
        return (len(src_tgt[0].split()), len(src_tgt[1].split()))
    batch.sort(key=sort_key, reverse=True)

    sources, abstract = zip(*batch)
    # print('sources',sources)
    # print('abstact',abstract)
    sources = [s.split()[:max_src_len] for s in sources]
    #print('sources',sources)
    inp_lengths = torch.LongTensor([len(s) for s in sources])
    # print('inp_lengths',inp_lengths)
    abstract = [t.split()[:max_tgt_len] for t in abstract]
    #print('abstract',abstract)

    tgt = [['<start>'] + t for t in abstract]
    # print('tgt',tgt)
    target = [t + ['<end>'] for t in abstract]
    # print('target',target)
    #ext_word2id contains oov
    ext_word2id = dict(word2id)
    #print(word2id)
    #print('ext_word2id',ext_word2id)
    # for source in sources:
    #     for word in source:
    #         if word not in word2id:
    #             ext_word2id[word] = len(ext_word2id)-1
    # print('ext_word2id', ext_word2id)
    #tensorize
    sources = tensorized(sources, word2id)
    #print(sources)
    ext_sources = tensorized(sources, ext_word2id)
    ext_vsize = len(word2id)

    tgt = tensorized(tgt, word2id)
    target = tensorized(target, word2id)

    return (sources, inp_lengths, tgt, ext_sources, ext_vsize), target

def tensorized(sents_batch, word2id):
    """return [batch_size, max_lengths] tensor"""

    batch_size = len(sents_batch)
    # print('batch_size',batch_size)
    max_lengths = max(len(sent) for sent in sents_batch)
    PAD = word2id['<pad>']
    batch = torch.ones(batch_size, max_lengths, dtype=torch.long) * PAD

    for sent_i, sent in enumerate(sents_batch):
        for word_i, word in enumerate(sent):
            batch[sent_i, word_i] = word2id.get(word,word2id['<unk>'])

    #print('batch-------',batch)
    return batch


