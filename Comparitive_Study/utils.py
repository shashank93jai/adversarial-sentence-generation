import os
import torch
import numpy as np
import random
from itertools import dropwhile
import collections

def rindex(lst, item):
    def index_ne(x):
        return lst[x] != item
    try:
        return next(dropwhile(index_ne, reversed(range(len(lst)))))
    except StopIteration:
        raise ValueError("rindex(lst, item): item not in list")

def load_kenlm():
    global kenlm
    import kenlm

def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var

def to_categorical(y, num_classes=3):
    """ 1-hot encodes a list """
    ret_val = np.zeros((len(y), num_classes), dtype='uint8')
    for y_id in range(len(y)):
        cat_idx = y[y_id] - 1
        if cat_idx > 3: ret_val[y_id][2] = 1
        if cat_idx < 3: ret_val[y_id][0] = 1
        if cat_idx == 3: ret_val[y_id][1] = 1
    return ret_val

def to_class_id(y):
    """ turn yelp stars to 3 classes """
    ret_val = []
    for y_id in range(len(y)):
        if y[y_id] > 3: ret_val.append(2)
        if y[y_id] < 3: ret_val.append(0)
        if y[y_id] == 3: ret_val.append(1)
    return ret_val

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word2idx['<pad>'] = 0
        self.word2idx['<sos>'] = 1
        self.word2idx['<eos>'] = 2
        self.word2idx['<oov>'] = 3
        self.wordcounts = {}

    # to track word counts
    def add_word(self, word):
        if word not in self.wordcounts:
            self.wordcounts[word] = 1
        else:
            self.wordcounts[word] += 1

    # prune vocab based on count k cutoff or most frequently seen k words
    def prune_vocab(self, k=5, cnt=False):
        # get all words and their respective counts
        vocab_list = [(word, count) for word, count in self.wordcounts.items()]
        if cnt:
            # prune by count
            self.pruned_vocab = \
                    {pair[0]: pair[1] for pair in vocab_list if pair[1] > k}
        else:
            # prune by most frequently seen words
            vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = [pair[0] for pair in vocab_list[:k]]
        # sort to make vocabulary determistic
        self.pruned_vocab.sort()

        # add all chosen words to new vocabulary/dict
        for word in self.pruned_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        print("original vocab {}; pruned to {}".
              format(len(self.wordcounts), len(self.word2idx)))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self, path, maxlen, vocab_size=20000, lowercase=False,
                 max_lines=-1, test_size=-1, train_path='train.txt',
                 test_path='test.txt', load_vocab_file='',
                 apply_bpe=False  # apply BPE now
                 ):
        self.dictionary = Dictionary()
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.train_path = os.path.join(path, train_path)
        self.test_path = os.path.join(path, test_path)
        self.max_lines = max_lines
        self.apply_bpe = apply_bpe

        # NL new: option to apply BPE now
        if self.apply_bpe:
            from bpemb import BPEmb
            BPE_VOCAB_SIZE = 25000
            BPE_DIM = 300
            self.bpemb_en = BPEmb(lang="en", vs=BPE_VOCAB_SIZE, dim=BPE_DIM)
            print("\n\n--Applying BPE live--\n\n")
            
        # load existing vocabulary or make it from training set
        if len(load_vocab_file) > 0:
            self.load_vocab(load_vocab_file)
        else:
            self.make_vocab()
        assert len(self.dictionary) > 1
        self.train = self.tokenize(self.train_path)
        if test_size > 0 and len(test_path) > 0:
            print("Test size and test path cannot both be present!")
            exit()
        if test_size > 0:
            print("Using {} in training set as test set".format(test_size))
            self.train, self.test = self.train[:-test_size], self.train[-test_size:]
            return
        elif len(test_path) > 0:
            print("Using {} as test set".format(test_path))
            self.test = self.tokenize(self.test_path)

    def make_vocab(self):
        assert os.path.exists(self.train_path)
        # Add words to the dictionary
        with open(self.train_path, 'r') as f:
            linecount = 0
            for line in f:
                linecount += 1
                if self.max_lines > 1 and linecount >= self.max_lines:
                    break
                if self.lowercase:
                    words = line.strip().lower().split()
                else:
                    words = line.strip().split()
                words = words[1:] # exclude tag
                for word in words:
                    if self.apply_bpe:
                        for bp in self.bpemb_en.encode(word):
                            self.dictionary.add_word(bp)
                    else:
                        self.dictionary.add_word(word)

        # prune the vocabulary
        self.dictionary.prune_vocab(k=self.vocab_size, cnt=False)

    def load_vocab(self, vocab_file='vocab.json'):
        assert os.path.exists(vocab_file)
        import json
        self.dictionary.word2idx = json.load(open("{}".format(vocab_file), "r"))
        self.dictionary.idx2word = {v: k for k, v in self.dictionary.word2idx.items()}
        print("Loaded vocab file {} with {} words".
              format(vocab_file, len(self.dictionary.word2idx)))

    def tokenize(self, path):
        """Tokenizes a text file."""
        # Convert class 1,2 to 0,1
        # print("Convert class 1,2 to 0,1")
        # Convert class 1,2 to 0,1
        dropped = cropped = 0
        oov_count = 0.
        word_count = 0.
        with open(path, 'r') as f:
            linecount = 0
            lines = []
            tags = []
            for line in f:
                linecount += 1
                if self.max_lines > 1 and linecount >= self.max_lines:
                    break
                if self.lowercase:
                    words = line.lower().strip().split()
                else:
                    words = line.strip().split()
                tag, words = int(words[0]), words[1:]

                # if applying BPE
                if self.apply_bpe:
                    words = [bp for word in words
                             for bp in self.bpemb_en.encode(word)]

                if len(words) > self.maxlen:
                    cropped += 1
                    words = words[:self.maxlen]
#                     try:
#                         crop_words = words[:maxlen]
#                         last_period = max(rindex(crop_words, '.'), rindex(crop_words, '!'), rindex(crop_words, ','))
#                     except:
#                         last_period = self.maxlen
#                     if last_period < 10:
#                         print("Sentence too short! {}".format(words))
#                     words = words[:last_period]
                if len(words) < 3:
                    dropped += 1
#                     print(words)
                    continue
                words = ['<sos>'] + words
                words += ['<eos>']

                # vectorize
                vocab = self.dictionary.word2idx
                unk_idx = vocab['<oov>']
                indices = [vocab[w] if w in vocab else unk_idx for w in words]
                word_count += len(indices)
                oov_count += sum([1 if ii==unk_idx else 0 for ii in indices])
                # add to output list
                lines.append(indices)
                # Convert class 1,2 to 0,1
                # tag = tag - 1
                tags.append(tag)
        # tags = to_class_id(tags)
        print("Number of sentences cropped from {}: {} out of {} total, dropped {}. OOV rate {:.3f}".
              format(path, cropped, linecount, dropped, oov_count/word_count))

        return list(zip(tags, lines))


def batchify(data, bsz, shuffle=False, gpu=False, pad_id=0):
    if shuffle:
        random.shuffle(data)

    tags, sents = zip(*data)
    nbatch = len(data) // bsz

    batches = []

    for i in range(nbatch):

        batch = sents[i*bsz:(i+1)*bsz]
        batch_tags = tags[i*bsz:(i+1)*bsz]
        # downsample biggest class
        # batch, batch_tags = balance_tags(batch, batch_tags)
        # subtract 1 from lengths b/c includes BOTH starts & end symbols
        lengths = [len(x)-1 for x in batch]
        # sort items by length (decreasing)
        batch, batch_tags, lengths = length_sort(batch, batch_tags, lengths)

        # source has no end symbol
        source = [x[:-1] for x in batch]
        # target has no start symbol
        target = [x[1:] for x in batch]

        # Pad batches to maximum sequence length in batch
        # find length to pad to
        maxlen = lengths[0]
        for x, y in zip(source, target):
            pads = (maxlen-len(x))*[pad_id]
            x += pads
            y += pads

        source = torch.LongTensor(np.array(source))
        target = torch.LongTensor(np.array(target)).view(-1)
        batch_tags = torch.LongTensor(np.array(batch_tags))

        if gpu:
            source = source.cuda()
            target = target.cuda()
            batch_tags = batch_tags.cuda()

        batches.append((source, target, lengths, batch_tags))

    return batches

def filter_flip_polarity(data):
    flipped = []
    tags, sents = zip(*data)

    for i in range(len(tags)):
        org_tag = tags[i]
        sent = sents[i]
        if org_tag == 1: new_tag = 0
        if org_tag == 0: new_tag = 1
        flipped.append((new_tag, sent))
    print("Filtered and flipped {} sents from {} sents.".format(len(flipped), len(data)))
    return flipped

def length_sort(items, tags, lengths, descending=True):
    """In order to use pytorch variable length sequence package"""
    old_items = list(zip(items, tags, lengths))
    old_items.sort(key=lambda x: x[2], reverse=True)
    items, tags, lengths = zip(*old_items)
    return list(items), list(tags), list(lengths)

def balance_tags(items, tags):
    """Downsample largest group of tags"""
    new_items = []
    new_tags = []

    biggest_class = 2
    drop_ratio = .666
    for i in range(len(items)):
        tag = tags[i]
        item = items[i]
        if tag == biggest_class:
            if random.random() < drop_ratio:
                continue
        new_items.append(item)
        new_tags.append(tag)
    return new_items, new_tags

def train_ngram_lm(kenlm_path, data_path, output_path, N):
    """
    Trains a modified Kneser-Ney n-gram KenLM from a text file.
    Creates a .arpa file to store n-grams.
    """
    # create .arpa file of n-grams
    curdir = os.path.abspath(os.path.curdir)
    #
    command = "bin/lmplz -o "+str(N)+" <"+os.path.join(curdir, data_path) + \
              " >"+os.path.join(curdir, output_path)
    os.system("cd "+os.path.join(kenlm_path, 'build')+" && "+command)

    load_kenlm()
    # create language model
    model = kenlm.Model(output_path)

    return model


def get_ppl(lm, sentences):
    """
    Assume sentences is a list of strings (space delimited sentences)
    """
    total_nll = 0
    total_wc = 0
    for sent in sentences:
        words = sent.strip().split()
        score = lm.score(sent, bos=True, eos=False)
        word_count = len(words)
        total_wc += word_count
        total_nll += score
    ppl = 10**-(total_nll/total_wc)
    return ppl
