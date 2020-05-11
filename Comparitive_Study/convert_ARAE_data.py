"""
Convert text from individual files
into a file with labels for each data partition
"""

import os

# for BPE data

DATA_DIR = 'data/hsieh_bpe'
ARAE_DIR = 'data/ARAE_yelp'

arae_files = {
    'train': {
        0: 'train1_bpe.txt',  # label: fname
        1: 'train2_bpe.txt'
    },
    'valid': {
        0: 'valid1_bpe.txt',
        1: 'valid2_bpe.txt'
    },
    'test': {
        0: 'test1_bpe.txt',
        1: 'test0_bpe.txt'
    }
}

for split, files in arae_files.items():
    with open(os.path.join(DATA_DIR, split + '.txt'), 'w') as f_write:
        for label, fname in files.items():
            with open(os.path.join(ARAE_DIR, fname), 'r') as f_read:
                for line in f_read:
                    f_write.write("{} {}".format(label, line))

                    
# for non-BPE data (e.g. if we want to apply BPE
# when loading the data, like in our original setup)

DATA_DIR = 'data/not_bpe'
ARAE_DIR = 'data/ARAE_yelp'

arae_files = {
    'train': {
        0: 'train1.txt',  # {label: fname}
        1: 'train2.txt'
    },
    'valid': {
        0: 'valid1.txt',
        1: 'valid2.txt'
    },
    'test': {
        0: 'test1.txt',
        1: 'test0.txt'
    }
}

for split, files in arae_files.items():
    with open(os.path.join(DATA_DIR, split + '.txt'), 'w') as f_write:
        for label, fname in files.items():
            with open(os.path.join(ARAE_DIR, fname), 'r') as f_read:
                for line in f_read:
                    f_write.write("{} {}".format(label, line))
