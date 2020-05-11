#!/bin/bash
python bytepairencoding/apply_bpe.py --codes bytepairencoding/bpecode_yelp --input data/ARAE_yelp/train1.txt --output data/ARAE_yelp/train1_bpe.txt 
python bytepairencoding/apply_bpe.py --codes bytepairencoding/bpecode_yelp --input data/ARAE_yelp/train2.txt --output data/ARAE_yelp/train2_bpe.txt
python bytepairencoding/apply_bpe.py --codes bytepairencoding/bpecode_yelp --input data/ARAE_yelp/valid1.txt --output data/ARAE_yelp/valid1_bpe.txt
python bytepairencoding/apply_bpe.py --codes bytepairencoding/bpecode_yelp --input data/ARAE_yelp/valid2.txt --output data/ARAE_yelp/valid2_bpe.txt
python bytepairencoding/apply_bpe.py --codes bytepairencoding/bpecode_yelp --input data/ARAE_yelp/test0.txt --output data/ARAE_yelp/test0_bpe.txt
python bytepairencoding/apply_bpe.py --codes bytepairencoding/bpecode_yelp --input data/ARAE_yelp/test1.txt --output data/ARAE_yelp/test1_bpe.txt