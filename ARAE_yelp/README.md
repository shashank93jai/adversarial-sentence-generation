# ARAE for language style transfer on Yelp

From https://github.com/jakezhaojb/ARAE

    # To train:

    python train.py --data_path ./data --vocab_size 8000 --maxlen 30 --emsize 300 --nhidden 512 --epochs 20 --arch_classify 128-128-128
    
    # To adversarially train
    
    python adversarial_training.py --data_path ./data --vocab_size 8000 --maxlen 30 --emsize 300 --nhidden 512 --epochs 20 --modelf temp_op --arch_classify 128-128-128 --perturb pgd --epsilon 0.005 --alpha 0.0005 --steps 40
    
## Requirements
- Python 3.6.3 on Linux
- PyTorch 0.3.1, JSON, Argparse
- KenLM (https://github.com/kpu/kenlm)
- Spacy, with `en_core_web_sm` installed
