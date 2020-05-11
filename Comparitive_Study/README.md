

Implementation of https://arxiv.org/pdf/1909.04495.pdf

We compare this against our work


# To train

python train-adversarial.py --data_path bpe_data_directory --outf output_directory --cuda --epochs 20 --no_earlystopping --perturb pgd --epsilon 0.05 --alpha 0.001 --steps 40
