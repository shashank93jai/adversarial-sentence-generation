#!/usr/bin/env python

import argparse
import os
import time
import math
import numpy as np
import random
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import to_gpu, Corpus, batchify, train_ngram_lm, get_ppl, filter_flip_polarity
from models import Seq2Seq, MLP_D, MLP_G, MLP_C

## NL: replaced all .data[0] and TENSOR[0] with .item()


parser = argparse.ArgumentParser(description='PyTorch ARAE for Text')
# Path Arguments
parser.add_argument('--data_path', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--kenlm_path', type=str, default='../Data/kenlm',
                    help='path to kenlm directory')
parser.add_argument('--outf', type=str, default='example',
                    help='output directory name')

# Data Processing Arguments
parser.add_argument('--vocab_size', type=int, default=12000,
                    help='cut vocabulary down to this size '
                         '(most frequently seen words in train)')
parser.add_argument('--maxlen', type=int, default=32,
                    help='maximum sentence length')
parser.add_argument('--lowercase', action='store_true',
                    help='lowercase all text')

# NL added:
parser.add_argument('--apply_bpe', action='store_true',
                    help='apply BPE to the text (e.g. if not already byte-pair-encoded)')

# Model Arguments
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhidden', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--noise_radius', type=float, default=0.2,
                    help='stdev of noise for autoencoder (regularizer)')
parser.add_argument('--noise_anneal', type=float, default=0.995,
                    help='anneal noise_radius exponentially by this'
                         'every 100 iterations')
parser.add_argument('--hidden_init', action='store_true',
                    help="initialize decoder hidden state with encoder's")
parser.add_argument('--arch_g', type=str, default='300-300',
                    help='generator architecture (MLP)')
parser.add_argument('--arch_d', type=str, default='300-300',
                    help='critic/discriminator architecture (MLP)')
parser.add_argument('--z_size', type=int, default=64,
                    help='dimension of random noise z to feed into generator')
parser.add_argument('--temp', type=float, default=1,
                    help='softmax temperature (lower --> more discrete)')
parser.add_argument('--enc_grad_norm', type=bool, default=True,
                    help='norm code gradient from critic->encoder')
parser.add_argument('--gan_toenc', type=float, default=-0.01,
                    help='weight factor passing gradient from gan to encoder')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')

# Training Arguments
parser.add_argument('--epochs', type=int, default=15,
                    help='maximum number of epochs')
parser.add_argument('--min_epochs', type=int, default=6,
                    help="minimum number of epochs to train for")
parser.add_argument('--no_earlystopping', action='store_true',
                    help="won't use KenLM for early stopping")
parser.add_argument('--patience', type=int, default=5,
                    help="number of language model evaluations without ppl "
                         "improvement to wait before early stopping")
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--niters_ae', type=int, default=1,
                    help='number of autoencoder iterations in training')
parser.add_argument('--niters_gan_d', type=int, default=5,
                    help='number of discriminator iterations in training')
parser.add_argument('--niters_gan_g', type=int, default=1,
                    help='number of generator iterations in training')
parser.add_argument('--niters_gan_schedule', type=str, default='', # default = '2-4-6',
                    help='epoch counts to increase number of GAN training '
                         ' iterations (increment by 1 each time)')
parser.add_argument('--lr_ae', type=float, default=1,
                    help='autoencoder learning rate')
parser.add_argument('--lr_gan_g', type=float, default=5e-05,
                    help='generator learning rate')
parser.add_argument('--lr_gan_d', type=float, default=1e-05,
                    help='critic/discriminator learning rate')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clipping, max norm')
parser.add_argument('--gan_clamp', type=float, default=0.01,
                    help='WGAN clamp')

# Evaluation Arguments
parser.add_argument('--sample', action='store_true',
                    help='sample when decoding for generation')
parser.add_argument('--N', type=int, default=5,
                    help='N-gram order for training n-gram language model')
parser.add_argument('--log_interval', type=int, default=200,
                    help='interval to log autoencoder training results')

# Other
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

parser.add_argument('--perturb', default=None)
parser.add_argument('--epsilon', type=float, default=0)
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--steps', type=int, default=0)

args = parser.parse_args()
print(vars(args))

# make output directory if it doesn't already exist
if not os.path.isdir('./output'):
    os.makedirs('./output')
if not os.path.isdir('./output/{}'.format(args.outf)):
    os.makedirs('./output/{}'.format(args.outf))

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

# create corpus
corpus = Corpus(args.data_path,
                maxlen=args.maxlen,
                vocab_size=args.vocab_size,
                lowercase=args.lowercase,
                max_lines=3001000,
                #test_size=1000
                test_size=-1, ## NL changed from test_size=1000 to test_size=-1, so that the Corpus will use the test.txt file instead of partitioning the train.txt file
                apply_bpe=args.apply_bpe  ## NL added: option to apply BPE live, e.g. if the input text isn't already byte-pair-encoded
               )
# dumping vocabulary
vocab = corpus.dictionary.word2idx
with open('./output/{}/vocab.json'.format(args.outf), 'w') as f:
    json.dump(vocab, f)

# save arguments
ntokens = len(corpus.dictionary.word2idx)
nclasses = 2  # NL: changed from 3 to 2 since we're predicting binary. Could change to 1, but would have to change other code to reflect this...
print("Vocabulary Size: {}".format(ntokens))
args.ntokens = ntokens
args.nclasses = nclasses
with open('./output/{}/args.json'.format(args.outf), 'w') as f:
    json.dump(vars(args), f)
with open("./output/{}/logs.txt".format(args.outf), 'w') as f:
    f.write(str(vars(args)))
    f.write("\n\n")

eval_batch_size = 100
pad_idx = vocab['<pad>']

train_data = batchify(corpus.train, args.batch_size, shuffle=True, pad_id=pad_idx)
test_data = batchify(corpus.test, eval_batch_size, shuffle=False, pad_id=pad_idx)

print("Loaded data!")

# ###############################################################################
# # Build the models
###############################################################################

autoencoder = Seq2Seq(emsize=args.emsize,
                      nhidden=args.nhidden,
                      ntokens=args.ntokens,
                      nlayers=args.nlayers,
                      noise_radius=args.noise_radius,
                      hidden_init=args.hidden_init,
                      dropout=args.dropout,
                      gpu=args.cuda)

gan_gen = MLP_G(ninput=args.z_size, noutput=args.nhidden, layers=args.arch_g)
gan_disc = MLP_D(ninput=args.nhidden, noutput=1, layers=args.arch_d)
enc_classifier = MLP_C(ninput=args.nhidden, noutput=args.nclasses, layers="100")
print(autoencoder)
print(enc_classifier)
print(gan_gen)
print(gan_disc)

optimizer_ae = optim.SGD(autoencoder.parameters(), lr=args.lr_ae)
# optimizer_enc_classifier = optim.SGD(enc_classifier.parameters(), lr=0.1)
optimizer_enc_classifier = optim.Adam(enc_classifier.parameters(),
                                      lr=args.lr_gan_d,
                                      betas=(args.beta1, 0.999))
optimizer_gan_g = optim.Adam(gan_gen.parameters(),
                             lr=args.lr_gan_g,
                             betas=(args.beta1, 0.999))
optimizer_gan_d = optim.Adam(gan_disc.parameters(),
                             lr=args.lr_gan_d,
                             betas=(args.beta1, 0.999))

criterion_ce = nn.CrossEntropyLoss()

if args.cuda:
    autoencoder = autoencoder.cuda()
    enc_classifier = enc_classifier.cuda()
    gan_gen = gan_gen.cuda()
    gan_disc = gan_disc.cuda()
    criterion_ce = criterion_ce.cuda()

###############################################################################
# Training code
###############################################################################


def save_model(epoch=''):
    print("Saving models")
    with open('./output/{}/autoencoder_model{}.pt'.format(args.outf, epoch), 'wb') as f:
        torch.save(autoencoder.state_dict(), f)
    with open('./output/{}/enc_classifier_model{}.pt'.format(args.outf, epoch), 'wb') as f:
        torch.save(enc_classifier.state_dict(), f)
    with open('./output/{}/gan_gen_model{}.pt'.format(args.outf, epoch), 'wb') as f:
        torch.save(gan_gen.state_dict(), f)
    with open('./output/{}/gan_disc_model{}.pt'.format(args.outf, epoch), 'wb') as f:
        torch.save(gan_disc.state_dict(), f)


def evaluate_autoencoder(data_source, epoch):
    # Turn on evaluation mode which disables dropout.
    autoencoder.eval()
    enc_classifier.eval()
    total_loss = 0
    ntokens = args.ntokens
    nclasses = args.nclasses
    all_accuracies = 0
    all_class_accuracies = 0
    bcnt = 0
    for i, batch in enumerate(data_source):
        source, target, lengths, tags = batch
        source = to_gpu(args.cuda, Variable(source, volatile=True))
        target = to_gpu(args.cuda, Variable(target, volatile=True))

        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        # examples x ntokens
        output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

        # output: batch x seq_len x ntokens
        output = autoencoder(source, lengths, noise=True)
        output_encode_only = autoencoder(source, lengths, noise=False, encode_only=True)
        output_classifier = enc_classifier(output_encode_only)
        _, output_classifier = torch.max(output_classifier, -1)

        flattened_output = output.view(-1, ntokens)

        masked_output = \
            flattened_output.masked_select(output_mask).view(-1, ntokens)
        total_loss += criterion_ce(masked_output/args.temp, masked_target).data

        # accuracy
        max_vals, max_indices = torch.max(masked_output, 1)
        all_accuracies += \
            torch.mean(max_indices.eq(masked_target).float()).item()
        bcnt += 1
        
        output_classifier = output_classifier.data.cpu().numpy()
        tags = tags.numpy()
        all_class_accuracies += \
            np.equal(output_classifier, tags).sum()

        aeoutf = "./output/%s/%d_autoencoder.txt" % (args.outf, epoch)
        with open(aeoutf, "a") as f:
            max_values, max_indices = torch.max(output, 2)
            max_indices = \
                max_indices.view(output.size(0), -1).data.cpu().numpy()
            target = target.view(output.size(0), -1).data.cpu().numpy()
            
            for t, idx, cls, cls_real in zip(target, max_indices, output_classifier, tags):
                # real sentence
                chars = " ".join([corpus.dictionary.idx2word[x] for x in t])
                f.write(str(cls_real))
                f.write("\t")
                f.write(chars)
                f.write("\n")
                # autoencoder output sentence
                chars = " ".join([corpus.dictionary.idx2word[x] for x in idx])
                f.write(str(cls))
                f.write("\t")
                f.write(chars)
                f.write("\n\n")

    return total_loss.item()/bcnt, all_accuracies/bcnt, all_class_accuracies/len(data_source)


def evaluate_generator(noise, epoch):
    gan_gen.eval()
    autoencoder.eval()

    # generate from fixed random noise
    fake_hidden = gan_gen(noise)
    max_indices = \
        autoencoder.generate(fake_hidden, args.maxlen, sample=args.sample)

    with open("./output/%s/%s_generated.txt" % (args.outf, epoch), "w") as f:
        max_indices = max_indices.data.cpu().numpy()
        for idx in max_indices:
            # generated sentence
            words = [corpus.dictionary.idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars)
            f.write("\n")


def train_lm(eval_path, save_path):
    # generate examples
    indices = []
    noise = to_gpu(args.cuda, Variable(torch.ones(100, args.z_size)))
    for i in range(1000):
        noise.data.normal_(0, 1)

        fake_hidden = gan_gen(noise)
        max_indices = autoencoder.generate(fake_hidden, args.maxlen)
        indices.append(max_indices.data.cpu().numpy())

    indices = np.concatenate(indices, axis=0)

    # write generated sentences to text file
    with open(save_path+".txt", "w") as f:
        # laplacian smoothing
        for word in corpus.dictionary.word2idx.keys():
            f.write(word+"\n")
        for idx in indices:
            # generated sentence
            words = [corpus.dictionary.idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars+"\n")

    # train language model on generated examples
    lm = train_ngram_lm(kenlm_path=args.kenlm_path,
                        data_path=save_path+".txt",
                        output_path=save_path+".arpa",
                        N=args.N)

    # load sentences to evaluate on
    with open(eval_path, 'r') as f:
        lines = f.readlines()
    sentences = [l.replace('\n', '') for l in lines]
    ppl = get_ppl(lm, sentences)

    return ppl

def fgsm_attack(sentence_embedding, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_embedding = sentence_embedding + epsilon*sign_data_grad
    #clip within normal range for embedding
    perturbed_embedding = torch.clamp(perturbed_embedding, -0.34, 0.32)
    return perturbed_embedding

def train_ae_and_classifier(batch, total_loss_ae, start_time, i, perturb=None, epsilon=0.0, alpha=0.0, pgd_iters=0):
    autoencoder.train()
    autoencoder.zero_grad()
    enc_classifier.train()
    enc_classifier.zero_grad()

    source, target, lengths, tags = batch
    source = to_gpu(args.cuda, Variable(source))
    target = to_gpu(args.cuda, Variable(target))
    tags = to_gpu(args.cuda, Variable(tags))

    # Create sentence length mask over padding
    mask = target.gt(0)
    masked_target = target.masked_select(mask)
    # examples x ntokens
    output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

    # output: batch x seq_len x ntokens
    output = autoencoder(source, lengths, noise=True)
    # output tags: batch_size x nclasses
    output_encode_only = autoencoder(source, lengths, noise=False, encode_only=True)
    output_classifier = enc_classifier(output_encode_only)

    perturbed_code = None
    if perturb == 'fgsm':
        output_encode_only.retain_grad()
        classifier_loss = criterion_ce(output_classifier, tags)
        enc_classifier.zero_grad()
        classifier_loss.backward(retain_graph=True)
        code_grad = output_encode_only.grad.data
        perturbed_code = fgsm_attack(output_encode_only, epsilon, code_grad)   
    elif perturb == 'pgd':
        perturbed_code = output_encode_only.clone().detach()
        for step_idx in range(pgd_iters):
            perturbed_code.requires_grad = True
            adv_scores = enc_classifier(perturbed_code)
            tmp_loss = criterion_ce(adv_scores, tags)
            enc_classifier.zero_grad()
            tmp_loss.backward(retain_graph=True)

            # step in the direction of the gradient
            perturbed_code = perturbed_code + alpha * perturbed_code.grad.sign()

            # Workaround as PyTorch doesn't have elementwise clip
            # from: https://gist.github.com/oscarknagg/45b187c236c6262b1c4bbe2d0920ded6#file-projected_gradient_descent-py
            perturbed_code = torch.max(torch.min(perturbed_code, output_encode_only + epsilon), output_encode_only - epsilon).detach()
            perturbed_code = torch.clamp(perturbed_code, -0.34, 0.32)
        print(i)
    print(i)
    # output_size: batch_size, maxlen, self.ntokens
    flattened_output = output.view(-1, ntokens)

    masked_output = \
        flattened_output.masked_select(output_mask).view(-1, ntokens)
    loss = criterion_ce(masked_output/args.temp, masked_target)
    classifier_loss = criterion_ce(output_classifier, tags)
    loss += classifier_loss
    
    if perturbed_code != None:
        output_classifier_adversarial = enc_classifier(perturbed_code)
        classifier_adversarial_loss = criterion_ce(output_classifier_adversarial, tags)
        loss += classifier_adversarial_loss
    
    loss.backward()

    # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)
    torch.nn.utils.clip_grad_norm(enc_classifier.parameters(), args.clip)
    optimizer_ae.step()
    optimizer_enc_classifier.step()

    total_loss_ae += loss.data

    accuracy = None
    if i % args.log_interval == 0 and i > 0:
        # accuracy
        probs = F.softmax(masked_output, dim=-1)
        max_vals, max_indices = torch.max(probs, 1)
        _, predicted_tags = torch.max(output_classifier, 1)

        accuracy = torch.mean(max_indices.eq(masked_target).float()).item()
        accuracy_classifier = torch.mean(predicted_tags.eq(tags).float()).item()

        cur_loss = total_loss_ae.item() / args.log_interval
        cur_loss_classifier = classifier_loss.item()
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
              'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f} | acc_cla {:8.2f} | loss_cla {:8.2f}'
              .format(epoch, i, len(train_data),
                      elapsed * 1000 / args.log_interval,
                      cur_loss, math.exp(cur_loss), accuracy, accuracy_classifier, cur_loss_classifier))

        with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
            f.write('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f} | acc_cla {:8.2f} | loss_cla {:8.2f}\n'.
                    format(epoch, i, len(train_data),
                           elapsed * 1000 / args.log_interval,
                           cur_loss, math.exp(cur_loss), accuracy, accuracy_classifier, cur_loss_classifier))

        total_loss_ae = 0
        start_time = time.time()

    return total_loss_ae, start_time


def train_gan_g():
    gan_gen.train()
    gan_gen.zero_grad()

    noise = to_gpu(args.cuda,
                   Variable(torch.ones(args.batch_size, args.z_size)))
    noise.data.normal_(0, 1)

    fake_hidden = gan_gen(noise)
    errG = gan_disc(fake_hidden).view(1)  # NL: added from our implementation

    # loss / backprop
    errG.backward(one)
    optimizer_gan_g.step()

    return errG


def grad_hook(grad):
    # Gradient norm: regularize to be same
    # code_grad_gan * code_grad_ae / norm(code_grad_gan)
    if args.enc_grad_norm:
        gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
        normed_grad = grad * autoencoder.grad_norm / gan_norm
    else:
        normed_grad = grad

    # weight factor and sign flip
    normed_grad *= -math.fabs(args.gan_toenc)
    return normed_grad


def train_gan_d(batch):
    # clamp parameters to a cube
    for p in gan_disc.parameters():
        p.data.clamp_(-args.gan_clamp, args.gan_clamp)

    autoencoder.train()
    autoencoder.zero_grad()
    gan_disc.train()
    gan_disc.zero_grad()

    # positive samples ----------------------------
    # generate real codes
    source, target, lengths, tags = batch
    source = to_gpu(args.cuda, Variable(source))
    target = to_gpu(args.cuda, Variable(target))

    # batch_size x nhidden
    real_hidden = autoencoder(source, lengths, noise=False, encode_only=True)
    real_hidden.register_hook(grad_hook)

    # loss / backprop
    errD_real = gan_disc(real_hidden).view(1)  # NL: added from our implementation
    errD_real.backward(one)

    # negative samples ----------------------------
    # generate fake codes
    noise = to_gpu(args.cuda,
                   Variable(torch.ones(args.batch_size, args.z_size)))
    noise.data.normal_(0, 1)

    # loss / backprop
    fake_hidden = gan_gen(noise)
    errD_fake = gan_disc(fake_hidden.detach()).view(1)  # NL: added from our implementation
    errD_fake.backward(mone)

    # `clip_grad_norm` to prvent exploding gradient problem in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)

    optimizer_gan_d.step()
    optimizer_ae.step()
    errD = -(errD_real - errD_fake)

    return errD, errD_real, errD_fake


print("Training...")
with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
    f.write('Training...\n')

# schedule of increasing GAN training loops
if args.niters_gan_schedule != "":
    gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
else:
    gan_schedule = []
niter_gan = 1

fixed_noise = to_gpu(args.cuda,
                     Variable(torch.ones(args.batch_size, args.z_size)))
fixed_noise.data.normal_(0, 1)
one = to_gpu(args.cuda, torch.FloatTensor([1]))
mone = one * -1

best_ppl = None
impatience = 0
all_ppl = []
for epoch in range(1, args.epochs+1):
    # update gan training schedule
    if epoch in gan_schedule:
        niter_gan += 1
        print("GAN training loop schedule increased to {}".format(niter_gan))
        with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
            f.write("GAN training loop schedule increased to {}\n".
                    format(niter_gan))

    total_loss_ae = 0
    epoch_start_time = time.time()
    start_time = time.time()
    niter = 0
    niter_global = 1

    # loop through all batches in training data
    while niter < len(train_data):

        # train autoencoder ----------------------------
        for i in range(args.niters_ae):
            if niter == len(train_data):
                break  # end of epoch
            total_loss_ae, start_time = \
                train_ae_and_classifier(train_data[niter], total_loss_ae, start_time, niter, args.perturb, args.epsilon, args.alpha, args.steps)
            niter += 1

        # train gan ----------------------------------
        for k in range(niter_gan):

            # train discriminator/critic
            for i in range(args.niters_gan_d):
                # feed a seen sample within this epoch; good for early training
                errD, errD_real, errD_fake = \
                    train_gan_d(train_data[random.randint(0, len(train_data)-1)])

            # train generator
            for i in range(args.niters_gan_g):
                errG = train_gan_g()

        niter_global += 1
        if niter_global % 500 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.8f (Real: %.8f '
                  'Fake: %.8f) Loss_G: %.8f'
                  % (epoch, args.epochs, niter, len(train_data),
                     errD.item(), errD_real.item(),
                     errD_fake.item(), errG.item()))
            with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
                f.write('[%d/%d][%d/%d] Loss_D: %.8f (Real: %.8f '
                        'Fake: %.8f) Loss_G: %.8f\n'
                        % (epoch, args.epochs, niter, len(train_data),
                           errD.item(), errD_real.item(),
                           errD_fake.item(), errG.item()))

            # exponentially decaying noise on autoencoder
            autoencoder.noise_radius = \
                autoencoder.noise_radius*args.noise_anneal

            if niter_global % 2000 == 0:
                evaluate_generator(fixed_noise, "epoch{}_step{}".
                                   format(epoch, niter_global))

                # evaluate with lm
                if not args.no_earlystopping and epoch > args.min_epochs:
                    ppl = train_lm(eval_path=os.path.join(args.data_path,
                                                          "test.txt"),
                                   save_path="output/{}/"
                                             "epoch{}_step{}_lm_generations".
                                             format(args.outf, epoch,
                                                    niter_global))
                    print("Perplexity {}".format(ppl))
                    all_ppl.append(ppl)
                    print(all_ppl)
                    with open("./output/{}/logs.txt".
                              format(args.outf), 'a') as f:
                        f.write("\n\nPerplexity {}\n".format(ppl))
                        f.write(str(all_ppl)+"\n\n")
                    if best_ppl is None or ppl < best_ppl:
                        impatience = 0
                        best_ppl = ppl
                        print("New best ppl {}\n".format(best_ppl))
                        with open("./output/{}/logs.txt".
                                  format(args.outf), 'a') as f:
                            f.write("New best ppl {}\n".format(best_ppl))
                        save_model()
                    else:
                        impatience += 1
                        # end training
                        if impatience > args.patience:
                            print("Ending training")
                            with open("./output/{}/logs.txt".
                                      format(args.outf), 'a') as f:
                                f.write("\nEnding Training\n")
                            sys.exit()

    # end of epoch ----------------------------
    # save model every epoch
    save_model(epoch="_{}".format(epoch))
    # evaluation
    test_loss, accuracy, class_accuracy = evaluate_autoencoder(test_data, epoch)
    eval_duration = (time.time() - epoch_start_time)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
          'test ppl {:5.2f} | acc {:3.3f} | acc_clas {:3.3f}'.
          format(epoch, eval_duration,
                 test_loss, math.exp(test_loss), accuracy, class_accuracy))
    print('-' * 89)

    with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
        f.write('-' * 89)
        f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
                ' test ppl {:5.2f} | acc {:3.3f} | acc_clas {:3.3f}\n'.
                format(epoch, eval_duration,
                       test_loss, math.exp(test_loss), accuracy, class_accuracy))
        f.write('-' * 89)
        f.write('\n')

    evaluate_generator(fixed_noise, "end_of_epoch_{}".format(epoch))
    if not args.no_earlystopping and epoch >= args.min_epochs:
        ppl = train_lm(eval_path=os.path.join(args.data_path, "test.txt"),
                       save_path="./output/{}/end_of_epoch{}_lm_generations".
                                 format(args.outf, epoch))
        print("Perplexity {}".format(ppl))
        all_ppl.append(ppl)
        print(all_ppl)
        with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
            f.write("\n\nPerplexity {}\n".format(ppl))
            f.write(str(all_ppl)+"\n\n")
        if best_ppl is None or ppl < best_ppl:
            impatience = 0
            best_ppl = ppl
            print("New best ppl {}\n".format(best_ppl))
            with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
                f.write("New best ppl {}\n".format(best_ppl))
            save_model()
        else:
            impatience += 1
            # end training
            if impatience > args.patience:
                print("Ending training")
                with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
                    f.write("\nEnding Training\n")
                sys.exit()

    # shuffle between epochs
    train_data = batchify(corpus.train, args.batch_size, shuffle=True, pad_id=pad_idx)
