import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import to_gpu
import json
import os
import numpy as np


class MLP_D(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.LeakyReLU(0.2)):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = torch.mean(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class MLP_G(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU()):
        super(MLP_G, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

class MLP_C(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU()):
        super(MLP_C, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
#             if i != 0:
#                 bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
#                 self.layers.append(bn)
#                 self.add_module("bn"+str(i+1), bn)
            dropout_layer = nn.Dropout(p=0.5)
            self.layers.append(dropout_layer)
            self.add_module("dropout"+str(i+1), dropout_layer)
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)
        self.layers.append(activation)
        self.add_module("activation"+str(i+1), activation)
        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)

class Seq2Seq(nn.Module):
    def __init__(self, emsize, nhidden, ntokens, nlayers, noise_radius=0.2,
                 hidden_init=False, dropout=0, gpu=False):
        super(Seq2Seq, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_radius = noise_radius
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder = nn.Embedding(ntokens, emsize)
        # Word dropout
        self.word_dropout_layer = nn.Dropout(p=self.dropout)
        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,
                               hidden_size=nhidden,
                               num_layers=nlayers,
                               dropout=dropout,
                               batch_first=True)

        decoder_input_size = emsize+nhidden
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=nlayers,
                               dropout=dropout,
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

    def init_weights(self):
        initrange = 0.01

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.weight.data[0] = 0.
        # self.embedding_decoder.weight.data.uniform_(-initrange, initrange)
        # self.embedding_decoder.weight.data[0] = 0.
        # Share embedding weights between encoder and decoder
        self.embedding_decoder.weight = self.embedding.weight
        

        # Initialize Encoder and Decoder Weights
        for name, param in self.encoder.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.decoder.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, indices, lengths, noise, encode_only=False):
        batch_size, maxlen = indices.size()

        hidden = self.encode(indices, lengths, noise)

        if hidden.requires_grad:
            hidden.register_hook(self.store_grad_norm)

        if encode_only:
            return hidden
        decoded = self.decode(hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded

    def encode(self, indices, lengths, noise):
        embeddings = self.embedding(indices)
        embeddings = self.word_dropout_layer(embeddings)
        packed_embeddings = pack_padded_sequence(embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        # Encode
        packed_output, state = self.encoder(packed_embeddings)

        hidden, cell = state
        # batch_size x nhidden
        hidden = hidden[-1]  # get hidden state of last layer of encoder

        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms = torch.norm(hidden, 2, 1)
        
        # For older versions of PyTorch use:
        # hidden = torch.div(hidden, norms.expand_as(hidden))
        # For newest version of PyTorch (as of 8/25) use this:
        hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))

        if noise and self.noise_radius > 0:
            gauss_noise = torch.zeros(hidden.size()).normal_(0, self.noise_radius)
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise, requires_grad=False))

        return hidden

    def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            repeat_hidden = hidden.unsqueeze(0).expand(self.nlayers, -1, -1).contiguous()
            state = (repeat_hidden, self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        embeddings = self.embedding_decoder(indices)
        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        packed_output, state = self.decoder(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        return decoded

    def generate(self, hidden, maxlen, sample=True, temp=1.0):
        """Generate through decoder; no backprop"""

        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            repeat_hidden = hidden.unsqueeze(0).expand(self.nlayers, -1, -1).contiguous()
            state = (repeat_hidden, self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # <sos>
        start_symbols = to_gpu(self.gpu, Variable(torch.ones(batch_size, 1).long(), requires_grad=False))

        embedding = self.embedding_decoder(start_symbols)
        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(maxlen):
            output, state = self.decoder(inputs, state)
            overvocab = self.linear(output.squeeze(1))

            if not sample:
                vals, indices = torch.max(overvocab, 1, keepdim=True)
            else:
                # sampling
                probs = F.softmax(overvocab/temp, -1)
                indices = torch.multinomial(probs, 1)

            all_indices.append(indices)

            embedding = self.embedding_decoder(indices)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)

        return max_indices


def load_models(load_path, suffix='', on_gpu=False, arch_cl=None):
    model_args = json.load(open("{}/args.json".format(load_path), "r"))
    
    # NL: added this function arg in case arch_cl was not one of the keys in model_args
    if arch_cl is not None:
        model_args["arch_cl"] = arch_cl
    
    word2idx = json.load(open("{}/vocab.json".format(load_path), "r"))
    idx2word = {v: k for k, v in word2idx.items()}

    autoencoder = Seq2Seq(emsize=model_args['emsize'],
                          nhidden=model_args['nhidden'],
                          ntokens=model_args['ntokens'],
                          nlayers=model_args['nlayers'],
                          hidden_init=model_args['hidden_init'],
                          gpu=on_gpu,
                          )
    gan_gen = MLP_G(ninput=model_args['z_size'],
                    noutput=model_args['nhidden'],
                    layers=model_args['arch_g'],
                    )
    gan_disc = MLP_D(ninput=model_args['nhidden'],
                     noutput=1,
                     layers=model_args['arch_d'],
                     )
    classifier = MLP_C(ninput=model_args['nhidden'],
                       noutput=model_args['nclasses'],
                       layers=model_args["arch_cl"],
                       )
    
    if on_gpu:
        autoencoder = autoencoder.cuda()
        classifier = classifier.cuda()
        gan_gen = gan_gen.cuda()
        gan_disc = gan_disc.cuda()
        

    print('Loading models from'+load_path)
    ae_path = os.path.join(load_path, "autoencoder_model{}.pt".format(suffix))
    gen_path = os.path.join(load_path, "gan_gen_model{}.pt".format(suffix))
    disc_path = os.path.join(load_path, "gan_disc_model{}.pt".format(suffix))
    cls_path = os.path.join(load_path, "enc_classifier_model{}.pt".format(suffix))

    autoencoder.load_state_dict(torch.load(ae_path))
    gan_gen.load_state_dict(torch.load(gen_path))
    gan_disc.load_state_dict(torch.load(disc_path))
    classifier.load_state_dict(torch.load(cls_path))
    return model_args, idx2word, autoencoder, gan_gen, gan_disc, classifier


def generate(autoencoder, gan_gen, z, vocab, sample, maxlen):
    """
    Assume noise is batch_size x z_size
    """
    if type(z) == Variable:
        noise = z
    elif type(z) == torch.FloatTensor or type(z) == torch.cuda.FloatTensor:
        noise = Variable(z)
    elif type(z) == np.ndarray:
        noise = Variable(torch.from_numpy(z).float(), volatile=True)
    else:
        raise ValueError("Unsupported input type (noise): {}".format(type(z)))

    gan_gen.eval()
    autoencoder.eval()

    # generate from random noise
    fake_hidden = gan_gen(noise)
    max_indices = autoencoder.generate(hidden=fake_hidden,
                                       maxlen=maxlen,
                                       sample=sample)

    max_indices = max_indices.data.cpu().numpy()
    sentences = []
    for idx in max_indices:
        # generated sentence
        words = [vocab[x] for x in idx]
        # truncate sentences to first occurrence of <eos>
        truncated_sent = []
        for w in words:
            if w != '<eos>':
                truncated_sent.append(w)
            else:
                break
        sent = " ".join(truncated_sent)
        sentences.append(sent)

    return sentences
