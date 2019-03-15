# -*- coding: utf-8 -*-
"""
QNA model
modified from Translation with a Sequence to Sequence Network and Attention
*************************************************************
**Author**: `Sean Robertson <https://github.com/spro/practical-pytorch>`_
"""

import argparse
import dill

from io import open
import random
import torch
import torch.nn as nn
from torch import optim
import time

from encoder_decoder import EncoderRNN, DecoderRNN, CaptionDecoderRNN
from plot_results import showPlot
from prepare_data import prepareData, prepareCaptions, shuffle_batched_pairs
from timing import asMinutes, timeSince

from lang import SOS_token, EOS_token, PAD_token, UNK_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 100
MAX_LENGTH = 20

teacher_forcing_ratio = 0.5


######################################################################
# Preparing Training Data
# -----------------------
#
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.
#

# TODO from here onwards requires modifications
def padSentInds(lang, sentence):
    no_padding = [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]
    padded = no_padding + ([PAD_token] * (MAX_LENGTH - len(no_padding)))
    return padded

def indexesFromBatch(cap_lang, pairs_batch):
    vatt_list = [pair[0] for pair in pairs_batch]
    cap_list = [padSentInds(cap_lang, pair[1]) for pair in pairs_batch]
    return vatt_list, cap_list


def tensorFromBatch(cap_lang, pairs_batch):
    vatt_list, cap_list = indexesFromBatch(cap_lang, pairs_batch)
    batch_size = len(vatt_list)
    vatt_tensor = torch.tensor(vatt_list, device=device, dtype=torch.float).view(1, batch_size, -1)
    cap_tensor = torch.tensor(cap_list, dtype=torch.long, device=device).view(-1, batch_size)
    return vatt_tensor, cap_tensor

def tensorsFromPairs(cap_lang, pairs_batch):
    return tensorFromBatch(cap_lang, pairs_batch)


######################################################################
# Training the Model
# ------------------
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf>`__.
#
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
#
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.
#


def train(vatt_tensor, cap_tensor, decoder, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    batch_size = cap_tensor.size()[1]
    decoder_optimizer.zero_grad()

    caption_length = cap_tensor.size(0)

    loss = 0

    decoder_input = torch.tensor([[SOS_token] * batch_size], device=device)
    decoder_output, decoder_hidden = decoder.special_forward(vatt_tensor, decoder_hidden)

    print('decoder input: {}'.format(decoder_input.shape))

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(caption_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = cap_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        #decoder_inputs = np.array()
        final_losses = [0] * caption_length
        for di in range(caption_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, cap_tensor[di])

    loss.backward()

    decoder_optimizer.step()

    return loss.item() / caption_length




######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(decoder, pairs, cap_lang, n_examples, print_every=1000, plot_every=100, learning_rate=0.01, save_every=1000):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    print('start map to tensor')
    training_pairs = [tensorsFromPairs(cap_lang, i) for i in pairs]
    print('end map to tensor')
    criterion = nn.NLLLoss(ignore_index=PAD_token)

    n_iters = int(n_examples / BATCH_SIZE)
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        vatt_tensor = training_pair[0] # v_att * batchsz
        cap_tensor = training_pair[1]

        print('iter {}:'.format(iter))
        print('vatt_shape: {}'.format(input_tensor.shape))

        loss = train(vatt_tensor, caption_file, 
                     decoder, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if iter % save_every == 0:
            with open('record.txt', 'w') as f:
                f.write(str(iter)+'\n')
            torch.save(decoder, 'decoder_iter_{:d}.pt'.format(iter%10000))


    showPlot(plot_losses)



######################################################################
# Training and Evaluating
# =======================
#
# With all these helper functions in place (it looks like extra work, but
# it makes it easier to run multiple experiments) we can actually
# initialize a network and start training.
#
# .. Note::
#    If you run this notebook you can train, interrupt the kernel,
#    evaluate, and continue training later. Comment out the lines where the
#    encoder and decoder are initialized and run ``trainIters`` again.
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vatt', help='vatt file')
    parser.add_argument('cap', help='caption file')
    args = parser.parse_args()

    vatts = args.vatt
    caps = args.cap

    cap_lang, batch_pairs, nExamples = prepareCaptions(vatts, caps)
    with open('caption_lang.pickle', 'wb') as f:
        dill.dump(cap_lang, f)

    vatt_size = 1020
    hidden_size = 256
    caption_decoder = CaptionDecoderRNN(vatt_size, hidden_size, cap_lang.n_words).to(device)
    epochs = 10

    for epoch in range(epochs):
        print('Epoch {:d}'.format(epoch))
        trainIters(caption_decoder, batch_pairs, cap_lang, nExamples, print_every=5000, learning_rate=0.001, save_every=1000)
        batch_pair = shuffle_batched_pairs(batch_pairs)
        torch.save(caption_decoder, 'caption_decoder_epoch_{:d}.pt'.format(epoch))
