# -*- coding: utf-8 -*-
"""
QNA model
modified from Translation with a Sequence to Sequence Network and Attention
*************************************************************
**Author**: `Sean Robertson <https://github.com/spro/practical-pytorch>`_

"""
import argparse

from io import open
import random
import torch
import torch.nn as nn
from torch import optim
import time

from encoder_decoder import EncoderRNN, DecoderRNN
from plot_results import showPlot
from prepare_data import prepareData, shuffle_batched_triples
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

def indexesFromBatch(input_lang, output_lang, triple_batch):
    vinput_list = [triple[0] for triple in triple_batch]
    input_list = [padSentInds(input_lang, triple[1]) for triple in triple_batch]
    target_list = [padSentInds(output_lang, triple[2]) for triple in triple_batch]
    return vinput_list, input_list, target_list,


def tensorFromBatch(input_lang, output_lang, triple_batch):
    vinput_list, input_list, target_list = indexesFromBatch(input_lang, output_lang, triple_batch)
    batch_size = len(vinput_list)
    vinput_tensor = torch.tensor(vinput_list, device=device, dtype=torch.float).view(1, batch_size, -1)
    input_tensor = torch.tensor(input_list, dtype=torch.long, device=device).view(-1, batch_size)
    target_tensor = torch.tensor(target_list, dtype=torch.long, device=device).view(-1, batch_size)
    return vinput_tensor, input_tensor, target_tensor

def tensorsFromTriples(triple_batch):
    return tensorFromBatch(input_lang, output_lang, triple_batch)


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


def train(vatt_tensor, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    batch_size = input_tensor.size()[1]
    encoder_hidden = encoder.initHidden(batch_size)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    encoder_output, encoder_hidden = encoder.special_forward(
            vatt_tensor, encoder_hidden) #vatt_size x batchsz input

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden) # input_tensor now input len * batchsz
    decoder_input = torch.tensor([[SOS_token] * batch_size], device=device)
    decoder_hidden = encoder_hidden

    print('decoder input: {}'.format(decoder_input.shape))

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        #decoder_inputs = np.array()
        final_losses = [0] * target_length
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length




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
def trainIters(encoder, decoder, triples, n_examples, print_every=1000, plot_every=100, learning_rate=0.01, save_every=1000):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    print('start map to tensor')
    training_triples = [tensorsFromTriples(i) for i in triples]
    print('end map to tensor')
    print('start shuffle')
    random.shuffle(training_triples)
    print('end shuffle')
    criterion = nn.NLLLoss(ignore_index=PAD_token)

    n_iters = int(n_examples / BATCH_SIZE)
    for iter in range(1, n_iters + 1):
        training_triple = training_triples[iter - 1]
        vinput_tensor = training_triple[0] # v_att * batchsz
        input_tensor = training_triple[1]
        target_tensor = training_triple[2]

        print('iter {}:'.format(iter))
        print('vatt_shape: {}'.format(input_tensor.shape))

        loss = train(vinput_tensor, input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
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
            torch.save(encoder, 'encoder_iter_{:d}.pt'.format(iter%10000))
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
    parser.add_argument('vcap', help='vcap file')
    parser.add_argument('vknow', help='vknow file')

    args = parser.parse_args()

    vatts = args.vatt
    vcaps = args.cap
    vknows = args.know


    input_lang, output_lang, batch_triples, nExamples = prepareData(vatts, vcaps, vknows,
                                                              '../qna_training_coco/v2_OpenEnded_mscoco_train2014_questions.json',
                                                              '../qna_training_coco/v2_mscoco_train2014_annotations.json')
    vatt_size = 1020
    vcap_size = 256
    vknow_size = 300
    hidden_size = 256
    encoder1 = EncoderRNN(vatt_size, vcap_size, vknow_size, input_lang.n_words).to(device)
    decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)
    epochs = 10

    for epoch in range(epochs):
        print('Epoch {:d}'.format(epoch))
        trainIters(encoder1, decoder1, batch_triples, nExamples, print_every=5000, learning_rate=0.001, save_every=1000)
        torch.save(encoder1, 'encoder_epoch_{:d}.pt'.format(epoch))
        torch.save(decoder1, 'decoder_epoch_{:d}.pt'.format(epoch))
        batch_triples = shuffle_batched_triples(batch_triples)

        if (epoch % 10 == 0):
            torch.save(encoder1, 'encoder_epoch_{:d}.pt'.format(epoch))
            torch.save(decoder1, 'decoder_epoch_{:d}.pt'.format(epoch))
    torch.save(encoder1, 'encoder_epoch_{:d}.pt'.format(epoch))
    torch.save(decoder1, 'decoder_epoch_{:d}.pt'.format(epoch))
