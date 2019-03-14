import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 100

VATT_EMBED_SIZE = 256
WORD_EMBED_SIZE = 256
HIDDEN_SIZE = 256


######################################################################
# The Seq2Seq Model
# =================
#
# A Recurrent Neural Network, or RNN, is a network that operates on a
# sequence and uses its own output as input for subsequent steps.
#
# A `Sequence to Sequence network <https://arxiv.org/abs/1409.3215>`__, or
# seq2seq network, or `Encoder Decoder
# network <https://arxiv.org/pdf/1406.1078v3.pdf>`__, is a model
# consisting of two RNNs called the encoder and decoder. The encoder reads
# an input sequence and outputs a single vector, and the decoder reads
# that vector to produce an output sequence.
#
# .. figure:: /_static/img/seq-seq-images/seq2seq.png
#    :alt:
#
# Unlike sequence prediction with a single RNN, where every input
# corresponds to an output, the seq2seq model frees us from sequence
# length and order, which makes it ideal for translation between two
# languages.
#
# Consider the sentence "Je ne suis pas le chat noir" → "I am not the
# black cat". Most of the words in the input sentence have a direct
# translation in the output sentence, but are in slightly different
# orders, e.g. "chat noir" and "black cat". Because of the "ne/pas"
# construction there is also one more word in the input sentence. It would
# be difficult to produce a correct translation directly from the sequence
# of input words.
#
# With a seq2seq model the encoder creates a single vector which, in the
# ideal case, encodes the "meaning" of the input sequence into a single
# vector — a single point in some N dimensional space of sentences.
#


######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#

class EncoderRNN(nn.Module):
    def __init__(self, vatt_size, input_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = HIDDEN_SIZE

        self.vatt_embedding = nn.Embedding(vatt_size, VATT_EMBED_SIZE)
        self.vatt_embedding = nn.Linear(vatt_size, VATT_EMBED_SIZE, bias = False)
        self.word_embedding = nn.Embedding(input_size, WORD_EMBED_SIZE)
        # self.vcap_embedding = nn.Embedding(input_size, VCAP_EMBED_SIZE)
        # self.vknow_embedding = nn.Embedding(input_size, VKNOW_EMBED_SIZE)
        # self.lstm = nn.LSTM(VATT_EMBED_SIZE + VCAP_EMBED_SIZE + VKNOW_EMBED_SIZE, hidden_size)
        self.lstm = nn.LSTM(VATT_EMBED_SIZE, HIDDEN_SIZE)

    def special_forward(self, vatt, hidden):
        vatt_embedded = self.vatt_embedding(vatt)#.view(1,BATCH_SIZE,-1)
        output, hidden = self.lstm(vatt_embedded)
        return output, hidden

    def forward(self, input, hidden):
        embedded = self.word_embedding(input).view(-1, input.shape[1], WORD_EMBED_SIZE)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)



######################################################################
# The Decoder
# -----------
#
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
#


######################################################################
# Simple Decoder
# ^^^^^^^^^^^^^^
#
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
#
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
#
# .. figure:: /_static/img/seq-seq-images/decoder-network.png
#    :alt:
#
#

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # print('decoder input {}'.format(input.shape))
        output = self.embedding(input).view(1, input.shape[1], -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)