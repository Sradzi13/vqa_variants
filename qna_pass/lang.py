from collections import defaultdict
import json


######################################################################
# We'll need a unique index per word to use as the inputs and targets of
# the networks later. To keep track of all this we will use a helper class
# called ``Lang`` which has word → index (``word2index``) and index → word
# (``index2word``) dictionaries, as well as a count of each word
# ``word2count`` to use to later replace rare words.
#

SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = defaultdict(lambda: UNK_token)
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS",
                           PAD_token: "PAD", UNK_token: "UNK"}
        self.n_words = 4  # Count SOS, EOS, PAD and UNK

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


