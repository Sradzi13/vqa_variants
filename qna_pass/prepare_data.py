import math
import numpy as np
from read_input import readInput, readCaptions

BATCH_SIZE = 100
MAX_QA_LENGTH = 20
MAX_CAP_LENGTH = 20

######################################################################
# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 20 words (that includes
# ending punctuation)
#

def filterTriple(t):
    return len(t[1].split(' ')) < MAX_QA_LENGTH and \
        len(t[2].split(' ')) < MAX_QA_LENGTH

def filterTriples(triples):
    return [triple for triple in triples if filterTriple(triple)]

def filterPair(p):
    return len(t[1].split(' ')) < MAX_CAP_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#

def prepareData(vatt_file, qns_file, ans_file):
    input_lang, output_lang, triples = readInput(vatt_file, qns_file, ans_file)
    print("Read %s sentence triples" % len(triples))
    triples = filterTriples(triples)
    print("Trimmed to %s sentence pairs" % len(triples))
    print("Counting words...")
    for triple in triples:
        input_lang.addSentence(triple[1])
        output_lang.addSentence(triple[2])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    length = len(triples)
    print('length', length)
    triples_batch = batch_examples(triples, False)
    return input_lang, output_lang, triples_batch, length

def batch_examples(examples, shuffle):
    """
    groups the examples into batches so that we can train in batches
    :param examples:
    :return:
    """
    examples_batch = [batch for batch in batch_iter(examples, BATCH_SIZE, shuffle)]
    return examples_batch


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of data

    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]
        yield examples

def prepareCaptions(vatt_file, caption_file):
    """
    Produces pairs matching vatt to captions
    @return cap_lang, pairs, length
    """
    cap_lang, pairs = readCaptions(vatt_file, caption_file)
    print("Read %s images" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s images" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        cap_lang.addSentence(pair[1])
    print("Counted words:")
    print(cap_lang.name, cap_lang.n_words)
    length = len(pairs)
    print('length', length)
    pairs_batch = batch_examples(pairs, true)
    return cap_lang, pairs_batch, length


def shuffle_batched_pairs(examples_batch):
    """
    reshuffle and rebatch batched examples
    """
    examples = []
    for batch in examples_batch:
        examples.extend(batch)
    return batch_examples(examples, shuffle)

