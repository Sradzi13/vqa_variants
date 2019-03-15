import json
from lang import Lang

######################################################################
# Loading data files
# ==================
#
# The data for this project is a set of many thousands of English to
# French translation pairs.
#
# `This question on Open Data Stack
# Exchange <https://opendata.stackexchange.com/questions/3888/dataset-of-sentences-translated-into-many-languages>`__
# pointed me to the open translation site https://tatoeba.org/ which has
# downloads available at https://tatoeba.org/eng/downloads - and better
# yet, someone did the extra work of splitting language pairs into
# individual text files here: https://www.manythings.org/anki/
#
# The English to French pairs are too big to include in the repo, so
# download to ``data/eng-fra.txt`` before continuing. The file is a tab
# separated list of translation pairs:
#
# ::
#
#     I am cold.    J'ai froid.
#
# .. Note::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/data.zip>`_
#    and extract it to the current directory.

######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
#

def readInput(vatt_file, question_file, answer_file):
    triples = []
    with open(vatt_file) as v:
        vatts = json.load(v)
    with open(question_file) as q:
        questions = json.load(q)
    with open(answer_file) as a:
        answers = json.load(a)
    n = len(questions['questions'])
    for i in range(n):
        img_id = questions['questions'][i]['image_id']
        qns = questions['questions'][i]['question']
        vatt = vatts['COCO_train2014_{:012d}.jpg'.format(img_id)]
        vatt += [0] * (20*6-len(vatt))
        for j in range(10):
            ans = answers['annotations'][i]['answers'][j]['answer']
            triples.append([vatt, qns, ans])

    return Lang('qns'), Lang('ans'), triples


def readCaptions(vatt_file, caption_file):
    pairs = []
    with open(vatt_file) as v:
        vatts = json.load(v)
    with open(caption_file) as c:
        caps = json.load(c)
    n = len(caps)
    for img in vatts:
        cap = caps[vatt]
        vatt = vatts[img]
        pairs.append([vatt, cap])

    return Lang('cap'), pairs
