from models import *
import argparse
import torch
import json
import random
from models import tensorFromSentence, prepareData2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 20


def generate(vatt_tensor, input_tensor, target_length, device):
    encoder_hidden = encoder.initHidden()
    input_length = input_tensor.size(0)
    decoded = []

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    encoder_output, encoder_hidden = encoder.special_forward(
        vatt_tensor, encoder_hidden)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        if decoder_input.item() == EOS_token:
            break
        decoded.append(decoder_input.item())

    return decoded


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('encoder_file', help='encoder file')
    parser.add_argument('decoder_file', help='decoder file')
    parser.add_argument('vatt', help='vatt file')
    parser.add_argument('qns', help='qns file')
    parser.add_argument('ans', help='ans file')
    parser.add_argument('num', help='number of questions', type=int)
    args = parser.parse_args()

    encoder = torch.load(args.encoder_file, map_location='cpu')
    decoder = torch.load(args.decoder_file, map_location='cpu')

    with open(args.vatt) as f:
        vatts = json.load(f)
    with open(args.qns) as f:
        questions = json.load(f)
    with open(args.ans) as f:
        answers = json.load(f)
    num = args.num

    input_lang, output_lang, triples, nExamples = prepareData2('../Vatt/Vatt20.json',
                                                               '../qna_training_coco/v2_OpenEnded_mscoco_train2014_questions.json',
                                                               '../qna_training_coco/v2_mscoco_train2014_annotations.json')

    nQns = len(questions['questions'])

    generated_dict = {}

    for i in range(nQns):
        iId = questions['questions'][i]["image_id"]
        qn = questions['questions'][i]["question"]
        vatt = vatts['COCO_train2014_{:012d}.jpg'.format(iId)]
        vatt += [0] * (20 * 6 - len(vatt))
        vatt_tensor = torch.tensor(vatt, dtype=torch.float, device=device)

        anses = [a["answer"] for a in answers['annotations'][i]['answers']]

        input_tensor = tensorFromSentence(input_lang, qn)
        target_length = max(len(ans) for ans in anses)
        generated = generate(vatt_tensor, input_tensor, target_length, device)
        gen_sent = []
        for i in generated:
            gen_sent.append(output_lang.index2word[i])

        generated_dict["question_id"] = str(gen_sent)

        print('imageID: ' + str(iId))
        print(qn)
        print(anses)
        print(gen_sent)







