from models import *
import argparse
import torch
import json
import random
from prepare_data import prepareData_noVcap, tensorFromSentence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(vatt_tensor, vatt20_tensor, vknow_tensor,input_tensor,target_length, device):
    encoder_hidden = encoder.initHidden()
    input_length = input_tensor.size(0)
    decoded = []

    encoder_output, encoder_hidden = encoder.special_forward(
        vatt_tensor, vatt20_tensor, vknow_tensor, encoder_hidden)
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
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('encoder_file', help='encoder file')
    parser.add_argument('decoder_file', help='decoder file')
    parser.add_argument('vatt', help='vatt file')
    parser.add_argument('vatt20', help='vatt20 file')
    parser.add_argument('vknow', help='vknow file')
    parser.add_argument('qns', help='qns file')
    parser.add_argument('ans', help='ans file')
    parser.add_argument('num', help='number of questions', type=int)

    args = parser.parse_args()

    encoder = torch.load(args.encoder_file)
    decoder = torch.load(args.decoder_file)
    vatts = json.load(args.vatt)
    vatts20 = json.load(args.vatt20)
    vknows = json.load(args.vknow)

    questions = json.load(args.qns)
    answers = json.load(args.ans)
    num = args.num

    input_lang, output_lang, triples, nExamples = prepareData_noVcap(args.vatt, args.vatt20, args.vknow,
            '../qna_training_coco/v2_OpenEnded_mscoco_train2014_questions.json',
            '../qna_training_coco/v2_mscoco_train2014_annotations.json')

    nQns = len(questions['questions'])


    for i in range(num):
        
        randQ = random.randint(nQns)
        qId = questions['questions'][randQ]["question_id"]
        qn = questions['questions'][randQ]["question"]

        vatt_string = vatts['COCO_train2014_{:012d}.jpg'.format(questions['questions'][randQ]["image_id"])].split(' ')
        vatt = [float(i) for i in vatt_string]
        vatt_tensor = torch.tensor(vatt, dtype=torch.long, device=device)

        vatt20 = vatts20['COCO_train2014_{:012d}.jpg'.format(questions['questions'][randQ]["image_id"])]
        vatt20 += [0] * (20*6-len(vatt20))
        vatt20_tensor = torch.tensor(vatt20, dtype=torch.long, device=device)

        vknow_string = vknows['COCO_train2014_{:012d}.jpg'.format(questions['questions'][randQ]["image_id"])].split(' ')
        vknow = [float(i) for i in vknow_string]
        vknow_tensor = torch.tensor(vknow, dtype=torch.long, device=device)

        anses = [a["answer"] for a in answers['annotations'][i]['answers']]
        for j in range(10):
            
            anses.append(answers)

        input_tensor = tensorFromSentence(input_lang, qn)
        target_length = max(len(ans) for ans in anses)
        generated = generate(vatt_tensor, vatt20_tensor, vknow_tensor, input_tensor,target_length)

        print(qn)
        print(anses)
        print(generated)
        
    

   
        