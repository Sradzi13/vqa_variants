from models import *
import argparse
import torch
import json
import random
from models import tensorFromSentence, prepareData2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(vatt_tensor,input_tensor,target_length, device):
    encoder_hidden = encoder.initHidden()
    input_length = input_tensor.size(0)
    decoded = []

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
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('encoder_file', help='encoder file')
    parser.add_argument('decoder_file', help='decoder file')
    parser.add_argument('vatt', help='vatt file')
    parser.add_argument('qns', help='qns file')
    parser.add_argument('ans', help='ans file')
    parser.add_argument('num', help='number of questions', type=int)
    args = parser.parse_args()

    encoder = torch.load(args.encoder_file)
    decoder = torch.load(args.decoder_file)
    vatts = json.load(args.vatt)
    questions = json.load(args.qns)
    answers = json.load(args.ans)
    num = args.num

    input_lang, output_lang, triples, nExamples = prepareData2('../Vatt/Vatt20.json',
            '../qna_training_coco/v2_OpenEnded_mscoco_train2014_questions.json',
            '../qna_training_coco/v2_mscoco_train2014_annotations.json')

    nQns = len(qns['questions'])
    

    for i in range(num):
        
        randQ = random.randint(nQns)
        qId = questions['questions'][randQ]["question_id"]
        qn = questions['questions'][randQ]["question"]
        vatt = vatts['COCO_train2014_{:012d}.jpg'.format(questions['questions'][randQ]["image_id"])]
        vatt += [0] * (20*6-len(vatt))
        vatt_tensor = torch.tensor(vatt, dtype=torch.long, device=device)
        
        anses = [a["answer"] for a in answers['annotations'][i]['answers']]
        for j in range(10):
            
            anses.append(ans)

        input_tensor = tensorFromSentence(input_lang, qn)
        target_length = max(len(ans) for ans in anses)
        generated = generate(vatt_tensor,input_tensor,target_length)

        print(qn)
        print(anses)
        print(generated)
        
    

   
        