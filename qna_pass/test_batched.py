import models 
import argparse
import torch
import json
import random
from prepare_data import prepareData, batch_examples
from lang import SOS_token, EOS_token, PAD_token, UNK_token


BATCH_SIZE = 100
MAX_LENGTH = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def padSentInds(lang, sentence):
    no_padding = [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]
    padded = no_padding + ([PAD_token] * (MAX_LENGTH - len(no_padding)))
    return padded

def indexesFromBatch(input_lang, data_batched):
    vatt_list = [triple[0] for triple in data_batched]
    vatt20_list = [triple[1] for triple in data_batched]
    vknow_list = [triple[2] for triple in data_batched]
    input_list = [padSentInds(input_lang, triple[3]) for triple in data_batched]
    target_length = [triple[4] for triple in data_batched]
    return vatt_list, vatt20_list, vknow_list, input_list

def tensorFromBatch(input_lang, data_batched):
    vatt_list, vatt20_list, vknow_list, input_list, target_length_list = indexesFromBatch(input_lang, data_batched)
    batch_size = len(vatt_list)
    vatt_tensor = torch.tensor(vatt_list, device=device, dtype=torch.float).view(1, batch_size, -1)
    vatt20_tensor = torch.tensor(vatt20_list, device=device, dtype=torch.float).view(1, batch_size, -1)
    vknow_tensor = torch.tensor(vknow_list, device=device, dtype=torch.float).view(1, batch_size, -1)
    input_tensor = torch.tensor(input_list, dtype=torch.long, device=device).view(-1, batch_size)
    target_length_tensor = torch.tensor(target_length_list, device=device, dtype=torch.float).view(1, batch_size, -1)
    return vatt_tensor, vatt20_tensor, vknow_tensor, input_tensor, target_length_tensor


def generate(vatt_tensor, vatt20_tensor, vknow_tensor,input_tensor,target_length, device=device):
    encoder_hidden = encoder.initHidden()
    input_length = input_tensor.size(0)
    decoded = []

    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)

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
    return decoded    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('encoder_file', help='encoder file')
    parser.add_argument('decoder_file', help='decoder file')
    parser.add_argument('vatt', help='vatt file')
    parser.add_argument('vatt20', help='vatt20 file')
    parser.add_argument('vknow', help='vknow file')
    parser.add_argument('qns', help='qns file')
    parser.add_argument('ans', help='ans file')

    args = parser.parse_args()
    print('here instead')
    encoder = torch.load(args.encoder_file, map_location=device)
    
    decoder = torch.load(args.decoder_file, map_location=device)
    print('he')
    with open(args.vatt) as f:
        vatts = json.load(f)
    with open(args.vatt20) as f:
        vatts20 = json.load(f)
    with open(args.vknow) as f:
        vknows = json.load(f)
    with open(args.qns) as f:
        questions = json.load(f)
    with open(args.ans) as f:
        answers = json.load(f)

    print('here')
    input_lang, output_lang, triples, nExamples = prepareData(
            '../vecs_train/train_vatt_conv.json',
            '../Vatt/Vatt20.json', 
            '../vecs_train/train_vknow.json',
            '../qna_training_coco/v2_OpenEnded_mscoco_train2014_questions.json',
            '../qna_training_coco/v2_mscoco_train2014_annotations.json')

    nQns = len(questions['questions'])
    generated_dict = {}

    data = []
    for i in range(nQns):
        iId = questions['questions'][i]["image_id"]
        qn = questions['questions'][i]["question"]

        vatt_string = vatts['COCO_val2014_{:012d}.jpg'.format(questions['questions'][i]["image_id"])].split(' ')
        vatt = [float(k) for k in vatt_string]
        vatt_tensor = torch.tensor(vatt, dtype=torch.long, device=device)

        vatt20 = vatts20['COCO_val2014_{:012d}.jpg'.format(questions['questions'][i]["image_id"])]
        vatt20 += [0] * (20*6-len(vatt20))
        vatt20_tensor = torch.tensor(vatt20, dtype=torch.long, device=device)

        vknow_string = vknows['COCO_val2014_{:012d}.jpg'.format(questions['questions'][i]["image_id"])].split(' ')
        vknow = [float(j) for j in vknow_string]
        vknow_tensor = torch.tensor(vknow, dtype=torch.long, device=device)

        anses = [a["answer"] for a in answers['annotations'][i]['answers']]
        target_length = max(len(ans) for ans in anses)


        data.append([vatt, vatt20, vknow, qn, anses, target_length])

    data_batched = batch_examples(data, False)
    training_batches = [tensorFromBatch(input_lang, i) for i in data_batched]

    n_iters = int(len(training_batches) / BATCH_SIZE)
    for iter in range(1, n_iters + 1):
        training_batch = training_batches[iter - 1]

        vatt_tensor = training_batch[0]  # v_att * batchsz
        vatt20_tensor = training_batch[1]
        vknow_tensor = training_batch[2]
        input_tensor = training_batch[3]
        target_len_tensor = training_batch[4]
        generated = generate(vatt_tensor, vatt20_tensor, vknow_tensor, input_tensor, target_len_tensor, device=device)

        gen_sent = []
        for m in generated:
            gen_sent.append(output_lang.index2word[m])
        #print(gen_sent)
        generated_dict["question_id"] = str(gen_sent)


        print(qn)
        print(anses)
        print(gen_sent)
        #print(generated)
        
    with('generated.json', 'w') as f:
        json.dump(generated_dict, f)

    with('used_questions.json', 'w') as f:
        json.dump(qn)

    with('answers.json', 'w') as f:
        json.dump(ans)
       
   
        
