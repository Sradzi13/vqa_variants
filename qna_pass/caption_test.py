import argparse
import dill
import torch
import json
import random

from lang import SOS_token
from models_batch_caption import tensorsFromPairs
import prepare_data

MAX_CAP_LENGTH = 20
BATCH_SIZE = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(vatt_tensor, caption_length, device):
    batch_size = vatt_tensor.shape[1]
    decoder_input = torch.tensor([[SOS_token] * batch_size], device=device)
    decoder_output, decoder_hidden = decoder.special_forward(vatt_tensor)

    decoded = []

    for di in range(caption_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        decoded.append(decoder_input)

    return decoded, decoder_hidden
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('decoder_file', help='decoder file')
    parser.add_argument('vatt', help='vatt file')
    parser.add_argument('lang', help='lang file')
    parser.add_argument('caps', help='caption file')
    parser.add_argument('num', help='number of images to process', type=int)
    parser.add_argument('output_file', help='vcaps file')
    args = parser.parse_args()

    decoder = torch.load(args.decoder_file)
    with open(args.lang, 'rb') as f:
        caption_lang = dill.load(f)
    num = args.num

    _, pairs_batch, length = prepare_data.prepareCaptions_with_names(args.vatt, args.caps, MAX_CAP_LENGTH)
    pairs_list = [(tensorsFromPairs(caption_lang, i), imgs_name)
                    for (i, imgs_name) in pairs_batch]

    vcaps = {}
    for (batch, imgs_name) in pairs_list:
        vatt_tensor, cap_tensor = batch
        decoded, decoder_hidden = generate(vatt_tensor, MAX_CAP_LENGTH, device)
        for i, name in enumerate(imgs_name):
            vcaps[name] = decoder_hidden[0][0, i].detach().cpu().numpy().tolist()

    with open(args.output_file, 'w') as f:
        json.dump(vcaps, f)

