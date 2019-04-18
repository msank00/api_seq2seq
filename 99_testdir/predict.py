from util_model import *
from smutil import *
import random
import matplotlib.ticker as ticker
import os
import json
import yaml

import warnings

warnings.filterwarnings("ignore")

import torch


def diaplay_Attention(input_sentence, output_words, attentions, dirOutput):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy())
    fig.colorbar(cax, fraction=0.02, pad=0.06)

    # Set up axes
    ax.set_xticklabels([""] + input_sentence.split(" ") + ["<EOS>"], rotation=90)
    ax.set_yticklabels([""] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    plt.savefig(os.path.join(dirOutput, "attention_test_data.png"))


def predict(input_sentence, encoder, decoder, dirOutput, input_lang, device):
    output_words, attentions = evaluate(
        encoder, decoder, input_sentence, input_lang, device
    )

    output_sentence = " ".join(output_words)
    diaplay_Attention(input_sentence, output_words, attentions, dirOutput)
    return output_sentence


def evaluate(encoder, decoder, sentence, input_lang, device, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[: di + 1]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device type: {}".format(device))


with open("config.yaml", 'r') as stream:
    config = yaml.load(stream)

dirData = config['directory']['data']
dirModel = config['directory']['model']
dirOutput = config['directory']['output']

input_lang, output_lang, pairs = prepareData("eng", "fra", dirData, True)

def predict_util():

    with open(os.path.join(dirOutput, "hyper_param.json"), "r") as fp:
        param = json.load(fp)

    hidden_size = param["hidden_size"]
    n_words_input_lang = param["n_words_iplang"]
    n_words_output_lang = param["n_words_oplang"]

    file_encoder = param["model_file_encoder"]
    file_decoder = param["model_file_decoder"]

    model_encoder = EncoderRNN(n_words_input_lang, hidden_size, device).to(device)
    model_encoder.load_state_dict(torch.load(os.path.join(dirModel, file_encoder)))
    model_encoder.eval()

    model_decoder = AttnDecoderRNN(hidden_size, n_words_output_lang, dropout_p=0.1).to(
        device
    )
    model_decoder.load_state_dict(torch.load(os.path.join(dirModel, file_decoder)))
    model_decoder.eval()

    
    # print(random.choice(pairs))

    input_sentence = "elle a cinq ans de moins que moi ."
    output_sentence = predict(input_sentence, model_encoder, model_decoder, dirOutput, input_lang, device)

    print("input >>", input_sentence)
    print("output <<", output_sentence)

    line_break(headline="PREDICTION COMPLETED")

predict_util()