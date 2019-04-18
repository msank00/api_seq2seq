from flask import Flask, request, jsonify, Response
import json
import pandas as pd
import numpy as np
import logging
import argparse


from util_model import *
from smutil import *
import random
import matplotlib.ticker as ticker
import os
import yaml
import torch

import warnings

warnings.filterwarnings("ignore")


# UNCOMMENT THE FOLLOWING LINE FOR OPTIONAL ARGUMENT PARSER
# setting optional argument parser
parser = argparse.ArgumentParser(description="Get Hosting parameters")
parser.add_argument("--optHost", type=str, help="An optional Host Name")
parser.add_argument("--optPort", type=int, help="An optional port Number")
parser.add_argument("--logLevel", type=str, help="Logging level")
args = parser.parse_args()

# creating an instance of the Flask APP
app = Flask(__name__)


# ============================================================


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


with open("config.yaml", "r") as stream:
    config = yaml.load(stream)

dirData = config["directory"]["data"]
dirModel = config["directory"]["model"]
dirOutput = config["directory"]["output"]
dirLog = config["directory"]["log"]

input_lang, output_lang, pairs = prepareData("eng", "fra", dirData, True)


# ============================================================


@app.route("/predict", methods=["POST"])
def predictionUtility():

    # logging client information
    app.logger.info("+++++ REQUEST RECEIVED +++++")
    req_method = request.environ["REQUEST_METHOD"]
    req_api = request.environ["PATH_INFO"]
    req_http_user_agent = request.environ["HTTP_USER_AGENT"]
    req_remote_address = request.environ["REMOTE_ADDR"]

    app.logger.info("REQUEST METHOD: %s" % (req_method))
    app.logger.info("REQUEST API: %s" % (req_api))
    app.logger.info("HTTP USER AGENT: %s" % (req_http_user_agent))
    app.logger.info("CLIENT ADDRESS: %s" % req_remote_address)

    try:
        # extrating data from the client request
        content = request.json  # get_json(silent=True)
        content = json.dumps(content)
        contentdf = pd.read_json(content, orient="records")

        # passing data to prediction module
        output = predictionModule(contentdf)
        outjson = output.to_json(orient="records")

        # creating the response to send back to the client
        resp = Response(outjson, status=200, mimetype="application/json")
        app.logger.info("----- REQUEST SERVED -----")

        return resp
    except Exception as e:
        app.logger.exception("message")
        raise Exception(e)


def predict_util(input_sentence):

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

    output_sentence = predict(
        input_sentence, model_encoder, model_decoder, dirOutput, input_lang, device
    )

    # print("input >>", input_sentence)
    # print("output <<", output_sentence)
    # line_break(headline="DONE")

    app.logger.info("prediction completed....")

    return output_sentence


def predictionModule(inputData):

    try:
        print(inputData)
        input_sentence = "elle a cinq ans de moins que moi ."

        # ====================================================
        # all the prediction algorithm function calls goes here

        output_sentence = predict_util(input_sentence)
        # ====================================================

        # dummy logic to get random prediction score
        n = inputData.shape[0]
        app.logger.info("Number of recorde: %d" % (n))

        outdf = pd.DataFrame()
        outdf["UID"] = inputData["UID"]
        outdf["output"] = output_sentence  # np.random.uniform(0,1,n)

        return outdf
    except Exception as e:
        app.logger.exception("message")
        raise Exception(e)


def configLogging(logPath):
    if args.logLevel and len(args.logLevel) > 0:
        if args.logLevel.upper() == "INFO":
            logLvl = logging.INFO
        elif args.logLevel.upper() == "DEBUG":
            logLvl = logging.DEBUG
        elif args.logLevel.upper() == "WARNING":
            logLvl = logging.WARNING
        elif args.logLevel.upper() == "ERROR":
            logLvl = logging.ERROR
        else:
            logLvl = logging.INFO
    else:
        logLvl = logging.INFO

    from logging import Formatter

    fileHandler = logging.FileHandler(logPath)
    fileHandler.setFormatter(
        Formatter(
            "%(asctime)s %(levelname)s: %(message)s " "[in %(pathname)s:%(lineno)d]"
        )
    )
    fileHandler.setLevel(logLvl)
    app.logger.addHandler(fileHandler)
    app.logger.setLevel(logLvl)


if __name__ == "__main__":

    # load ip and port from the config file
    ip = str(config["api"]["predict"]["ip"])
    port = int(config["api"]["predict"]["port"])  #  5000

    logPath = os.path.join(dirLog, "python_prediction_API_server.log")

    configLogging(logPath)

    app.logger.info("SERVER STARTED ON  %s:%s" % (ip, port))
    app.run(host=ip, port=port)
