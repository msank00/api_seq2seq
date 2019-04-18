from util_model import *
from smutil import *
import random
import matplotlib.ticker as ticker
import yaml

import warnings

warnings.filterwarnings("ignore")

import json
import torch


SEED_rand = 2019
random.seed(SEED_rand)


teacher_forcing_ratio = 0.5


def train(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    max_length=MAX_LENGTH,
):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(
    encoder,
    decoder,
    input_lang,
    output_lang,
    device,
    dirOutput,
    n_iters,
    print_every=1000,
    plot_every=100,
    learning_rate=0.01,
):
    line_break(headline="train started")
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [
        tensorsFromPair(random.choice(pairs), input_lang, output_lang, device)
        for i in range(n_iters)
    ]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(
            input_tensor,
            target_tensor,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                "%s \t(%d %d%%)\t%.4f"
                % (
                    timeSince(start, iter / n_iters),
                    iter,
                    iter / n_iters * 100,
                    print_loss_avg,
                )
            )

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    line_break(headline="train ended")
    print("saving loss curve...")
    showPlot(plot_losses, dirOutput)


def showPlot(points, dirOutput):
    with mpl.style.context("seaborn-talk"):
        fig = plt.figure(figsize=(5, 4))
        plt.plot(points, color="b", alpha=0.5, label="loss")
        plt.title("Loss vs Epoch", fontweight="bold")
        plt.xlabel("Epoch", fontweight="bold")
        plt.ylabel("Loss", fontweight="bold")
        plt.grid(color="gray", alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(dirOutput + "/loss_curve.png")


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
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


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print(">", pair[0])
        print("=", pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("loading connfig file...")

with open("config.yaml", 'r') as stream:
    config = yaml.load(stream)

dirData = config['directory']['data']
dirModel = config['directory']['model']
dirOutput = config['directory']['output']

print("creating directories....")

create_directory(dirData)
create_directory(dirModel)
create_directory(dirOutput)

input_lang, output_lang, pairs = prepareData("eng", "fra", dirData, True)
print(random.choice(pairs))

hidden_size = 256

line_break(headline="hyper-parameter info")
print("hidden dimensoin size: {}".format(hidden_size))
print("input lang: {} words".format(input_lang.n_words))
print("output lang: {} words".format(output_lang.n_words))


print("initializing the model...")
encoder1 = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(
    device
)

trainIters(
    encoder1,
    attn_decoder1,
    input_lang,
    output_lang,
    device,
    dirOutput,
    n_iters=400,
    print_every=20,
    plot_every=2,
)

print("saving model....")

# encoder1, attn_decoder1
file_encoder = "encoder.pt"
file_decoder = "attn_decoder.pt"
torch.save(encoder1.state_dict(), os.path.join(dirModel, file_encoder))
torch.save(attn_decoder1.state_dict(), os.path.join(dirModel, file_decoder))
print("model saved...")

print("saving hyper-parameter and meta information in json...")
dict_param = {
    "hidden_size": hidden_size,
    "n_words_iplang": input_lang.n_words,
    "n_words_oplang": output_lang.n_words,
    "model_file_encoder": file_encoder,
    "model_file_decoder": file_decoder
}

with open(os.path.join(dirOutput, "hyper_param.json"), "w") as fp:
    json.dump(dict_param, fp)

print("hyper-param saved...")

line_break(headline="DONE")