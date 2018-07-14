import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import torch.cuda
import argparse
import json
import os
import time
from tensorboardX import SummaryWriter

from copy import copy
from tqdm import tqdm, trange
from help_functions import *
from model import *
from random import randint
from nltk.tokenize.moses import MosesDetokenizer

# the path for the glove word2vec file

detokenizer = MosesDetokenizer()
writer = SummaryWriter('runs/lr10powerminustwo13rd')

# Hyper-Parameters:
hidden_size = 100
pooling_size = 16
train_word_embeddings = False
number_of_iterations = 4
batch_size = 32
learning_rate = 0.002 / 3
num_epochs = 75
redo = 0
regularization_param = 0.005
dropout_fraction = 0.35
grad_clip = 5
loss_calc_after_every = 15
number_of_layers = 2
print_to_file_after_every = 100
apply_batch_norm = True
# after_every_n = 6

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Question Answering')
    parser.add_argument('--lr', type=float, default=learning_rate, help='Learning Rate')
    parser.add_argument('--epochs', type=float, default=num_epochs, help='No of Epochs')
    parser.add_argument('--redo', type=int, default=redo, help='Preprocess whole data again or not')
    parser.add_argument('--regu', type=float, default=regularization_param, help='L2 Regularization Parameter')
    parser.add_argument('--dropout', type=float, default=dropout_fraction, help='Dropout')
    parser.add_argument('--clip', type=float, default=grad_clip, help='Gradient Clipping')
    args = parser.parse_args()

    print("Learning Rate -> ", args.lr)
    print("Lambda -> ", args.regu)
    print("Dropout -> ", args.dropout)
    print("Batch Size ->", batch_size)
    print("Pooling Size ->", pooling_size)
    print("Gradient Clipping ->", args.clip)

    count = 1
    start_from = 0
    f = open('./results/result_' + str(count) + '.txt', 'w')
    g = open('./results/train_answers_' + str(count) + '.txt', 'w')

    embedding_matrix = word_embedding_layer()
    word_to_idx, idx_to_word = word_idx_map()

    Model = Model(hidden_size, embedding_matrix, train_word_embeddings, args.dropout, pooling_size, number_of_iterations, number_of_layers).cuda()
    loss_function = nn.CrossEntropyLoss()

    parameters = list(Model.parameters())
    filtered_params = filter(lambda x: x.requires_grad, parameters)

    optimizer = optim.Adam(filtered_params, lr=args.lr, weight_decay=args.regu)

    _, _, _, _, train_spans, train_token_passages, train_token_questions, train_num_passages, train_num_questions,\
    train_p_lengths, train_q_lengths = load_data('train')

    max_question_length, max_passage_length = max(train_q_lengths), max(train_p_lengths)

    val_ids, _, _, _, val_spans, val_token_passages, val_token_questions,\
    val_num_passages, val_num_questions, val_passage_lengths,val_questions_lengths = load_data('dev')
    max_val_ques_len, max_val_pass_len = max(val_questions_lengths), max(val_passage_lengths)

    ids_removed = ids_removed_from_dev()
    json_file_path = dev_json_path()

    egs_in_training_batch = get_batch_distribution(batch_size, len(train_num_passages))
    egs_in_validation_batch = get_batch_distribution(batch_size, len(val_num_passages))

    bar = trange(args.epochs)

    train_counting = 0
    dev_counting = 0
    for i in bar:
        train_spans, train_token_passages, train_token_questions, train_num_passages, train_num_questions, train_p_lengths, train_q_lengths\
        = shuffle_data(train_spans, train_token_passages, train_token_questions, train_num_passages, train_num_questions,train_p_lengths, train_q_lengths)

        f.write("Epoch->" + str(i + 1 + start_from) + '\n')
        g.write("Epoch->" + str(i + 1 + start_from) + '\n')
        f.write("learning_rate-> " + str(optimizer.param_groups[0]['lr']) + '\n')

        training_losses, dev_losses = 0, 0
        bar1 = trange(len(egs_in_training_batch), desc='Training.....')
        n_batch_loss = 0

        ## ----------------------------TRAIN----------------------- ##
        for batch in bar1:

            number_of_examples = egs_in_training_batch[batch]
            batch_passages = torch.tensor(copy(train_num_passages[number_of_examples * batch: number_of_examples * (batch + 1)])).cuda()
            batch_questions = torch.tensor(copy(train_num_questions[number_of_examples * batch: number_of_examples * (batch + 1)])).cuda()
            batch_spans = copy(train_spans[number_of_examples * batch: number_of_examples * (batch + 1)])
            batch_p_lengths = torch.tensor(copy(train_p_lengths[number_of_examples * batch: number_of_examples * (batch + 1)])).cuda()
            batch_q_lengths = torch.tensor(copy(train_q_lengths[number_of_examples * batch: number_of_examples * (batch + 1)])).cuda()
            batch_token_passages = copy(train_token_passages[number_of_examples * batch: number_of_examples * (batch + 1)])
            batch_token_questions = copy(train_token_questions[number_of_examples * batch: number_of_examples * (batch + 1)])

            optimizer.zero_grad()

            start_outputs, end_outputs, entropies = Model(max_passage_length, max_question_length, batch_passages, batch_questions, batch_p_lengths, \
                                                          batch_q_lengths, number_of_examples, apply_batch_norm, True)

            '''
                    start_outputs, end_outputs -> batch_size
                    entropies -> [ [(batch_size*passage_len),(batch_size*passage_len)], .......   ]

            '''
            s_ents, e_ents = list(zip(*copy(entropies)))

            batch_start_end = list(zip(*copy(batch_spans)))
            start_spans = torch.tensor(batch_start_end[0]).cuda()
            end_spans = torch.tensor(batch_start_end[1]).cuda()

            loss_start, loss_end = 0, 0

            for m in range(len(entropies)):
                loss_start += loss_function(s_ents[m], start_spans.view(-1))
                loss_end += loss_function(e_ents[m], end_spans.view(-1))

            loss = loss_start + loss_end

            loss.backward()
            nn.utils.clip_grad_norm_(filtered_params, args.clip)
            optimizer.step()

            n_batch_loss += loss.item()

            if batch % print_to_file_after_every == 0:
                random_eg = randint(0, number_of_examples - 1)
                if start_outputs[random_eg].item() <= end_outputs[random_eg].item():

                    list_para = batch_token_passages[random_eg]
                    text_para = detokenizer.detokenize(list_para, return_str=True)
                    text_ques = detokenizer.detokenize(batch_token_questions[random_eg] ,return_str=True)

                    actual_answer = detokenizer.detokenize(
                        list_para[start_spans[random_eg].item(): end_spans[random_eg].item() + 1], return_str=True)
                    answer_string = detokenizer.detokenize(
                        list_para[start_outputs[random_eg].item(): end_outputs[random_eg].item() + 1], return_str=True)

                    g.write("Passage-> " + str(text_para) + '\n\n')
                    g.write("Question-> " + str(text_ques) + '\n\n')
                    g.write("Actual Answer-> " + str(actual_answer) + '\n\n')
                    g.write("Predicted Answer-> " + str(answer_string) + '\n\n\n\n')

                else:
                    list_para = batch_token_passages[random_eg]
                    text_para = detokenizer.detokenize(list_para, return_str=True)
                    text_ques = detokenizer.detokenize(batch_token_questions[random_eg] ,return_str=True)

                    actual_answer = detokenizer.detokenize(
                        list_para[start_spans[random_eg].item(): end_spans[random_eg].item() + 1], return_str=True)
                    answer_string = detokenizer.detokenize(
                        list_para[end_outputs[random_eg].item(): start_outputs[random_eg].item() + 1], return_str=True)

                    g.write("Passage-> " + str(text_para) + '\n\n')
                    g.write("Question-> " + str(text_ques) + '\n\n')
                    g.write("Actual Answer-> " + str(actual_answer) + '\n\n')
                    g.write("Predicted Answer-> " + str(answer_string) + '\n\n\n\n')

            g.flush()

            if (batch + 1) % loss_calc_after_every == 0 and batch != 0:
                train_counting += 1
                bar1.set_postfix(loss_="{0:0.5f}".format(n_batch_loss / loss_calc_after_every))
                training_losses += n_batch_loss
                writer.add_scalar('Batch Loss training', n_batch_loss / loss_calc_after_every, train_counting)
                n_batch_loss = 0

            elif batch == len(egs_in_training_batch) - 1:
                train_counting += 1
                divide = (len(egs_in_validation_batch) % loss_calc_after_every)
                bar1.set_postfix(loss_="{0:0.5f}".format(n_batch_loss / divide))
                writer.add_scalar('Batch Loss training', n_batch_loss / divide, train_counting)
                training_losses += n_batch_loss
                n_batch_loss = 0

        torch.save(Model, './saved_model/Model_epoch_' + str(i + 1 + start_from) + '.pt')

        # ------------------------VALIDATE-----------------------#

        results = {}

        bar2 = trange(len(egs_in_validation_batch), desc='Validating.....')

        for batch in bar2:

            number_of_examples = egs_in_validation_batch[batch]
            batch_passages = torch.tensor(copy(val_num_passages[number_of_examples * batch: number_of_examples * (batch + 1)])).cuda()
            batch_questions = torch.tensor(copy(val_num_questions[number_of_examples * batch: number_of_examples * (batch + 1)])).cuda()
            batch_spans = copy(val_spans[number_of_examples * batch: number_of_examples * (batch + 1)])
            batch_ids = copy(val_ids[number_of_examples * batch: number_of_examples * (batch + 1)])
            batch_text_para = copy(val_token_passages[number_of_examples * batch: number_of_examples * (batch + 1)])
            batch_p_lengths = torch.tensor(copy(val_passage_lengths[number_of_examples * batch: number_of_examples * (batch + 1)])).cuda()
            batch_q_lengths = torch.tensor(copy(val_questions_lengths[number_of_examples * batch: number_of_examples * (batch + 1)])).cuda()

            start_outputs, end_outputs, entropies = Model(max_val_pass_len, max_val_ques_len, batch_passages, batch_questions, batch_p_lengths, \
                                                          batch_q_lengths, number_of_examples, False, False)

            '''
                    start_outputs, end_outputs -> batch_size
                    entropies -> [ [(batch_size*passage_len),(batch_size*passage_len)], .......   ]

            '''
            s_ents, e_ents = list(zip(*copy(entropies)))

            batch_start_end = list(zip(*copy(batch_spans)))
            start_spans = torch.tensor(batch_start_end[0]).cuda()
            end_spans = torch.tensor(batch_start_end[1]).cuda()

            loss_start, loss_end = 0, 0

            for m in range(len(entropies)):
                loss_start += loss_function(s_ents[m], start_spans.view(-1))
                loss_end += loss_function(e_ents[m], end_spans.view(-1))

            loss = loss_start + loss_end

            for j in range(len(batch_ids)):

                if start_outputs[j].item() <= end_outputs[j].item():
                    answer_str = detokenizer.detokenize(
                        batch_text_para[j][start_outputs[j].item():end_outputs[j].item() + 1], return_str=True)
                    results[str(batch_ids[j])] = str(answer_str)
                else:
                    answer_str = detokenizer.detokenize(
                        batch_text_para[j][end_outputs[j].item():start_outputs[j].item() + 1], return_str=True)
                    results[str(batch_ids[j])] = str(answer_str)

            n_batch_loss += loss.item()

            if (batch + 1) % loss_calc_after_every == 0 and batch != 0:
                dev_counting += 1
                bar2.set_postfix(loss_="{0:0.5f}".format(n_batch_loss / loss_calc_after_every))
                writer.add_scalar('Batch Loss Validation', n_batch_loss / loss_calc_after_every, dev_counting)
                dev_losses += n_batch_loss
                n_batch_loss = 0

            elif batch == len(egs_in_validation_batch) - 1:
                dev_counting += 1
                divide = (len(egs_in_validation_batch) % loss_calc_after_every)
                writer.add_scalar('Batch Loss Validation', n_batch_loss / divide, dev_counting)
                bar2.set_postfix(loss_="{0:0.5f}".format(n_batch_loss / divide))
                dev_losses += n_batch_loss
                n_batch_loss = 0

        for k in ids_removed:
            results[str(k)] = ""

        obj = json.dumps(results)
        ss = open("./results/answers.txt", 'w')
        ss.write(obj)
        ss.flush()
        ss.close()

        os.system("python evaluate-v2.0\ \(1\).py" + str(json_file_path) + " ./results/answers.txt")

        time.sleep(1)

        with open('./results/f1_exact.pickle', 'rb') as h:
            f1, exact = pickle.load(h)

        bar.set_postfix(tr_loss=training_losses / len(egs_in_training_batch),
                        dev_loss=dev_losses / len(egs_in_validation_batch), f1=f1, exact=exact)

        f.write("\nTraining Loss Epoch " + str(i + 1 + start_from) + "--> " + str(
            training_losses / len(egs_in_training_batch)) + '\n')
        f.write("Dev Loss Epoch " + str(i + 1 + start_from) + "--> " + str(
            dev_losses / len(egs_in_validation_batch)) + '\n')
        f.write("F1 " + str(i + 1 + start_from) + "--> " + str(f1) + '\n')
        f.write("Exact " + str(i + 1 + start_from) + "--> " + str(exact) + '\n\n')
        f.flush()
        writer.add_scalar('training loss', training_losses / len(egs_in_training_batch), i)
        writer.add_scalar('dev loss', dev_losses / len(egs_in_validation_batch), i)
        writer.add_scalar('F score', f1, i)
        writer.add_scalar('exact', exact, i)

    print('done!')
