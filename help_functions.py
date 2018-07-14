import torch
import torch.cuda
import pickle
import numpy as np
import json

from random import shuffle
from nltk import word_tokenize
from tqdm import tqdm

def word_embedding_layer():

        emb_weights = torch.load('./model_pickles/word_embeddings.pt')

        return emb_weights.cuda()


def word_idx_map():
    with open('./model_pickles/word_to_idx.pickle', 'rb') as h:
        word_to_idx = pickle.load(h)

    with open('./model_pickles/idx_to_word.pickle', 'rb') as h:
        idx_to_word = pickle.load(h)

    return word_to_idx, idx_to_word

def load_data(mode):

	if mode == 'train':
		
		with open('./model_pickles/training_data.pickle','rb') as h:
        	data = pickle.load(h)

    id mode == 'dev':
    	
    	with open('./model_pickles/validation_data.pickle','rb') as h:
        	data = pickle.load(h)


    '''
    [text_passage, text_question, text_answer, (start_span, end_span), 
     tokenized_passage, tokenized_question, tokenized_num_passage_padded, tokenized_num_question_padded]

    '''

    ids, text_passages, text_questions, text_answers, spans, token_passages, token_questions, num_passages, num_questions = [],[],[],[],[],[],[],[],[]

    passage_lengths, questions_lengths = [], []

    for i in data:
    	ids.append(i)

    	text_passages.append(data[i][0])
    	text_questions.append(data[i][1])
    	text_answers.append(data[i][2])
    	spans.append(data[i][3])
    	token_passages.append(data[i][4])
    	token_questions.append(data[i][5])
    	num_passages.append(data[i][6])
    	num_questions.append(data[i][7])
    	passage_lengths.append(len(data[i][4]))
    	questions_lengths.append(len(data[i][5]))

    return ids, text_passages, text_questions, text_answers, spans, token_passages, token_questions, num_passages, num_questions, passage_lengths, questions_lengths

def dev_json_path():

	with open('./model_pickles/dev_file_path.pickle','rb') as h:
        json_dev_file = pickle.load(h)

    return json_dev_file

def shuffle_data(train_spans, train_token_passages, train_token_questions, train_num_passages, train_num_questions, train_p_lengths, train_q_lengths):
    dataset = list(zip(train_spans, train_token_passages, train_token_questions, train_num_passages, train_num_questions, train_p_lengths, train_q_lengths))

    shuffle(dataset)

    train_spans, train_token_passages, train_token_questions, train_num_passages, train_num_questions, train_p_lengths, train_q_lengths = list(zip(*dataset))

    return list(train_spans), list(train_token_passages), list(train_token_questions), list(train_num_passages), list(train_num_questions), list(train_p_lengths), list(train_q_lengths)


def get_one_zero_mat(batch_size, passage_lens, question_lens):
    zero_one = np.ones((batch_size, max(passage_lens), max(question_lens)))

    for i in range(batch_size):
        zero_one[i][0:passage_lens[i].item(), 0:question_lens[i].item()] = np.zeros((passage_lens[i].item(),
                                                                                    question_lens[i].item()))

    zero_one = zero_one * (10 ** 15)
    zero_one = torch.tensor(zero_one, dtype=torch.float32).cuda()

    return zero_one


def get_zero_one_mat(batch_size, passage_lens):
    zero_one = np.ones((batch_size, max(passage_lens).item()))

    for i in range(len(passage_lens)):
        zero_one[i][0:passage_lens[i].item()] = np.zeros((passage_lens[i].item()))

    zero_one = zero_one * (10 ** 15)
    zero_one = torch.tensor(zero_one, dtype=torch.float32).cuda()

    return zero_one


def get_ones_zeros_mat(batch_size, lengths, max_length):
    zero_one = np.ones((batch_size, max_length.item(), max(lengths).item()))

    for i in range(batch_size):
        if max(lengths).item() != lengths[i].item():
            zero_one[i][:, lengths[i].item():] = np.zeros((max_length.item(),
                                                          (max(lengths) - lengths[i]).item()))

    zero_one = torch.tensor(zero_one, dtype=torch.float32).cuda()

    return zero_one



def get_ones_and_zeros_mat(batch_size, sequence_lengths, hidden_dim):
    zero_one = np.ones((batch_size, sequence_lengths[0].item(), 2 * hidden_dim))

    for i in range(batch_size):
        if sequence_lengths[0].item() != sequence_lengths[i].item():
            zero_one[i][sequence_lengths[i].item():, :] = np.zeros((
                (sequence_lengths[0].item() - sequence_lengths[i].item()), 2 * hidden_dim))

    zero_one = torch.tensor(zero_one, dtype=torch.float32).cuda()

    return zero_one

def ids_removed_from_dev():

    with open('./model_pickles/dev_ids_removed.pickle','rb') as h:
        dev_ids_removed = pickle.load(h)

    return dev_ids_removed

def get_batch_distribution(batch_size, length):

    number_of_batches = length // batch_size

    egs_in_batch = [batch_size] * number_of_batches

    if length % batch_size != 0:
        egs_in_batch = egs_in_batch + [length % batch_size]

    return egs_in_batch


if __name__ == "__main__":
    pass
