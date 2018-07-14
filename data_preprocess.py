import pickle
import json
import numpy
from tqdm import tqdm
import unicodedata
import string
from nltk import word_tokenize
from nltk.tokenize.moses import MosesDetokenizer
import argparse
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import torch

detokenizer = MosesDetokenizer()
all_letters = string.printable
max_length_of_passage = 400


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters)

def preprocess_json_to_get_data(filepath, mode):

    with open(filepath, encoding='utf-8') as q:
        data = json.load(q)

    data_dicts = {}
    all_ids_in_data = []

    for i in tqdm(range(len(data['data']))):
        for j in range(len(data['data'][i]['paragraphs'])):

            passage = unicodeToAscii(data['data'][i]['paragraphs'][j]['context'])
            while '\n' in passage:
                index = passage.index('\n')
                passage = passage[0:index] + passage[index + 1:]

            for k in range(len(data['data'][i]['paragraphs'][j]['qas'])):

                question = unicodeToAscii(data['data'][i]['paragraphs'][j]['qas'][k]['question'])
                id1 = data['data'][i]['paragraphs'][j]['qas'][k]['id']
                all_ids_in_data.append(id1)

                if data['data'][i]['paragraphs'][j]['qas'][k]['is_impossible'] == True:
                    answer_key = 'plausible_answers'
                else:
                    answer_key = 'answers'

                all_starts, all_ends, all_answers = [], [], []
                for l in range(len(data['data'][i]['paragraphs'][j]['qas'][k][answer_key])):
                    answer = unicodeToAscii(data['data'][i]['paragraphs'][j]['qas'][k][answer_key][l]['text'])
                    start = data['data'][i]['paragraphs'][j]['qas'][k][answer_key][l]['answer_start']
                    end = data['data'][i]['paragraphs'][j]['qas'][k][answer_key][l]['answer_start'] + len(answer)

                    while '\n' in answer:
                        index = answer.index('\n')
                        answer = answer[0:index] + answer[index + 1:]

                    all_starts.append(start)
                    all_ends.append(end)
                    all_answers.append(answer)

                if len(set(all_starts)) != 1:
                    for popop in range(len(all_starts)):
                        if all_answers[popop] != '':

                            if mode == 'train':
                                data_dicts['#'*popop + id1] = [passage.lower(), question.lower(), all_answers[popop].lower(), (all_starts[popop], all_ends[popop])]

                            else:
                                data_dicts[id1] = [passage.lower(), question.lower(), all_answers[popop].lower(), (all_starts[popop], all_ends[popop])]
                else:
                    data_dicts[id1] = [passage.lower(), question.lower(), all_answers[0].lower(), (all_starts[0], all_ends[0])]

    return data_dicts, all_ids_in_data

def get_word_index_from_span(tokens_passage,tokens_answers):
    
    len_passage = len(tokens_passage)
    len_answer = len(tokens_answers)

    answer_str = detokenizer.detokenize(tokens_answers, return_str=True)

    for i in range(len_passage- len_answer+1):
        passage_str = detokenizer.detokenize(tokens_passage[i:i+len_answer] , return_str=True)

        if answer_str == passage_str:
            return i, i+len_answer-1
        elif answer_str in passage_str:
            return i, i+len_answer-1

    return "not"

def make_dict(dictionary):

    final_dict = {}

    for i in dictionary:
        token_passage = word_tokenize(dictionary[i][0])
        token_answers = word_tokenize(dictionary[i][2])

        span = get_word_index_from_span(token_passage, token_answers)

        if span != 'not':

            if span[1]>=span[0] and span[1]>=0:
                final_dict[i] = [dictionary[i][0], dictionary[i][1], dictionary[i][2], (span[0], span[1]), token_passage, word_tokenize(dictionary[i][1])]

    return final_dict            

def form_vocab(dictionary1, dictionary2):

    vocabulary = []

    for i in (dictionary1):

        vocabulary.extend(dictionary1[i][4])
        vocabulary.extend(dictionary1[i][5])

        vocabulary = list(set(vocabulary))

    for j in (dictionary2):
        
        vocabulary.extend(dictionary2[j][4])
        vocabulary.extend(dictionary2[j][5])

        vocabulary = list(set(vocabulary))

    return vocabulary


def form_embedding_matrix(filename, words):
    glove_input_file = filename
    word2vec_output_file = 'glove_word2vec_file.txt.word2vec'
    glove2word2vec(glove_input_file, word2vec_output_file)
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    word_to_idx={}
    embeddings = []
    words_not_found = []
    count = 0

    for idx,i in enumerate(words):
        try:
            embeddings.append(model[i.lower()].tolist())
            word_to_idx[i.lower()] = count
            count += 1
        except:
        
            try:
                embeddings.append(model[i.capitalize()].tolist())
                word_to_idx[i.lower()] = count
                count += 1
                
            except:
                words_not_found.append(i.lower())  


    for idx,i in enumerate(word_to_idx):
        word_to_idx[i.lower()] = idx

    embeddings=[]
    for idx,i in enumerate(word_to_idx):
        try:
            embeddings.append(model[i].tolist())
        except:
            embeddings.append(model[i.capitalize()].tolist())

    word_to_idx['#$#$#$'] = len(word_to_idx)

    embeddings.append(numpy.random.ranf(100).tolist())
    embeddings.append(numpy.zeros(100).tolist())

    idx_to_word = {}

    for i in word_to_idx:
        idx_to_word[word_to_idx[i]] = i


    return word_to_idx, idx_to_word, embeddings

def convert_to_num_form(dictionary, word_to_idx):

    max_question_length = 0

    for i in dictionary:

        num_list1, num_list2 = [], []
        for j in dictionary[i][4]:
            try:
                num_list1.append(word_to_idx[j])
            except:
                num_list1.append(word_to_idx['#$#$#$'])
           
        for j in dictionary[i][5]:
                
            try:
                num_list2.append(word_to_idx[j])
            except:
                num_list2.append(word_to_idx['#$#$#$'])

            if max_question_length < len(num_list2):
                max_question_length = len(num_list2)

        dictionary[i].append(num_list1)
        dictionary[i].append(num_list2)

    return dictionary, max_question_length

    ## id : [text_passage, text_question, text_answer, (start_span, end_span), tokenized_passage, tokenized_question, tokenized_num_passage, tokenized_num_question]    

def add_padding_and_remove_big_passages(dictionary, padding_index, max_question_length):

    new_dict = {}
    for i in dictionary:

        if len(dictionary[i][4]) <= max_length_of_passage:
            new_dict[i] = dictionary[i]

            new_dict[i][6].extend([padding_index]*(max_length_of_passage-len(dictionary[i][4])))
            
            new_dict[i][7].extend([padding_index]*(max_question_length-len(dictionary[i][5])))

    return new_dict
    
def dev_ids_not_found(dict_keys, dev_ids):

    found_ids = set(dict_keys)

    all_ids = set(dev_ids)

    return list(all_ids-found_ids)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Question Answering Data Preprocess')
    parser.add_argument('--train_json', type=str, required = True, help='Training JSON File for SQuAD 2.0')
    parser.add_argument('--dev_json', type=str, required = True, help='Development JSON File for SQuAD 2.0 ')
    parser.add_argument('--glove_file', type=str, required = True, help='Glove Embedding File')
    args = parser.parse_args()

    json_dev_file = args.dev_json       ##  './json_data_files/dev-v1.1.json'
    json_train_file = args.train_json   ##  './json_data_files/train-v1.1.json'
    glove_file = args.glove_file        


    train_dict, train_ids = preprocess_json_to_get_data(json_train_file, 'train')
    dev_dict, dev_ids = preprocess_json_to_get_data(json_dev_file, 'dev')

    '''
        train_dict[id] = [text_passage, text_question, text_answer, (start_index, end_index)]  
        train_ids = list of all ids

    '''

    training_dict = make_dict(train_dict)
    development_dict = make_dict(dev_dict)

    full_vocab = form_vocab(training_dict, development_dict)

    word_to_idx, idx_to_word, embeddings = form_embedding_matrix(glove_file, full_vocab)

    training_dict, max_q_length_train = convert_to_num_form(training_dict, word_to_idx)
    development_dict, max_q_length_dev = convert_to_num_form(development_dict, word_to_idx)

    training_dict = add_padding_and_remove_big_passages(training_dict, len(word_to_idx), max_q_length_train)
    development_dict = add_padding_and_remove_big_passages(development_dict, len(word_to_idx),  max_q_length_dev)

    dev_ids_removed = dev_ids_not_found(development_dict.keys(), dev_ids)

    '''
        dicts ->    [text_passage, text_question, text_answer, (start_span, end_span), 
                    tokenized_passage, tokenized_question, tokenized_num_passage_padded, tokenized_num_question_padded]

    '''


    with open('./model_pickles/training_data.pickle','wb') as h:
        pickle.dump(training_dict, h)

    with open('./model_pickles/validation_data.pickle','wb') as h:
        pickle.dump(development_dict, h)

    with open('./model_pickles/full_vocab.pickle','wb') as h:
        pickle.dump(full_vocab, h)

    with open('./model_pickles/dev_ids_removed.pickle','wb') as h:
        pickle.dump(dev_ids_removed, h)

    with open('./model_pickles/word_to_idx.pickle','wb') as h:
        pickle.dump(word_to_idx, h)

    with open('./model_pickles/idx_to_word.pickle','wb') as h:
        pickle.dump(idx_to_word, h)

    with open('./model_pickles/dev_file_path.pickle','wb') as h:
        pickle.dump(json_dev_file, h)    

    torch.save(torch.tensor(embeddings) ,'./model_pickles/word_embeddings.pt')

