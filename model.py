import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
from help_functions import *
from copy import copy
import numpy as np


class Encoder(nn.Module):

    def __init__(self, hidden_dim, embedding_matrix, train_word_embeddings, dropout, number_of_layers):

        '''
                arguments passed:
                            1) hidden_dim : the words and chars would be concatenated and would be projected to a space of size=hidden_dim
                            2) embedding_matrix : number of pretrained words, size of embeddings.
                            3) train_word_embeddings : train the word embeddings or not(Boolean).
                            4) dropout : Fraction of neurons dropped

        '''

        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_embedding_dim = embedding_matrix.shape[1]
        self.no_of_pretrained_words = embedding_matrix.shape[0]
        self.train_word_embeddings = train_word_embeddings  # boolean

        self.word_embeddings = nn.Embedding.from_pretrained(embedding_matrix)

        self.dropout_fraction = dropout
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

        self.layers = number_of_layers

        self.word_lstm = nn.LSTM(self.word_embedding_dim, hidden_dim, num_layers=number_of_layers, dropout=dropout, batch_first=True, bidirectional=True)

        self.batch_norm = nn.BatchNorm1d(2 * hidden_dim)

        self.mlp = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

        self.mlp1 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

    def init_hidden(self, dimension, batch_size):
        return torch.zeros(2 * self.layers, batch_size, dimension).cuda(), torch.zeros(2 * self.layers, batch_size, dimension).cuda()

    def sort_sents(self, sentences, lengths):

        sorted_lengths, indexes = torch.sort(lengths, descending=True)
        sorted_sentences = sentences[indexes]

        return sorted_sentences, indexes, sorted_lengths

    def forward(self, max_text_length, batch_sentences, batch_lengths, batch_size, apply_batch_norm, istraining, isquestion):

        '''

                    batch_sentences : [	[1,3,34,4,23.......],
                                           [24,3,pad,pad,pad..],
                                         [1,2,2,pad,pad.....],
                                         [1,3,2,4,pad.......],]  -> torch tensor

                    batch_lengths : [5,2,3,4]
                    batch_size : number of examples received .

        '''

        word_idx_tensor, indexes, sequence_lengths = self.sort_sents(batch_sentences, batch_lengths)

        word_embed = (self.word_embeddings(word_idx_tensor)).view(batch_size, max_text_length,
                                                                  self.word_embedding_dim)

        if istraining:
            word_embed = self.dropout(word_embed)

        packed_embedds = torch.nn.utils.rnn.pack_padded_sequence(word_embed, sequence_lengths, batch_first=True)

        hidden_layer = self.init_hidden(self.hidden_dim, batch_size)

        text_representation, hidden_layer = self.word_lstm(packed_embedds, hidden_layer)

        h, lengths = torch.nn.utils.rnn.pad_packed_sequence(text_representation, batch_first=True)

        text_representation = torch.zeros_like(h).scatter_(0, indexes.unsqueeze(1).unsqueeze(1).expand(-1, h.shape[1], h.shape[2]), h)

        if istraining and apply_batch_norm:
            text_representation = (self.batch_norm(text_representation.permute(0, 2, 1).contiguous())).permute(0, 2, 1)

        text_representation = self.mlp(text_representation)
        if istraining:
            text_representation = self.dropout(text_representation)

        if isquestion:
            text_representation = self.tanh(self.mlp1(text_representation))
            zero_one = get_ones_and_zeros_mat(batch_size, sequence_lengths, self.hidden_dim)
            text_representation = text_representation * zero_one

        return text_representation


class Coattention_Encoder(nn.Module):

    def __init__(self, hidden_dim, dropout, number_of_layers):

        '''

            arguments passed:
                 1) hidden_dim : the words and chars would be concatenated and
                 would be projected to a space of size=hidden_dim

        '''

        super(Coattention_Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.layers = number_of_layers

        self.lstm_layer1 = nn.LSTM(2 * hidden_dim, hidden_dim, num_layers=number_of_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.lstm_layer2 = nn.LSTM(12 * hidden_dim, hidden_dim, num_layers=number_of_layers, dropout=dropout, bidirectional=True, batch_first=True)

        self.mlp1 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.mlp2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

        self.batch_norm = nn.BatchNorm1d(2 * hidden_dim)

    def init_hidden(self, dimension, batch_size):
        return torch.zeros(2 * self.layers, batch_size, dimension).cuda(), torch.zeros(2 * self.layers, batch_size, dimension).cuda()

    def sort_sents(self, S, lengths):

        sorted_lengths, indexes = torch.sort(lengths, descending=True)
        sorted_sentences = S[indexes]

        return sorted_sentences, sorted_lengths, indexes

    def forward(self, question, question_lens, passage, passage_lens, batch_size, apply_batch_norm, istraining):

        question = question.permute(0, 2, 1)
        L = torch.bmm(passage, question)

        AQ = copy(L)
        AD = copy(L.permute(0, 2, 1))

        subtract_from_AQ_AD = get_one_zero_mat(batch_size, passage_lens, question_lens)
        multiply_to_AD = get_ones_zeros_mat(batch_size, passage_lens, max(question_lens))
        multiply_to_AQ = get_ones_zeros_mat(batch_size, question_lens, max(passage_lens))

        AQ = AQ - subtract_from_AQ_AD
        AQ = F.softmax(AQ, dim=1)
        AQ = AQ * multiply_to_AQ

        AD = AD - (subtract_from_AQ_AD.permute(0, 2, 1))
        AD = F.softmax(AD, dim=1)
        AD = AD * multiply_to_AD

        S1D = torch.bmm(question, AD)
        S1Q = torch.bmm(passage.permute(0, 2, 1), AQ)

        C1D = torch.bmm(S1Q, AD)

        S1D = S1D.permute(0, 2, 1)
        S1Q = S1Q.permute(0, 2, 1)

        S1D1, S1D_len, S1D_index = self.sort_sents(S1D, passage_lens)
        S1Q1, S1Q_len, S1Q_index = self.sort_sents(S1Q, question_lens)

        # ------#

        pack_S1D = torch.nn.utils.rnn.pack_padded_sequence(S1D1, S1D_len, batch_first=True)
        hidden_layer = self.init_hidden(self.hidden_dim, batch_size)

        S1D1, hidden_layer = self.lstm_layer1(pack_S1D, hidden_layer)
        S1D1, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(S1D1, batch_first=True)

        # ------#

        pack_S1Q = torch.nn.utils.rnn.pack_padded_sequence(S1Q1, S1Q_len, batch_first=True)
        hidden_layer = self.init_hidden(self.hidden_dim, batch_size)

        S1Q1, hidden_layer = self.lstm_layer1(pack_S1Q, hidden_layer)
        S1Q1, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(S1Q1, batch_first=True)

        # ------#

        E2D = torch.zeros_like(S1D1).scatter_(0, S1D_index.unsqueeze(1).unsqueeze(1).expand(-1, S1D1.shape[1], S1D1.shape[2]), S1D1)
        E2Q = torch.zeros_like(S1Q1).scatter_(0, S1Q_index.unsqueeze(1).unsqueeze(1).expand(-1, S1Q1.shape[1], S1Q1.shape[2]), S1Q1)

        if istraining and apply_batch_norm:
            E2D = self.batch_norm(E2D.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
            E2Q = self.batch_norm(E2Q.permute(0, 2, 1).contiguous()).permute(0, 2, 1)

        E2D = self.mlp1(E2D)
        E2Q = self.mlp1(E2Q)

        # ------------------------------------------------------------#

        if istraining:
            E2D = self.dropout(E2D)
            E2Q = self.dropout(E2Q)

        E2Q = E2Q.permute(0, 2, 1)
        L = torch.bmm(E2D, E2Q)

        AQ = copy(L)
        AD = copy(L.permute(0, 2, 1))

        subtract_from_AQ_AD = get_one_zero_mat(batch_size, passage_lens, question_lens)
        multiply_to_AD = get_ones_zeros_mat(batch_size, passage_lens, max(question_lens))
        multiply_to_AQ = get_ones_zeros_mat(batch_size, question_lens, max(passage_lens))

        AQ = AQ - subtract_from_AQ_AD
        AQ = F.softmax(AQ, dim=1)
        AQ = AQ * multiply_to_AQ

        AD = AD - (subtract_from_AQ_AD.permute(0, 2, 1))
        AD = F.softmax(AD, dim=1)
        AD = AD * multiply_to_AD

        S2D = torch.bmm(E2Q, AD)
        S2Q = torch.bmm(E2D.permute(0, 2, 1), AQ)

        C2D = torch.bmm(S2Q, AD)
        S2D = S2D.permute(0, 2, 1)

        C1D = C1D.permute(0, 2, 1)
        C2D = C2D.permute(0, 2, 1)

        # ---------------------------------------------------------#

        '''
            S1D -> batch*passage_lens*2hidden
            S2D -> batch*passage_lens*2hidden

            E1D(passage) ->  batch_size*passage_length*(2hidden)
            E2D -> batch*passage_lens*2hidden

            C1D -> batch*passage_lens*2hidden
            C2D -> batch*passage_lens*2hidden 

        '''

        U = torch.cat((passage, E2D, S1D, S2D, C1D, C2D), dim=2)

        U, U_length, U_index = self.sort_sents(U, passage_lens)

        # ----#
        pack_U = torch.nn.utils.rnn.pack_padded_sequence(U, U_length, batch_first=True)
        hidden_layer = self.init_hidden(self.hidden_dim, batch_size)

        pack_U, hidden_layer = self.lstm_layer2(pack_U, hidden_layer)
        pack_U, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(pack_U, batch_first=True)

        # ----#

        U = torch.zeros_like(pack_U).scatter_(0, U_index.unsqueeze(1).unsqueeze(1).expand(-1, pack_U.shape[1], pack_U.shape[2]), pack_U)

        if istraining and apply_batch_norm:
            U = self.batch_norm(U.permute(0, 2, 1).contiguous()).permute(0, 2, 1)

        U = self.mlp2(U)

        if istraining:
            U = self.dropout(U)

        return U.permute(0, 2, 1)


class Maxout(nn.Module):

    def __init__(self, dim_in, dim_out, pooling_size):
        super().__init__()

        self.d_in, self.d_out, self.pool_size = dim_in, dim_out, pooling_size
        self.lin = nn.Linear(dim_in, dim_out * pooling_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m


class Decoder(nn.Module):

    def __init__(self, hidden_dim, pooling_size, number_of_iters, dropout):

        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layer = nn.LSTMCell(4 * hidden_dim, hidden_dim)
        self.pooling_size = pooling_size
        self.number_of_iters = number_of_iters

        self.tanh = nn.Tanh()
        self.mlp = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        self.WD_start = nn.Linear(5 * hidden_dim, hidden_dim, bias=False)

        self.maxout1_start = Maxout(3 * hidden_dim, hidden_dim, pooling_size)
        self.maxout2_start = Maxout(hidden_dim, hidden_dim, pooling_size)
        self.maxout3_start = Maxout(2 * hidden_dim, 1, pooling_size)

        self.WD_end = nn.Linear(5 * hidden_dim, hidden_dim, bias=False)

        self.maxout1_end = Maxout(3 * hidden_dim, hidden_dim, pooling_size)
        self.maxout2_end = Maxout(hidden_dim, hidden_dim, pooling_size)
        self.maxout3_end = Maxout(2 * hidden_dim, 1, pooling_size)

    def init_hidden(self, dimension, batch_size):
        return torch.zeros(batch_size, dimension).cuda(), torch.zeros(batch_size, dimension).cuda()

    def forward(self, encoding_matrix, batch_size, passage_lens, apply_batch_norm, istraining):

        '''
                    1) encoding_matrix: batch_size*(2*hidden_dim)*max_passage
                    2) batch_size: batch_size
                    3) passage_lens : list having all the proper lengths of the passages(unpaded)

        '''

        start, end = torch.tensor([0] * batch_size).cuda(), torch.tensor([0] * batch_size).cuda()
        hx, cx = self.init_hidden(self.hidden_dim, batch_size)

        zero_one = get_zero_one_mat(batch_size, passage_lens)

        s1 = start.view(-1, 1, 1).expand(encoding_matrix.size(0), encoding_matrix.size(1), 1)
        e1 = end.view(-1, 1, 1).expand(encoding_matrix.size(0), encoding_matrix.size(1), 1)

        encoding_start_state = encoding_matrix.gather(2, s1).view(batch_size, -1)
        encoding_end_state = encoding_matrix.gather(2, e1).view(batch_size, -1)

        entropies = []

        for i in range(self.number_of_iters):
            alphas, betas = torch.tensor([], requires_grad=True).cuda(), torch.tensor([], requires_grad=True).cuda()

            h_s_e = torch.cat((hx, encoding_start_state, encoding_end_state), dim=1)
            r = self.tanh(self.WD_start(h_s_e))

            for j in range(encoding_matrix.shape[2]):
                jth_states = encoding_matrix[:, :, j]

                e_r = torch.cat((jth_states, r), dim=1)

                m1 = self.maxout1_start.forward(e_r)

                m2 = self.maxout2_start.forward(m1)

                hmn = self.maxout3_start.forward(torch.cat((m1, m2), dim=1))

                alphas = torch.cat((alphas, hmn), dim=1)

            alphas = alphas - zero_one
            start = torch.argmax(alphas, dim=1)

            s1 = start.view(-1, 1, 1).expand(encoding_matrix.size(0), encoding_matrix.size(1), 1)
            encoding_start_state = encoding_matrix.gather(2, s1).view(batch_size, -1)

            h_s_e = torch.cat((hx, encoding_start_state, encoding_end_state), dim=1)

            r = self.tanh(self.WD_end(h_s_e))
            for j in range(encoding_matrix.shape[2]):
                jth_states = encoding_matrix[:, :, j]

                e_r = torch.cat((jth_states, r), dim=1)

                m1 = self.maxout1_end.forward(e_r)

                m2 = self.maxout2_end.forward(m1)

                hmn = self.maxout3_end.forward(torch.cat((m1, m2), dim=1))

                betas = torch.cat((betas, hmn), dim=1)

            betas = betas - zero_one
            end = torch.argmax(betas, dim=1)

            e1 = end.view(-1, 1, 1).expand(encoding_matrix.size(0), encoding_matrix.size(1), 1)
            encoding_end_state = encoding_matrix.gather(2, e1).view(batch_size, -1)

            hx, cx = self.lstm_layer(torch.cat((encoding_start_state, encoding_end_state), dim=1), (hx, cx))

            if istraining and apply_batch_norm:
                hx = self.batch_norm(hx)

            hx = self.mlp(hx)

            if istraining:
                hx = self.dropout(hx)

            entropies.append([alphas, betas])

        return start, end, entropies


class Model(nn.Module):

    def __init__(self, hidden_dim, embedding_matrix, train_word_embeddings, dropout, pooling_size, number_of_iters, number_of_layers):
        super(Model, self).__init__()

        self.Encoder = Encoder(hidden_dim, embedding_matrix, train_word_embeddings, dropout, number_of_layers)
        self.Coattention_Encoder = Coattention_Encoder(hidden_dim, dropout, number_of_layers)
        self.Decoder = Decoder(hidden_dim, pooling_size, number_of_iters, dropout)

    def forward(self, max_p_length, max_q_length, batch_passages, batch_questions, batch_p_lengths, batch_q_lengths, number_of_examples, apply_batch_norm, istraining):
        passage_representation = self.Encoder.forward(max_p_length, batch_passages, batch_p_lengths, number_of_examples, apply_batch_norm, istraining, isquestion=False)

        question_representation = self.Encoder.forward(max_q_length, batch_questions, batch_q_lengths, number_of_examples, apply_batch_norm, istraining, isquestion=True)

        '''
            In passage_length_index and question_length_index-> corresponding elements denote same passage question pair

        '''

        u_matrix = self.Coattention_Encoder.forward(question_representation,
                                                    batch_q_lengths.clone(),
                                                    passage_representation,
                                                    batch_p_lengths.clone(),
                                                    number_of_examples,
                                                    apply_batch_norm,
                                                    istraining)

        # size is batch*(2*hidden)*passage_lens

        start_outputs, end_outputs, entropies = self.Decoder.forward(u_matrix,
                                                                     number_of_examples,
                                                                     batch_p_lengths.clone(),
                                                                     apply_batch_norm,
                                                                     istraining)

        return start_outputs, end_outputs, entropies