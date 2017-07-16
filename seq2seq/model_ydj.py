# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from utils.vocab import Constants


class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, attn_input, context):
        """
        attn_input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(attn_input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        context_combined = torch.cat((weighted_context, attn_input), 1)

        context_output = self.tanh(self.linear_out(context_combined))

        return context_output, attn


class MaskGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(MaskGRU, self).__init__()
        self.gru_cell = nn.GRUCell(input_size, hidden_size, num_layers)

    def forward(self, seq_embs, mask_x, hidden):
        outputs = []
        for t in range(seq_embs.size(0)):
            hidden_t = self.gru_cell(seq_embs[t], hidden)
            # expand mask as step t
            mask_t = mask_x[t].unsqueeze(1).expand_as(hidden_t)
            # apply mask, note that hidden is kept with last hidden and output is set to zero
            hidden = hidden_t * mask_t + hidden * (1.0 - mask_t)
            outputs.append(hidden_t * mask_t)

        outputs = torch.stack(outputs)
        return outputs, hidden


class MaskBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(MaskBiGRU, self).__init__()
        self.forward_gru = nn.GRUCell(input_size, hidden_size, num_layers)
        self.backward_gru = nn.GRUCell(input_size, hidden_size, num_layers)

    def forward(self, seq_input, seq_mask, hidden):
        pass


class EncoderRNN(nn.Module):
    def __init__(self, config, vocab_size):
        super(EncoderRNN, self).__init__()

        self.layers = config['n_layers']
        self.num_directions = 2 if config['bidirectional'] else 1
        assert config['rnn_hidden_size'] % self.num_directions == 0
        self.hidden_size = config['rnn_hidden_size'] // self.num_directions
        input_size = config['word_emb_size']
        self.embedding = nn.Embedding(vocab_size,
                                      config['word_emb_size'],
                                      padding_idx=Constants.PAD)

        self.gru = nn.GRU(input_size, self.hidden_size,
                          num_layers=config['n_layers'],
                          dropout=config['dropout'],
                          bidirectional=config['bidirectional'])

    def forward(self, src_input, hidden=None):
        if isinstance(src_input, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = src_input[1].data.view(-1).tolist()
            emb = pack(self.embedding(src_input[0]), lengths)
        else:
            emb = self.embedding(src_input)
        outputs, hidden_t = self.gru(emb, hidden)
        if isinstance(src_input, tuple):
            outputs = unpack(outputs)[0]
        return hidden_t, outputs


class AttnDecoderRNN(nn.Module):
    def __init__(self, config, vocab_size):
        super(AttnDecoderRNN, self).__init__()
        self.layers = config['n_layers']
        self.input_feed = config['input_feed']
        input_size = config['word_emb_size']
        self.hidden_size = config['rnn_hidden_size']

        if self.input_feed:
            input_size += config['rnn_hidden_size']

        self.embedding = nn.Embedding(vocab_size,
                                      config['word_emb_size'],
                                      padding_idx=Constants.PAD)

        self.gru = nn.GRU(input_size, self.hidden_size,
                          num_layers=config['n_layers'],
                          dropout=config['dropout'],
                          bidirectional=False)

        self.attn = GlobalAttention(config['rnn_hidden_size'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, tgt_input, hidden, context, init_output):
        emb = self.embedding(tgt_input)
        outputs = []
        output = init_output
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)
            # print(emb_t.size(), hidden.size())
            output, hidden = self.gru(emb_t.unsqueeze(0), hidden)
            output = output.squeeze(0)
            output, attn = self.attn(output, context.t())
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden, attn


class Seq2SeqModel(nn.Module):
    def __init__(self, config, src_vocab_size, trg_vocab_size):
        super(Seq2SeqModel, self).__init__()
        self.encoder = EncoderRNN(config, src_vocab_size)
        self.decoder = AttnDecoderRNN(config, trg_vocab_size)
        self.hidden_size = config['rnn_hidden_size']
        self.generator = nn.Linear(self.hidden_size, trg_vocab_size)

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                .transpose(1, 2).contiguous() \
                .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, src_input, tgt_input):
        enc_hidden, context = self.encoder(src_input)
        init_output = self.make_init_decoder_output(context)
        #print(enc_hidden.size())
        enc_hidden = self.fix_enc_hidden(enc_hidden)
        #print(enc_hidden.size(), context.size())
        out, dec_hidden, _attn = self.decoder(tgt_input,
                                              enc_hidden,
                                              context,
                                              init_output)
        #print(out.size())
        return out
