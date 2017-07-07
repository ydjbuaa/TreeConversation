# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.vocab import Constants, ConstantTransition


def state_bundle(h, c):
    return torch.cat([h, c], 1)


def state_unbundle(hc, hidden_size):
    return torch.split(hc, hidden_size, 1)


def batch_bundle(batch_iter):
    return torch.cat(batch_iter, 0)


def batch_unbundle(batch_tensor):
    return torch.split(batch_tensor, 1)


class BinaryTreeLeafModule(nn.Module):
    """
  local input = nn.Identity()()
  local c = nn.Linear(self.in_dim, self.mem_dim)(input)
  local h
  if self.gate_output then
    local o = nn.Sigmoid()(nn.Linear(self.in_dim, self.mem_dim)(input))
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end

  local leaf_module = nn.gModule({input}, {c, h})
    """

    def __init__(self, in_dim, mem_dim):
        super(BinaryTreeLeafModule, self).__init__()

        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.cx = nn.Linear(self.in_dim, self.mem_dim)
        self.ox = nn.Linear(self.in_dim, self.mem_dim)

    def forward(self, seq_inputs):
        # print(seq_inputs.size())
        leaf_states = []
        for seq_input in seq_inputs:
            c = self.cx(seq_input)
            o = F.sigmoid(self.ox(seq_input))
            h = o * F.tanh(c)
            leaf_states += [state_bundle(h, c)]
        leaf_states = torch.stack(leaf_states)
        return leaf_states


class BinaryTreeComposer(nn.Module):
    """
  local lc, lh = nn.Identity()(), nn.Identity()()
  local rc, rh = nn.Identity()(), nn.Identity()()
  local new_gate = function()
    return nn.CAddTable(){
      nn.Linear(self.mem_dim, self.mem_dim)(lh),
      nn.Linear(self.mem_dim, self.mem_dim)(rh)
    }
  end

  local i = nn.Sigmoid()(new_gate())    -- input gate
  local lf = nn.Sigmoid()(new_gate())   -- left forget gate
  local rf = nn.Sigmoid()(new_gate())   -- right forget gate
  local update = nn.Tanh()(new_gate())  -- memory cell update vector
  local c = nn.CAddTable(){             -- memory cell
      nn.CMulTable(){i, update},
      nn.CMulTable(){lf, lc},
      nn.CMulTable(){rf, rc}
    }

  local h
  if self.gate_output then
    local o = nn.Sigmoid()(new_gate()) -- output gate
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end
  local composer = nn.gModule(
    {lc, lh, rc, rh},
    {c, h})
    """

    def __init__(self, in_dim, mem_dim):
        super(BinaryTreeComposer, self).__init__()

        self.in_dim = in_dim
        self.mem_dim = mem_dim

        def new_gate():
            lh = nn.Linear(self.mem_dim, self.mem_dim)
            rh = nn.Linear(self.mem_dim, self.mem_dim)
            return lh, rh

        self.ilh, self.irh = new_gate()
        self.lflh, self.lfrh = new_gate()
        self.rflh, self.rfrh = new_gate()
        self.ulh, self.urh = new_gate()

    def forward(self, left_states, right_states):
        left_states = batch_bundle(left_states)
        right_states = batch_bundle(right_states)
        lh, lc = state_unbundle(left_states, self.mem_dim)
        rh, rc = state_unbundle(right_states, self.mem_dim)

        i = F.sigmoid(self.ilh(lh) + self.irh(rh))
        lf = F.sigmoid(self.lflh(lh) + self.lfrh(rh))
        rf = F.sigmoid(self.rflh(lh) + self.rfrh(rh))
        u = F.tanh(self.ulh(lh) + self.urh(rh))
        c = i * u + lf * lc + rf * rc
        h = F.tanh(c)
        return batch_unbundle(state_bundle(h, c))


class SpinnTreeLSTM(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(SpinnTreeLSTM, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.leaf_module = BinaryTreeLeafModule(emb_size, hidden_size)
        self.reduce_composer = BinaryTreeComposer(hidden_size, hidden_size)

    def generate_hidden_states(self, stacks):
        h_states = []
        c_states = []
        for stack in stacks:
            assert len(stack) == 1
            h, c = state_unbundle(stack.pop(), self.hidden_size)
            h_states += [h]
            c_states += [c]

        h_states = batch_bundle(h_states)
        c_states = batch_bundle(c_states)

        # print(h_states.size(), c_states.size())
        return h_states, c_states

    def forward(self, seq_embs, transitions):
        """
        :param seq_embs: embs sequences(seqL * batch_size * dim)
        :param transitions: transitions(transL * batch_size)
        :return:
        """
        leaf_states = self.leaf_module(seq_embs)
        # transpose to batch_size * seqL * (hidden_dim * 2)
        leaf_states = leaf_states.transpose(0, 1)
        # print(leaf_states.size())
        buffers = []
        for i in range(leaf_states.size(0)):
            buffer = list(batch_unbundle(leaf_states[i]))
            buffers += [buffer]

        num_transitions = transitions.size(0)

        stacks = [[] for _ in range(transitions.size(1))]
        stack_outputs = [[] for _ in range(transitions.size(1))]

        for i in range(num_transitions):
            trans = transitions[i]
            lefts, rights = [], []
            batch = zip(trans.data, buffers, stacks, stack_outputs)
            for transition, buf, stack, stack_output in batch:
                if transition == ConstantTransition.SHIFT:  # shift
                    stack.append(buf.pop())
                    # print(stack[-1].size())
                    stack_output.append(stack[-1].squeeze(0))

                elif transition == ConstantTransition.REDUCE: # reduce
                    # kep the tmp result
                    rights.append(stack.pop())
                    lefts.append(stack.pop())

            if rights:
                reduced = iter(self.reduce_composer(lefts, rights))
                for transition, stack, stack_output in zip(trans.data, stacks, stack_outputs):
                    if transition == 2:
                        stack.append(next(reduced))
                        # print(stack[-1].size())
                        stack_output.append(stack[-1].squeeze(0))

        # print(len(stacks))
        hidden_states = self.generate_hidden_states(stacks)
        return hidden_states


class EncoderTreeLSTM(nn.Module):
    def __init__(self, config, vocab_size):
        super(EncoderTreeLSTM, self).__init__()
        self.hidden_size = config['rnn_hidden_size']
        self.emb_size = config['word_emb_size']
        self.num_directions = 1
        self.embedding = nn.Embedding(vocab_size,
                                      config['word_emb_size'],
                                      padding_idx=Constants.PAD)
        self.tree_lstm = SpinnTreeLSTM(self.emb_size, self.hidden_size)

    def load_pretrained_vectors(self, config):
        if config['pre_word_embs_enc'] is not None:
            pretrained = torch.load(config['pre_word_embs_enc'])
            self.embedding.weight.data.copy_(pretrained)

    def forward(self, src_inputs):
        # src_inputs:(sent_inputs, trees)
        embs = self.embedding(src_inputs[0])
        hidden_states = self.tree_lstm(embs, src_inputs[1])
        return hidden_states


class TreeGuidedAttention(nn.Module):
    def __init__(self, dim):
        super(TreeGuidedAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        pass

    def get_child_attn(self, tree):
        la = tree.children[0].attn_score
        ra = tree.children[1].attn_score

        lh = tree.children[0].attn_state
        rh = tree.children[1].attn_state

        child_attn = torch.stack([la, ra])
        child_attn_weights = self.sm(child_attn)
        return child_attn_weights[0] * lh + child_attn_weights[1] * rh

    def attn_forward(self, target, tree):
        tree.attn_score = torch.dot(target, tree.state[1])
        if tree.num_children == 0:
            tree.attn_state = torch.dot(target, tree.state[1])
        else:
            for idx in range(tree.num_children):
                _, = self.attn_forward(target, tree.children[idx])
            child_attn = self.get_child_attn(tree)
            tree.attn_state = tree.attn_score * (child_attn + tree.state[1])
        return tree.attn_state

    def forward(self, hidden_inputs, trees):
        """
        consider the tree states as the whole context of decoder hidden state
        :param hidden_inputs: hidden state of decoder
        :param trees: encoder tree states
        :return:
        """
        targets = self.linear_in(hidden_inputs).unsqueeze(1)  # batch x 1* dim
        attn_outputs = []
        for i in range(hidden_inputs.size(0)):
            attn_output = self.attn_forward(targets[i], trees[i])
            attn_outputs += [attn_output.squeeze(0)]
        attn_outputs = torch.stack(attn_outputs)  # batch * dim
        attn_outputs = self.tanh(self.linear_out(attn_outputs))
        return attn_outputs


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, net_input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(net_input, (h_0[i], c_0[i]))
            net_input = h_1_i
            if i + 1 != self.num_layers:
                net_input = self.dropout(net_input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return net_input, (h_1, c_1)


class DecoderStackLSTM(nn.Module):
    def __init__(self, config, vocab_size):
        self.layers = config['n_layers']
        self.input_feed = config['input_feed']
        input_size = config['word_emb_size']

        if self.input_feed:
            input_size += config['rnn_hidden_size']

        super(DecoderStackLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      config['word_emb_size'],
                                      padding_idx=Constants.PAD)
        self.lstm = StackedLSTM(self.layers, input_size,
                                config['rnn_hidden_size'],
                                config['dropout'])

        # self.attn = GlobalAttention(config['rnn_hidden_size'])
        self.dropout = nn.Dropout(config['dropout'])

        self.hidden_size = config['rnn_hidden_size']

    def load_pretrained_vectors(self, config):
        if config['pre_word_embs_dec'] is not None:
            pretrained = torch.load(config['pre_word_embs_dec'])
            self.embedding.weight.data.copy_(pretrained)

    def forward(self, tgt_input, hidden, context, init_output):
        embs = self.embedding(tgt_input)
        outputs = []
        output = init_output
        for emb_t in torch.split(embs, 1):
            emb_t = emb_t.squeeze(0)
            # print(emb_t.size(), output.size())
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)
            output, hidden = self.lstm(emb_t, hidden)
            output = self.dropout(output)
            outputs += [output]
        outputs = torch.stack(outputs)
        return outputs, hidden, None


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def make_init_decoder_output(self, hidden_state):
        batch_size = hidden_state.size(0)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(hidden_state.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                .transpose(1, 2).contiguous() \
                .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h.view(-1, h.size(0), h.size(1))

    def forward(self, src_input, tgt_input):
        # src = net_input[0]
        # tgt = net_input[1][:-1]  # exclude last target from inputs
        enc_hidden = self.encoder(src_input)
        init_output = self.make_init_decoder_output(enc_hidden[0])

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))

        out, dec_hidden, _attn = self.decoder(tgt_input, enc_hidden,
                                              src_input[1], init_output)
        return out
