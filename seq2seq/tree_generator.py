from seq2seq.tree_models import *
from utils.beam import Beam
from utils.dataset import TreeDataset


class TreeConversationGenerator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self.beam_accum = None

        checkpoint = torch.load(opt.model)

        model_config = checkpoint['config']
        self.src_vocab = checkpoint['vocabs']['src']
        self.tgt_vocab = checkpoint['vocabs']['tgt']

        encoder = EncoderTreeLSTM(model_config, self.src_vocab.size)
        decoder = DecoderStackLSTM(model_config, self.tgt_vocab.size)
        generator = nn.Linear(model_config['rnn_hidden_size'], self.tgt_vocab.size)

        model = TreeSeq2SeqModel(encoder, decoder)

        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])

        if opt.cuda:
            model.cuda()
            generator.cuda()
        else:
            model.cpu()
            generator.cpu()

        model.generator = generator
        print(model)
        self.model = model
        self.model.eval()

    def init_beam_accum(self):
        self.beam_accum = {
            "predicted_ids": [],
            "beam_parent_ids": [],
            "scores": [],
            "log_probs": []}

    def build_data(self, src_batch, trans_batch, gold_batch):
        # This needs to be the same as preprocess_tree.py.

        src_data = [self.src_vocab.convert2idx(b, Constants.UNK_WORD)
                    for b in src_batch]

        tgt_data = None
        if gold_batch:
            tgt_data = [self.tgt_vocab.convert2idx(b,
                                                   Constants.UNK_WORD,
                                                   Constants.BOS_WORD,
                                                   Constants.EOS_WORD) for b in gold_batch]

        return TreeDataset(src_data, trans_batch, tgt_data, self.opt.batch_size,
                           self.opt.cuda, volatile=True)

    def build_target_tokens(self, pred, src, attn):
        tokens = self.tgt_vocab.convert2words(pred, Constants.EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == Constants.UNK_WORD:
                    _, max_index = attn[i].max(0)
                    tokens[i] = src[max_index[0]]
        return tokens

    def generate_conversation_batch(self, src_batch, tgt_batch):
        # Batch size is in different location depending on data.

        beam_size = self.opt.beam_size
        batch_size = src_batch[0].size(1)

        #  (1) run the encoder on the src
        context, enc_states = self.model.encoder(src_batch)

        enc_states = (self.model.fix_enc_hidden(enc_states[0]),
                      self.model.fix_enc_hidden(enc_states[1]))

        # (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        gold_scores = enc_states[0].data.new(batch_size).zero_()
        if tgt_batch is not None:
            dec_states = enc_states
            dec_out = self.model.make_init_decoder_output(enc_states[0])
            # init_output = self.model.make_init_decoder_output(context)

            dec_out, dec_states, attn = self.decoder(tgt_batch[:-1], dec_states, context,
                                                     src_batch[1], dec_out)

            for dec_t, tgt_t in zip(dec_out, tgt_batch[1:].data):
                gen_t = self.model.generator.forward(dec_t)
                tgt_t = tgt_t.unsqueeze(1)
                scores = gen_t.data.gather(1, tgt_t)
                scores.masked_fill_(tgt_t.eq(Constants.PAD), 0)
                gold_scores += scores

        # (3) run the decoder to generate sentences, using beam search

        # Expand tensors for each beam.
        context = context * beam_size  # Variable(context.data.repeat(1, beam_size, 1))
        dec_states = (Variable(enc_states[0].data.repeat(1, beam_size, 1)),
                      Variable(enc_states[1].data.repeat(1, beam_size, 1)))

        beam = [Beam(beam_size, self.opt.cuda) for k in range(batch_size)]

        dec_out = self.model.make_init_decoder_output(dec_states[0])

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size
        rnn_size = enc_states[0].size(2)

        for i in range(self.opt.max_sent_length):
            # Prepare decoder input.
            tgt_input = torch.stack([b.getCurrentState() for b in beam
                                     if not b.done]).t().contiguous().view(1, -1)

            # print(tgt_input.size(), dec_out.size())

            dec_out, dec_states, attn = self.model.decoder(
                Variable(tgt_input, volatile=True), dec_states, context, src_batch[1], dec_out)
            # decOut: 1 x (beam*batch) x numWords
            dec_out = dec_out.squeeze(0)
            out = self.model.generator(dec_out)
            out = F.log_softmax(out)

            # batch x beam x numWords
            word_lk = out.view(beam_size, remaining_sents, -1) \
                .transpose(0, 1).contiguous()

            # attn = attn.view(beam_size, remaining_sents, -1) \
            #     .transpose(0, 1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                # if not beam[b].advance(word_lk.data[idx], attn.data[idx]):
                #    active += [b]

                if not beam[b].advance(word_lk.data[idx], None):
                    active += [b]

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    sent_states = dec_state.view(-1, beam_size,
                                                 remaining_sents,
                                                 dec_state.size(2))[:, :, idx]
                    sent_states.data.copy_(
                        sent_states.data.index_select(
                            1, beam[b].getCurrentOrigin()))

                #iterate decoder output
                sent_dec_out = dec_out.view(beam_size, remaining_sents, dec_out.size(1))[:, idx]
                sent_dec_out.data.copy_(sent_dec_out.data.index_select(0, beam[b].getCurrentOrigin()))

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = self.tt.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remaining_sents, rnn_size)
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
                return Variable(view.index_select(1, active_idx)
                                .view(*new_size), volatile=True)

            dec_states = (update_active(dec_states[0]),
                          update_active(dec_states[1]))

            dec_out = update_active(dec_out)

            # update context note that context is the type of list
            #context = update_active(context)
            context = [context[i] for i in active_idx] * beam_size

            remaining_sents = len(active)

        # (4) package everything up
        all_hyp, all_scores, all_attn = [], [], []
        n_best = self.opt.n_best

        for b in range(batch_size):
            scores, ks = beam[b].sortBest()

            all_scores += [scores[:n_best]]
            hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            all_hyp += [hyps]
            all_attn += [attn]

            if self.beam_accum:
                self.beam_accum["beam_parent_ids"].append(
                    [t.tolist()
                     for t in beam[b].prevKs])
                self.beam_accum["scores"].append([
                                                     ["%4f" % s for s in t.tolist()]
                                                     for t in beam[b].allScores][1:])
                self.beam_accum["predicted_ids"].append(
                    [[self.tgt_vocab.get_word(idx)
                      for idx in t.tolist()]
                     for t in beam[b].nextYs][1:])

        return all_hyp, all_scores, all_attn, gold_scores

    def generate_conversation(self, src_batch, trans_batch, gold_batch):
        #  (1) convert words to indexes
        dataset = self.build_data(src_batch, trans_batch, gold_batch)
        src, tgt, indices = dataset[0]
        batch_size = src[0].size(1)

        #  (2) translate
        pred, pred_score, attn, gold_score = self.generate_conversation_batch(src, tgt)
        pred, pred_score, attn, gold_score = list(zip(
            *sorted(zip(pred, pred_score, attn, gold_score, indices),
                    key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        pred_batch = []
        for b in range(batch_size):
            pred_batch.append(
                [self.build_target_tokens(pred[b][n], src_batch[b], attn[b][n])
                 for n in range(self.opt.n_best)]
            )

        return pred_batch, pred_score, gold_score
