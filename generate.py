from __future__ import division

import argparse
import math

import torch

from seq2seq.generator import ConversationGenerator

parser = argparse.ArgumentParser(description='generate.py')
parser.add_argument('-model', default="./data/stc.small/checkpoints/model_acc_19.73_ppl_272.28_e10.pt",
                    help='Path to model .pt file')
parser.add_argument('-src', default="./data/stc.small/test/src.test.txt",
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-src_img_dir', default="",
                    help='Source image directory')
parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-output', default='./data/stc.small/test/stc.pred.e10.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size', type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=50,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=30,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', default=0,
                    help="""Replace the generated UNK tokens with the source
                    token that had highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")

parser.add_argument('-verbose', default=0,
                    help='Print scores and predictions for each sentence')
parser.add_argument('-dump_beam', type=str, default="",
                    help='File to dump beam information to.')

parser.add_argument('-n_best', type=int, default=5,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")

parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")


def report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total / words_total)))


def add_one(f):
    for line in f:
        yield line
    yield None


def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    generator = ConversationGenerator(opt)

    out_fw = open(opt.output, 'w', encoding='utf-8')

    pred_score_total, pred_words_total, gold_score_total, gold_words_total = 0, 0, 0, 0

    src_batch, tgt_batch = [], []

    count = 0

    tgt_fr = open(opt.tgt, 'r', encoding='utf-8') if opt.tgt else None

    if opt.dump_beam != "":
        import json
        generator.init_beam_accum()

    for line in add_one(open(opt.src, 'r', encoding='utf-8')):
        if line is not None:
            src_tokens = line.split()
            src_batch += [src_tokens]
            if tgt_fr:
                tgt_tokens = tgt_fr.readline().split() if tgt_fr else None
                tgt_batch += [tgt_tokens]

            if len(src_batch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(src_batch) == 0:
                break

        pred_batch, pred_score, gold_score = generator.generate_conversation(src_batch, tgt_batch)
        pred_score_total += sum(score[0] for score in pred_score)
        pred_words_total += sum(len(x[0]) for x in pred_batch)
        if tgt_fr is not None:
            gold_score_total += sum(gold_score)
            gold_words_total += sum(len(x) for x in tgt_batch)

        for b in range(len(pred_batch)):
            count += 1
            for n in range(opt.n_best):
                out_fw.write("SENT %d: %s\t[%.4f]\t%s\n" % (
                    count,
                    ' '.join(src_batch[b]),
                    pred_score[b][n],
                    (" ".join(pred_batch[b][n]).replace("<unk>", ""))))

            # outF.write(" ".join(predBatch[b][0]) + '\n')
            out_fw.flush()
            if count % 10 == 0:
                print('generate {} lines over!'.format(count))

            if opt.verbose:
                src_sent = ' '.join(src_batch[b])
                if generator.tgt_vocab.lower:
                    src_sent = src_sent.lower()
                print('SENT %d: %s' % (count, src_sent))
                print('PRED %d: %s' % (count, " ".join(pred_batch[b][0])))
                print("PRED SCORE: %.4f" % pred_score[b][0])

                if tgt_fr is not None:
                    tgt_sent = ' '.join(tgt_batch[b])
                    if generator.tgt_vocab.lower:
                        tgt_sent = tgt_sent.lower()
                    print('GOLD %d: %s ' % (count, tgt_sent))
                    print("GOLD SCORE: %.4f" % gold_score[b])

                if opt.n_best > 1:
                    print('\nBEST HYP:')
                    for n in range(opt.n_best):
                        print("[%.4f] %s" % (pred_score[b][n],
                                             " ".join(pred_batch[b][n])))
                print('')

        src_batch, tgt_batch = [], []

    report_score('PRED', pred_score_total, pred_words_total)
    if tgt_fr:
        report_score('GOLD', gold_score_total, gold_words_total)

    if tgt_fr:
        tgt_fr.close()

    if opt.dump_beam:
        json.dump(generator.beam_accum, open(opt.dump_beam, 'w'))


if __name__ == "__main__":
    main()
