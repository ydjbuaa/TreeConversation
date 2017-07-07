import argparse

from utils.vocab import *
from seq2seq_tree.tree import *

parser = argparse.ArgumentParser(description='preprocess_tree.py')

# **Preprocess Options**

parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-train_src', default="./data/stc.small/train/src.train.txt",
                    help="Path to the training source data")

parser.add_argument('-train_tgt', default="./data/stc.small/train/trg.train.txt",
                    help="Path to the training target data")

parser.add_argument('-valid_src', default="./data/stc.small/train/src.train.txt",
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', default="./data/stc.small/train/trg.train.txt",
                    help="Path to the validation target data")

parser.add_argument('-save_data', default="./data/stc.small/",
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=20000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=10000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")

parser.add_argument('-src_seq_length', type=int, default=30,
                    help="Maximum source sequence length")
parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                    help="Truncate source sequence length.")
parser.add_argument('-tgt_seq_length', type=int, default=30,
                    help="Maximum target sequence length to keep.")
parser.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                    help="Truncate target sequence length.")

parser.add_argument('-shuffle', type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed', type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)


def make_vocabulary(filename, size):
    vocab = Vocabulary([Constants.PAD_WORD, Constants.UNK_WORD,
                        Constants.BOS_WORD, Constants.EOS_WORD],
                       lower=opt.lower)

    with open(filename, 'r', encoding='utf-8') as fr:
        for sent in fr:
            for word in sent.split():
                vocab.add(word)

    original_size = vocab.size
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size, original_size))
    return vocab


def init_vocabulary(name, data_file, vocab_file, vocab_size):
    vocab = None
    if vocab_file is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocab_file + '\'...')
        vocab = Vocabulary()
        vocab.load_file(vocab_file)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        vocab = make_vocabulary(data_file, vocab_size)

    print()
    return vocab


def save_vocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.write_file(file)


def make_tree_data(src_file, tgt_file, src_vocab, tgt_vocab):
    src, tgt, transitions = [], [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (src_file, tgt_file))
    src_fr = open(src_file, 'r', encoding='utf-8')
    tree_fr = open(src_file.replace("txt", "cparents"), 'r', encoding='utf-8')

    tgt_fr = open(tgt_file, 'r', encoding='utf-8')

    while True:
        src_line = src_fr.readline()
        tree_line = tree_fr.readline()
        tgt_line = tgt_fr.readline()

        # normal end of file
        if src_line == "" and tgt_line == "":
            break

        # source or target does not have same number of lines
        if src_line == "" or tgt_line == "":
            print('WARNING: src and tgt do not have the same # of sentences')
            break

        src_line = src_line.strip()
        tgt_line = tgt_line.strip()
        tree_line = tree_line.strip()

        # source and/or target are empty
        if src_line == "" or tgt_line == "":
            print('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        src_words = src_line.split()
        tgt_words = tgt_line.split()
        tree_words = tree_line.split()

        # if len(src_words)!=len(tree_words):
        #     print("WARNING: wrong format of src data:{}=>{}".format(src_line, tree_line))
        #     continue

        if len(src_words) <= opt.src_seq_length \
                and len(tgt_words) <= opt.tgt_seq_length:

            tree = read_tree(tree_words)
            transitions += [tree2transition(tree, src_words)]
            # reverse source sentence
            src_words.reverse()
            src += [src_vocab.convert2idx(src_words, Constants.UNK_WORD)]
            tgt += [tgt_vocab.convert2idx(tgt_words,
                                          Constants.UNK_WORD,
                                          Constants.BOS_WORD,
                                          Constants.EOS_WORD)]
            sizes += [len(src_words)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    src_fr.close()
    tgt_fr.close()
    tree_fr.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        transitions = [transitions[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.IntTensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]
    transitions = [transitions[idx] for idx in perm]

    print(('Prepared %d sentences ' +
           '(%d ignored due to length == 0 or src len > %d or tgt len > %d)') %
          (len(src), ignored, opt.src_seq_length, opt.tgt_seq_length))

    return src, tgt, transitions


def main():
    vocabs = {}
    vocabs['src'] = init_vocabulary('source', opt.train_src, opt.src_vocab,
                                    opt.src_vocab_size)

    vocabs['tgt'] = init_vocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                    opt.tgt_vocab_size)

    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'], train['trans'] = make_tree_data(opt.train_src, opt.train_tgt,
                                                               vocabs['src'], vocabs['tgt'])

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'], valid['trans'] = make_tree_data(opt.valid_src, opt.valid_tgt,
                                                               vocabs['src'], vocabs['tgt'])

    if opt.src_vocab is None:
        save_vocabulary('source', vocabs['src'], opt.save_data + '/vocab/src.tree.dict')
    if opt.tgt_vocab is None:
        save_vocabulary('target', vocabs['tgt'], opt.save_data + '/vocab/tgt.tree.dict')

    print('Saving vocabs to \'' + opt.save_data + '/vocab/vocab.pt\'...')
    torch.save(vocabs, opt.save_data + '/vocab/vocab.tree.pt')

    print('Saving train data to \'' + opt.save_data + '/train/train.pt\'...')
    torch.save(train, opt.save_data + '/train/train.tree.pt')

    print('Saving valid data to \'' + opt.save_data + '/valid/valid.pt\'...')
    torch.save(valid, opt.save_data + '/valid/valid.tree.pt')


if __name__ == "__main__":
    main()
