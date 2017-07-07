import torch


class Constants:
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3
    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'


class ConstantTransition:
    PAD = 0
    SHIFT = 1
    REDUCE = 2

    PAD_TRANS = '<pad>'
    SHIFT_TRANS = '<shift>'
    REDUCE_TRANS = '<reduce>'


class Vocabulary(object):
    def __init__(self, data=None, lower=False):
        self.idx2word = {}
        self.word2idx = {}
        self.frequencies = {}
        self.lower = lower

        # Special entries will not be pruned.
        self.special = []

        if data is not None:
            if type(data) == str:
                self.load_file(data)
            else:
                self.add_specials(data)

    @property
    def size(self):
        return len(self.idx2word)

    def load_file(self, filename):
        """Load entries from a file."""
        for line in open(filename):
            fields = line.split()
            word = fields[0]
            idx = int(fields[1])
            self.add(word, idx)

    def write_file(self, filename):
        """Write entries to a file."""
        with open(filename, 'w', encoding='utf-8') as file:
            for i in range(self.size):
                word = self.idx2word[i]
                file.write('%s %d\n' % (word, i))

        file.close()

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.word2idx[key]
        except KeyError:
            return default

    def get_word(self, idx, default=None):
        try:
            return self.idx2word[idx]
        except KeyError:
            return default

    def add_special(self, word, idx=None):
        """Mark this `word` and `idx` as special (i.e. will not be pruned)."""
        idx = self.add(word, idx)
        self.special += [idx]

    def add_specials(self, words):
        """Mark all words in `words` as specials (i.e. will not be pruned)."""
        for word in words:
            self.add_special(word)

    def add(self, word, idx=None):
        """Add `word` in the dictionary. Use `idx` as its index if given."""
        word = word.lower() if self.lower else word
        if idx is not None:
            self.idx2word[idx] = word
            self.word2idx[word] = idx
        else:
            if word in self.word2idx:
                idx = self.word2idx[word]
            else:
                idx = len(self.idx2word)
                self.idx2word[idx] = word
                self.word2idx[word] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    def prune(self, size):
        """Return a new dictionary with the `size` most frequent entries."""
        if size >= self.size:
            return self

        # Only keep the `size` most frequent entries.
        freq = torch.IntTensor(
            [self.frequencies[i] for i in range(len(self.frequencies))])

        _, idx = torch.sort(freq, 0, True)

        new_vocab = Vocabulary()
        new_vocab.lower = self.lower

        # Add special entries in all cases.
        for i in self.special:
            new_vocab.add_special(self.idx2word[i])

        for i in idx[:size]:
            new_vocab.add(self.idx2word[i])

        return new_vocab

    def convert2idx(self, words, unk_word, bos_word=None, eos_word=None):
        """
        Convert `words` to indices. Use `unkWord` if not found.
        Optionally insert `bosWord` at the beginning and `eosWord` at the .
        """
        vec = []

        if bos_word is not None:
            vec += [self.lookup(bos_word)]

        unk = self.lookup(unk_word)
        vec += [self.lookup(word, default=unk) for word in words]

        if eos_word is not None:
            vec += [self.lookup(eos_word)]

        return torch.LongTensor(vec)

    def convert2words(self, idx, stop):
        """
        Convert `idx` to words.
        If index `stop` is reached, convert it and return.
        """
        words = []
        for i in idx:
            words += [self.get_word(i)]
            if i == stop:
                break
        return words
