import torch
import math
from torch.autograd import Variable
from utils.vocab import Constants, ConstantTransition


class Dataset(object):
    def __init__(self, src_data, tgt_data, batch_size, cuda, volatile=False, ret_limit=None):
        self.src = src_data

        if tgt_data:
            self.tgt = tgt_data
            assert (len(self.src) == len(self.tgt))
            if ret_limit:
                self.src = self.src[:ret_limit]
                self.tgt = self.tgt[:ret_limit]
        else:
            self.tgt = None
            if ret_limit:
                self.src = self.src[:ret_limit]

        self.cuda = cuda

        self.batch_size = batch_size
        self.num_samples = len(self.src)
        self.num_batches = math.ceil(len(self.src) / batch_size)
        self.volatile = volatile

    @staticmethod
    def _batch_identity(data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.num_batches, "%d > %d" % (index, self.num_batches)
        src_batch, lengths = self._batch_identity(
            self.src[index * self.batch_size:(index + 1) * self.batch_size],
            align_right=False, include_lengths=True)

        if self.tgt:
            tgt_batch, _ = self._batch_identity(
                self.tgt[index * self.batch_size:(index + 1) * self.batch_size],
                align_right=False, include_lengths=True)
        else:
            tgt_batch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(src_batch))
        batch = (zip(indices, src_batch) if tgt_batch is None
                 else zip(indices, src_batch, tgt_batch))

        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgt_batch is None:
            indices, src_batch = zip(*batch)
        else:
            indices, src_batch, tgt_batch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0)
            b = b.t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)
        lengths = Variable(lengths, volatile=self.volatile)
        return (wrap(src_batch), lengths), \
               wrap(tgt_batch), indices

    def __len__(self):
        return self.num_batches

    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])


class TreeDataset(object):
    def __init__(self, src_data, trans_data, tgt_data, batch_size, cuda, volatile=False, ret_limit=None):
        self.src = src_data
        self.trans = trans_data
        assert len(self.src) == len(self.trans)

        if tgt_data:
            self.tgt = tgt_data
            assert (len(self.src) == len(self.tgt))
            if ret_limit:
                self.src = self.src[:ret_limit]
                self.trans = self.trans[:ret_limit]
                self.tgt = self.tgt[:ret_limit]
        else:
            self.tgt = None
            if ret_limit:
                self.src = self.src[:ret_limit]
                self.trans = self.trans[:ret_limit]

        self.cuda_flag = cuda

        self.batch_size = batch_size
        self.num_samples = len(self.src)
        self.num_batches = math.ceil(len(self.src) / batch_size)
        self.volatile = volatile

    @staticmethod
    def _trans_batch_identity(trans):
        trans = [torch.LongTensor(t) for t in trans]
        lengths = [x.size(0) for x in trans]
        max_length = max(lengths)
        out = trans[0].new(len(trans), max_length).fill_(ConstantTransition.PAD)
        for i in range(len(trans)):
            data_length = trans[i].size(0)
            out[i].narrow(0, 0, data_length).copy_(trans[i])
        return out

    @staticmethod
    def _batch_identity(data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.num_batches, "%d > %d" % (index, self.num_batches)
        src_batch, lengths = self._batch_identity(
            self.src[index * self.batch_size:(index + 1) * self.batch_size],
            align_right=True, include_lengths=True)

        trans_batch = self._trans_batch_identity(
            self.trans[index * self.batch_size:(index + 1) * self.batch_size])

        if self.tgt:
            tgt_batch = self._batch_identity(
                self.tgt[index * self.batch_size:(index + 1) * self.batch_size],
                align_right=False, include_lengths=False)
        else:
            tgt_batch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(src_batch))
        batch = (zip(indices, src_batch, trans_batch) if tgt_batch is None
                 else zip(indices, src_batch, trans_batch, tgt_batch))

        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgt_batch is None:
            indices, src_batch, trans_batch = zip(*batch)
        else:
            indices, src_batch, trans_batch, tgt_batch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0)
            b = b.t().contiguous()
            if self.cuda_flag:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        return (wrap(src_batch), wrap(trans_batch)), wrap(tgt_batch), indices

    def __len__(self):
        return self.num_batches

    def shuffle(self):
        data = list(zip(self.src, self.trans, self.tgt))
        self.src, self.trans, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])
