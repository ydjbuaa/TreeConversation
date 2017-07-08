from __future__ import division
import torch
from utils.vocab import Constants

"""
 Class for managing the internals of the beam search process.


         hyp1-hyp1---hyp1 -hyp1
                 \             /
         hyp2 \-hyp2 /-hyp2hyp2
                               /      \
         hyp3-hyp3---hyp3 -hyp3
         ========================

 Takes care of beams, back pointers, and scores.
"""


class Beam(object):
    def __init__(self, size, cuda=False):

        self.size = size
        self.done = False

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(Constants.PAD)]
        self.nextYs[0][0] = Constants.BOS

        # The attentions (matrix) for each time.
        self.attn = []

    def getCurrentState(self):
        """Get the outputs for the current timestep."""
        return self.nextYs[-1]

    def getCurrentOrigin(self):
        """Get the backpointers for the current timestep."""
        return self.prevKs[-1]

    def advance(self, word_lk, attn_out=None):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        num_words = word_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
        else:
            beam_lk = word_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)
        self.allScores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(best_scores_id - prev_k * num_words)
        if attn_out is not None:
            self.attn.append(attn_out.index_select(0, prev_k))

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == Constants.EOS:
            self.done = True
            self.allScores.append(self.scores)

        return self.done

    def sortBest(self):
        return torch.sort(self.scores, 0, True)

    def getBest(self):
        "Get the score of the best in the beam."
        scores, ids = self.sortBest()
        return scores[1], ids[1]

    def getHyp(self, k):
        """
        Walk back to construct the full hypothesis.

        Parameters.

             * `k` - the position in the beam to construct.

         Returns.

            1. The hypothesis
            2. The attention at each time step.
        """
        hyp, attn = [], []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            if self.attn is not None and len(self.attn) > 0:
                attn.append(self.attn[j][k])
            k = self.prevKs[j][k]
        if len(attn) == 0:
            return hyp[::-1], None
        return hyp[::-1], torch.stack(attn[::-1])
