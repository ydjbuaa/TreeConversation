# tree object from stanfordnlp/treelstm
from utils.vocab import ConstantTransition


class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth


def read_tree(tree_words):
    parents = list(map(int, tree_words))
    trees = dict()
    root = None
    for i in range(1, len(parents) + 1):
        # if not trees[i-1] and parents[i-1]!=-1:
        if i - 1 not in trees.keys() and parents[i - 1] != -1:
            idx = i
            prev = None
            while True:
                parent = parents[idx - 1]
                if parent == -1:
                    break
                tree = Tree()
                if prev is not None:
                    tree.add_child(prev)
                trees[idx - 1] = tree
                tree.idx = idx - 1

                # if trees[parent-1] is not None:
                if parent - 1 in trees.keys():
                    trees[parent - 1].add_child(tree)
                    break

                elif parent == 0:
                    root = tree
                    break

                else:
                    prev = tree
                    idx = parent
    return root


def convert2sr_format(tree_root, sent):
    def _convert_rec(tree):
        if tree.num_children == 0:
            return [str(sent[tree.idx - 1])]
        else:
            return [ConstantTransition.SHIFT_TRANS] + _convert_rec(tree.children[0]) + _convert_rec(
                tree.children[1]) + [ConstantTransition.REDUCE_TRANS]

    return _convert_rec(tree_root)


def tree2transition(tree, sentence):
    sr_words = convert2sr_format(tree, sentence)
    transition = []
    for s in sr_words:
        if s == ConstantTransition.REDUCE_TRANS:
            transition += [ConstantTransition.REDUCE]  # reduce
            continue
        elif s == ConstantTransition.SHIFT_TRANS:
            continue
        else:
            transition += [ConstantTransition.SHIFT]  # shift
    assert (len(sentence) * 2 - 1) == len(transition)
    return transition
