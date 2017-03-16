import numpy as np
import random
from functools import reduce


class Dataset:
    def __init__(self, keywords, win_size):
        self.keywords = keywords
        self.win_size = win_size
        self.word_to_id = {}
        self.id_to_word = {}
        for i, k in enumerate(self.keywords):
            self.word_to_id[k] = i
            self.id_to_word[i] = k

    def token2id(self, tokens):
        ids = []
        for token in tokens:
            if token not in self.word_to_id:
                self.id_to_word[len(self.word_to_id)] = token
                self.word_to_id[token] = len(self.word_to_id)
            ids.append(self.word_to_id[token])
        return ids

    def id2token(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.id_to_word[i])
        return tokens

    def make_window(self, window, target, isoPosition=False):
        """
        converts the integer tokenIDs in @window to be of the following
        form:
        - token ids in [0,K-1] are left unchanged
        - token ids >= K are converted to a contiguous range with lower IDs to
        initial elements. The way IDs are assigned here depends on the parameter
        @isoPosition. If @isoPosition=False, numbering of non-keyword tokens
        start at K. Otherwise, a non-keyword token first occuring at position w
        in the window is assigned ID K+w.
        So, for example, if K=3, and the window is [0,2,1,5,7,10,2], the
        transformed window will be [0,2,1,3,4,5,2] if @isoPosition=False, else
        it will be [0,2,1,6,7,8,2].
        converts @target using the same dictionary
        """
        positional_map = {}
        new_window = [t for t in window]
        for i, token in enumerate(new_window):
            if token >= len(self.keywords):
                if token not in positional_map:
                    if isoPosition:
                        windowTokenID = len(self.keywords) + i
                    else:
                        windowTokenID = (len(self.keywords) +
                                         len(positional_map))
                    positional_map[token] = windowTokenID
                new_window[i] = positional_map[token]
        if target < len(self.keywords):
            new_target = target
        else:
            if target not in positional_map:
                new_target = len(self.keywords) + self.win_size
            else:
                new_target = positional_map[target]
        return new_window, new_target

    def next_batch(self, filedata, batch_size=1):
        tokenids = [(name, self.token2id(tokens))
                    for name, tokens in filedata]
        # filter out files with < win_size+1 tokens
        filtered = [tokids for _, tokids in tokenids
                    if len(tokids) >= self.win_size+1]
        weights = [len(toks)-self.win_size for toks in filtered]
        num_wins = sum(weights)

        for i in range(0, num_wins, batch_size):
            # Pick the file by weights
            idxs = [np.random.randint(len(weights)) for _ in range(batch_size)]
            # Beginning Index of the window
            widxs = [np.random.randint(len(filtered[idx])-self.win_size)
                     for idx in idxs]
            Xs = [filtered[idx][widx:widx+self.win_size]
                  for widx, idx in zip(widxs, idxs)]
            ys = [filtered[idx][widx+self.win_size]
                  for widx, idx in zip(widxs, idxs)]
            Xys = [self.make_window(x, y, True) for x, y in zip(Xs, ys)]
            Xys = [(x, y) for x, y in Xys
                   if y != len(self.keywords) + self.win_size]
            Xs = [x for x, y in Xys]
            ys = [y for x, y in Xys]
            if Xs and ys:
                yield np.array(Xs), np.array(ys)
            else:
                continue


if __name__ == '__main__':
    keywords = ['for', 'int', '=', '<', '>', ';', '(', ')', '{', '}']
    test_file = [('./test1.py',
                  ['for', 'i', 'in', 'range', '(', '1000', ')', ':',
                   'print', '(', '"hello"', ',', 'i', ')']),
                 ('./test2.py',
                  ['print', '(', 'hello', 'world', ')'])]
    print("keywords: \t", keywords)
    print("%s : \t %s" % (test_file[0][0], test_file[0][1]))
    print("%s : \t %s" % (test_file[1][0], test_file[1][1]))

    win_size = 3
    dataset = Dataset(keywords, win_size)
    for X, Y in dataset.next_batch(test_file):
        print("X: ", X)
        print("Y: ", Y)
        print("-------")
