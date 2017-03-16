import os
import re
import time
import argparse
import numpy as np
from normalize import search_files, tokenize
from dataset import Dataset
from model import LSTMModel


def parse_arguments():
    def listparse(arg):
        return arg.split(',')
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['train', 'predict'],
                   help='train/test/predict')
    p.add_argument('-p', '--project', default='django',
                   help='project name, should be a subdir in data')
    p.add_argument('--restore', default='',
                   help='load weights from a saved file.')
    p.add_argument('--win', type=int, default=40,
                   help='window size')
    p.add_argument('--dim', type=int, default=32,
                   help='word vector dimensions')
    p.add_argument('--lr', type=float, default=0.05,
                   help='learning rate')
    p.add_argument('-s', '--suffixes', type=listparse, default=['py'],
                   help='languages to use as a comma separated list')
    p.add_argument('--batch', type=int, default=32,
                   help='batch size')
    p.add_argument('--epoches', type=int, default=32,
                   help='batch size')
    p.add_argument('--files', type=listparse,
                   help='list of files for prediction')
    return p.parse_args()

def train():
    smooth_loss = None
    smooth_known_acc = None
    smooth_abs_acc = None

    startTime = time.time()

    def smooth_update(sx, x):
        if sx is None:
            return x
        else:
            return 0.99*sx + 0.01*x

    for e in range(100):
        b = 0
        for X, Y in dataset.next_batch(filetokens, args.batch):
            b += 1
            loss, acc = model.train(X, Y)
            smooth_loss = smooth_update(smooth_loss, loss)
            smooth_known_acc = smooth_update(smooth_known_acc, acc)
            smooth_abs_acc = smooth_update(smooth_abs_acc,
                                           acc*len(Y)/float(args.batch))
            currentTime = time.time()
            print('[%.2fs], E %d, B: %d, loss %f, acc %.2f%%, abs_acc %.2f%%'
                  % (currentTime-startTime, e, b, smooth_loss,
                     smooth_known_acc*100, smooth_abs_acc*100))
            if b > 1 and b % 100 == 0:
                print("saved!")
                model.save('./save/save.h5')


if __name__ == '__main__':
    args = parse_arguments()
    vocabs = []
    vocabs_file = os.path.join('./keywords', args.project)
    with open(vocabs_file) as fp:
        for line in fp:
            kw = re.sub(' [0-9]*$', '', line.strip())
            vocabs.append(kw)
    data_dir = os.path.join('./data', args.project)
    files = search_files([data_dir], args.suffixes)
    print("found %s files" % len(files))
    filetokens = []
    for i, name in enumerate(files):
        if i % 1000 == 0:
            print("%s files processed" % i)
        filetokens.append((name, tokenize(name)))

    dataset = Dataset(vocabs, args.win)
    model = LSTMModel(vocabs, args.dim, args.win)
    # model.load("./save/save.h5")
    train()
    for X, Y in dataset.next_batch(filetokens, 10):
        print("prediction: ", np.argmax(model.predict(X), axis=1),
              "correct: ", Y)
