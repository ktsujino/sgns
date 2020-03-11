#!/usr/local/bin/python3

import argparse
from collections import defaultdict
import numpy as np

from corpus import Corpus
from document_reader import document_reader
from lexicon import Lexicon, LexiconBuilder

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

class SGNS:
    def __init__(self, args):
        lexicon_builder = LexiconBuilder()
        print('building lexicon...')
        count = 0
        for line in document_reader(args.corpus_dir):
            count += 1
            if count % 100000 == 0:
                print('read {} lines'.format(count))
            lexicon_builder.register_corpus(line)
        self.lexicon = lexicon_builder.generate_lexicon(args.lexicon_size,
                                                        args.negative_sample_power)
        self.corpus = Corpus()
        print('loading corpus...')
        count = 0
        for line in document_reader(args.corpus_dir):
            count += 1
            if count % 100000 == 0:
                print('read {} lines'.format(count))
            line_ids = [self.lexicon.word2id(word) for word in line]
            if not args.train_oov_token:
                line_ids = [i for i in line_ids if i != Lexicon.OOV_WORD_ID]
            self.corpus.add_document(line_ids)
        self.args = args
        self.learning_rate = args.init_learning_rate
        self.minibatch_count = 0
        
        self.v = np.random.rand(args.lexicon_size, args.vector_dim) - 0.5
        self.u = np.random.rand(args.lexicon_size, args.vector_dim) - 0.5

    def train_minibatch(self):
        v_update = defaultdict(lambda: np.zeros(self.args.vector_dim))
        u_update = defaultdict(lambda: np.zeros(self.args.vector_dim))
        objective = 0
        for _ in range(self.args.minibatch_size):
            center_word, target_word = self.corpus.sample_word_pair(self.args.window_size)
            vp, up = self.v[center_word], self.u[target_word]
            prob = sigmoid(np.dot(vp, up))
            v_update[center_word] += self.learning_rate * (1 - prob) * up
            u_update[target_word] += self.learning_rate * (1 - prob) * vp
            objective += np.log(prob)
            for __ in range(self.args.negative_sample_size):
                negative_word = self.lexicon.sample_word()
                un = self.u[negative_word]
                prob = sigmoid(np.dot(vp, un))
                v_update[center_word] -= self.learning_rate * prob * un
                u_update[negative_word] -= self.learning_rate * prob * vp
                objective += np.log(1 - prob)
        self.update_vectors(v_update, u_update)
        print("batch: %d lr: %.4f objective: %.04f " % (self.minibatch_count,
                                                        self.learning_rate,
                                                        objective / self.args.minibatch_size))
        self.minibatch_count += 1
        self.learning_rate *= self.args.learning_rate_decay

    def update_vectors(self, v_update, u_update):
        for i, dv in v_update.items():
            self.v[i] += dv
        for i, du in u_update.items():
            self.u[i] += du

    def train(self):
        for _ in range(self.args.num_minibatches):
            self.train_minibatch()

    def save(self, output_file):
        fp = open(output_file, 'w')
        fp.write('{} {}\n'.format(self.args.lexicon_size, self.args.vector_dim))
        for i in range(self.args.lexicon_size):
            vec = 0.5 * (self.v[i] + self.u[i])
            fp.write('{} {}\n'.format(self.lexicon.id2word(i),
                                      ' '.join([str(x) for x in vec])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--vector-dim', type=int, default=100)
    parser.add_argument('--lexicon-size', type=int, default=50000)
    parser.add_argument('--negative-sample-size', type=int, default=5)
    parser.add_argument('--negative-sample-power', type=float, default=0.75,
                        help='discount factor to negative sample high freq words less frequently')
    parser.add_argument('--init-learning-rate', type=int, default=0.025)
    parser.add_argument('--learning-rate-decay', type=int, default=0.9999)
    parser.add_argument('--minibatch-size', type=int, default=128)
    parser.add_argument('--num-minibatches', type=int, default=10000)
    parser.add_argument('--train-oov-token', action='store_true', default=False)
    parser.add_argument('--corpus-dir', default=None,
                        help='if skipped download and use text8')
    parser.add_argument('--output-model', default='sgns.model')
                        
    args = parser.parse_args()
    sgns = SGNS(args)
    sgns.train()
    sgns.save(args.output_model)
