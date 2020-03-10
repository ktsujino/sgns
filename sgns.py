from collections import defaultdict
import numpy as np

from corpus import Corpus
from document_reader import document_reader
from lexicon import Lexicon, LexiconBuilder

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

class SGNS:
    def __init__(self,
                 text_dir,
                 window_size=5,
                 vector_dim=300,
                 lexicon_size=30000,
                 negative_sample_size=10,
                 negative_sample_power=0.75,
                 init_learning_rate=0.1,
                 learning_rate_decay=0.9999,
                 minibatch_size=256,
                 num_minibatches=10000):
        lexicon_builder = LexiconBuilder()
        print('building lexicon...')
        count = 0
        for line in document_reader(text_dir):
            count += 1
            if count % 100000 == 0:
                print('read {} lines'.format(count))
            lexicon_builder.register_corpus(line)
        self.lexicon = lexicon_builder.generate_lexicon(lexicon_size,
                                                        negative_sample_power)
        self.corpus = Corpus()
        print('loading corpus...')
        count = 0
        for line in document_reader(text_dir):
            count += 1
            if count % 100000 == 0:
                print('read {} lines'.format(count))
            line_ids = [self.lexicon.word2id(word) for word in line]
            self.corpus.add_document(line_ids)
        self.window_size = window_size
        self.vector_dim = vector_dim
        self.lexicon_size = lexicon_size
        self.negative_sample_size = negative_sample_size
        self.negative_sample_power = negative_sample_power
        self.init_learning_rate = init_learning_rate
        self.learning_rate = init_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.minibatch_size = minibatch_size
        self.num_minibatches = num_minibatches
        self.minibatch_count=0
        
        self.v = np.random.rand(lexicon_size, vector_dim) - 0.5
        self.u = np.random.rand(lexicon_size, vector_dim) - 0.5

    def train_minibatch(self):
        v_update = defaultdict(lambda: np.zeros(self.vector_dim))
        u_update = defaultdict(lambda: np.zeros(self.vector_dim))
        objective = 0
        for _ in range(self.minibatch_size):
            center_word, target_word = self.corpus.sample_word_pair(self.window_size)
            vp, up = self.v[center_word], self.u[target_word]
            prob = sigmoid(np.dot(vp, up))
            v_update[center_word] += self.learning_rate * (1 - prob) * up
            u_update[target_word] += self.learning_rate * (1 - prob) * vp
            objective += np.log(prob)
            for __ in range(self.negative_sample_size):
                negative_word = self.lexicon.sample_word()
                un = self.u[negative_word]
                prob = sigmoid(np.dot(vp, un))
                v_update[center_word] -= self.learning_rate * prob * un
                u_update[negative_word] -= self.learning_rate * prob * vp
                objective += np.log(1 - prob)
        self.update_vectors(v_update, u_update)
        print("batch: %d lr: %.4f objective: %.04f " % (self.minibatch_count,
                                                        self.learning_rate,
                                                        objective / self.minibatch_size))
        self.minibatch_count += 1
        self.learning_rate *= self.learning_rate_decay

    def train(self):
        for _ in range(self.num_minibatches):
            self.train_minibatch()

    def update_vectors(self, v_update, u_update):
        for i, dv in v_update.items():
            self.v[i] += dv
        for i, du in u_update.items():
            self.u[i] += du

    def save(self, output_file):
        fp = open(output_file, 'w')
        fp.write('{} {}\n'.format(self.lexicon_size, self.vector_dim))
        for i in range(self.lexicon_size):
            vec = 0.5 * (self.v[i] + self.u[i])
            fp.write('{} {}\n'.format(self.lexicon.id2word(i),
                                      ' '.join([str(x) for x in vec])))


if __name__ == '__main__':
    sgns = SGNS(text_dir='./corpus')
    sgns.train()
    sgns.save('output.txt')
