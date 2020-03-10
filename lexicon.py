import bisect
import random

from collections import defaultdict


class Lexicon:
    OOV_TOKEN='<OOV>'
    OOV_WORD_ID=0

    def __init__(self, words, counts, sample_discount_power):
        self._word2id = defaultdict(lambda: Lexicon.OOV_WORD_ID)
        self._words = words
        self._counts = counts
        self._cum_probs = []
        for i, word in enumerate(words):
            self._word2id[word] = i
        probs = [count ** sample_discount_power for count in counts]
        sum_probs = sum(probs)
        probs = [prob / sum_probs for prob in probs]
        cum_prob = 0.0
        for prob in probs:
            cum_prob += prob
            self._cum_probs.append(cum_prob)

    def word2id(self, word):
        return self._word2id[word]

    def id2word(self, i):
        if i < 0 or i >= len(self._words):
            return OOV_TOKEN
        else:
            return self._words[i]

    def id2count(self, i):
        if i < 0 or i >= len(self._counts):
            return 0
        else:
            return self._counts[i]

    def word2count(self, word):
        return self.id2count(self.word2id(word))

    def sample_word(self):
        rand = random.random()
        return bisect.bisect_right(self._cum_probs, rand)

class LexiconBuilder:
    def __init__(self):
        self.word2count = defaultdict(int)

    def register_corpus(self, words):
        for word in words:
            self.word2count[word] += 1

    def generate_lexicon(self, lexicon_size, sample_discount_power):
        word_count = [(word, count) for word, count in self.word2count.items()]
        word_count.sort(key=lambda entry: -entry[1])
        oov_count = sum([entry[1] for entry in word_count[(lexicon_size-1):]])
        words = [Lexicon.OOV_TOKEN] + [entry[0] for entry in word_count[:(lexicon_size-1)]]
        counts = [oov_count] + [entry[1] for entry in word_count[:(lexicon_size-1)]]
        return Lexicon(words, counts, sample_discount_power)
