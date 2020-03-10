import random
import bisect

class Corpus:
    def __init__(self):
        self.documents = []
        self.lengths = []
        self.sum_length = 0

    def add_document(self, document):
        self.documents.append(document)
        self.sum_length += len(document)
        self.lengths.append(self.sum_length)

    def sample_document(self):
        pos = random.randrange(self.sum_length)
        doc_id = bisect.bisect_right(self.lengths, pos)
        return self.documents[doc_id]

    def sample_word_pair(self, window_size):
        while True:
            document = self.sample_document()
            distance = random.randrange(1, window_size + 1)
            if len(document) <= distance:
                continue
            left_pos = random.randrange(len(document) - distance)
            right_pos = left_pos + distance
            return document[left_pos], document[right_pos]
