import glob
import os
import MeCab

from document_reader import document_reader

if __name__ == '__main__':
    mecab = MeCab.Tagger()
    IN_DIR = '../sese/wikipedia/text'
    OUT_FILE = './corpus.txt'
    ofp = open(OUT_FILE, 'w')
    for line in document_reader(IN_DIR, split=False):
        words = []
        node = mecab.parseToNode(line)
        while node:
            word = node.surface
            if word:
                words.append(word)
            node = node.next
        ofp.write('{}\n'.format(' '.join(words)))
                
