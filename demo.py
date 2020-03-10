#!/usr/local/bin/python3

import gensim
import sys
import traceback

print('loading model...')
model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format('sgns.model')
print('Provide input word:')
for line in sys.stdin:
    try:
        print(model.most_similar(positive=[line.rstrip('\n')]))
    except:
        traceback.print_exc()
