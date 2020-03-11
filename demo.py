#!/usr/local/bin/python3

import argparse
import gensim
import sys
import traceback

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model-file', default='sgns.model')
args = parser.parse_args()

print('loading model...')
model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(args.model_file)
print('Provide input word(s). One word invokes similarity, and three words invoke analogy:')
for line in sys.stdin:
    try:
        words = line.rstrip('\n').split(' ')
        num_positive = (len(words)+1) // 2
        positive, negative = words[:num_positive], words[num_positive:]
        print(model.most_similar(positive=positive, negative=negative))
    except:
        traceback.print_exc()
