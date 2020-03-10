import gensim
from gensim.models import Word2Vec


model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format('output.txt')
print(model.wv.most_similar(positive=['重要']))
