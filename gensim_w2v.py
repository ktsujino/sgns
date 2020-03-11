import gensim
from gensim import downloader

text8 = downloader.load('text8')
model = gensim.models.word2vec.Word2Vec(text8,
                                        max_vocab_size=100000)
                                        
model.wv.save_word2vec_format("gensim.model")
