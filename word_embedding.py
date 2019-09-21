import nltk
nltk.download("movie_reviews")
from nltk.corpus import movie_reviews
sentences = [list(s) for s in movie_reviews.sents()]
print(sentences[0])

from gensim.models import Word2Vec
model = Word2Vec(sentences)
model.similarity('actor','actress')
model.most_similar('accident')
model.most_similar(positive=['she','actor'], negative='actress', topn=1)
model.most_similar()


