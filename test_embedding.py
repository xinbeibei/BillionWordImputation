import gensim
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def main():
    model = gensim.models.Word2Vec.load('w2v.mdl')
    scores = model.accuracy('questions-words.txt')
if __name__ == '__main__':
    main()

