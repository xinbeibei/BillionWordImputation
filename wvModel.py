import gensim
import numpy as np

class Model():
    def __init__(self, fileName='w2v_model/w2v.mdl', embedding_size=100, NCS=3):
        self.model = gensim.models.Word2Vec.load(fileName)
        self.missing_word = np.concatenate((np.array([1,0,0], dtype=np.float32),np.zeros(embedding_size, dtype=np.float32)))
        self.unknown_word = np.concatenate((np.array([0,1,0], dtype=np.float32),np.zeros(embedding_size, dtype=np.float32)))
        self.padding_word = np.concatenate((np.array([0,0,1], dtype=np.float32),np.zeros(embedding_size, dtype=np.float32)))
        self.ncs = NCS
    
    def closestWord(self, vector):
        """Finds the closest string represetnation of a word given a 
        word vector. The vector is assummed to have a barcode appended 
        to it."""
        lookup = self.model.similar_by_vector(vector[self.ncs:], topn=1)
        padding_score = self.distance(vector, self.padding_word)
        missing_score = self.distance(vector, self.missing_word)
        unknown_score = self.distance(vector, self.unknown_word)
        lookup_score = 1 - lookup[0][1]
        
        if(padding_score == min(padding_score, missing_score, unknown_score, lookup_score)):
            return '/PAD'
        elif(missing_score == min(padding_score, missing_score, unknown_score, lookup_score)):
            return '/MISSING_WORD'
        elif(unknown_score == min(padding_score, missing_score, unknown_score, lookup_score)):
            return '/UNK'
        else:
            return lookup[0][0]
    
    def wordLookup(self, word):
        """'word' is a string representation of a word. If 'word' exists
        in self.model's vocabulary, then the word vector representation 
        will be returned. Otherwise None is returned."""
        if(word in self.model.wv):
            return self.model.wv[word]
        else:
            return None
    
    def distance(self, v1, v2):
        """Compute the cosine distance between two word vectors."""
        return 1 - np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
