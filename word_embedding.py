import gensim
import os
import sys
import string

# Reusable buffer class for performance increase... unfinished
#class Sequence(object):
    #def __init__(self, maxlen=1000):
        #self.buff = [None]*maxlen
        #self.length = 0
        
    #def __len__(self):
        #return self.length
        
    #def __iter__(self):
        #for i in xrange(self.length):
            #yield self.buff[i]


class Sentences(object):
    """Defines a class which is used to iterate over the training or testing
    corpus. This prevents having to read the entire file into memory."""
    def __init__(self, corpusFile, delimiter=None, remove='', samecase=True):
        self.corpusFile = corpusFile
        self.delimiter = delimiter
        self.remove = remove
        self.samecase = samecase
 
    def __iter__(self):
        if(self.samecase):
            for line in open(self.corpusFile):
                yield line.translate(None, self.remove).lower().split(self.delimiter)
        else:
            for line in open(self.corpusFile):
                yield line.translate(None, self.remove).split(self.delimiter)

def main():
    trainingData = sys.argv[1]
    sentences = Sentences(trainingData, remove=string.punctuation)
    model = gensim.models.Word2Vec(sentences, min_count=10, workers=4, size=100, alpha=0.05, cbow_mean=1, negative=20)
    model.save('w2v.mdl')
    
    return 0

if __name__ == '__main__':
    main()

