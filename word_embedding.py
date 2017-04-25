import gensim
import argparse
import numpy as np
import os
import sys
import string
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument("input", help="input file name")
parser.add_argument("--embedding_dim", help="Dimension of the word embedding.", type=int, default=100, dest="embedding")
parser.add_argument("--train", help="Train the model given the specified INPUT file and write to disk.", action="store_true")
parser.add_argument("--encode", help="Load a trained model from disk and encode the given sentences.", action="store_true")
parser.add_argument("--filename", help="Filename to use for output encodings", default='output')
args = parser.parse_args()

embedding_size = args.embedding

class Sentences(object):
    """Defines a class which is used to iterate over the training or 
    testing corpus. This prevents having to read the entire file into 
    memory."""
    def __init__(self, corpusFile, delimiter=None):
        self.corpusFile = corpusFile
        self.delimiter = delimiter
        self._length = 0
        self._length_not_set = True
        self.max_length = 0
 
    def __iter__(self):
        for line in open(self.corpusFile):
            line = line.split(self.delimiter)
            if(self._length_not_set):
                self._length += 1
                self.max_length = max(self.max_length, len(line))
            yield line
        self._length_not_set = False
    
    def __len__(self):
        if(self._length_not_set):
            return None
        else:
            return self._length

def main():
    """This program reads in the pre-processed text corpus and trains a 
    word2vec model which creates the word embeddings. The program can also 
    write embedded sentences to a numpy array and save them to disk."""
    
    # Train the model and save to disk
    if(args.train):
        trainingData = args.input
        sentences = Sentences(trainingData)
        model = gensim.models.Word2Vec(sentences, min_count=5, workers=4, size=embedding_size, alpha=0.05, cbow_mean=1, negative=20)
        model.save('w2v_model/w2v.mdl')
        print("\tModel successfully trained and saved to directory w2v_model.")
        print("\tEmbedding dimision: {}".format(embedding_size))
    elif(args.encode):
        encodingData = args.input
        model = gensim.models.Word2Vec.load('w2v_model/w2v.mdl')
        sentences = Sentences(encodingData)
        # Need to iterate once over the sentences to accumulate some information
        for s in sentences:
            pass
        padding_size = sentences.max_length
        
        # Special words - control symbol "barcode" + all zeros.
        NCS = 3 # Number of Control Symbols
        missing_word = np.concatenate((np.array([1,0,0], dtype=np.float32), np.zeros(embedding_size, dtype=np.float32)))
        unknown_word = np.concatenate((np.array([0,1,0], dtype=np.float32), np.zeros(embedding_size, dtype=np.float32)))
        padding_word = np.concatenate((np.array([0,0,1], dtype=np.float32), np.zeros(embedding_size, dtype=np.float32)))
        
        # Barcode for vocabulary words
        vocab_word = np.zeros(NCS, dtype=np.float32)
    
        # Embed the text corpus
        fname = '{}.npy'.format(args.filename)
        output = np.memmap(fname, dtype=np.float32, shape=(len(sentences), padding_size, NCS + embedding_size), mode='w+')
        maxIndex = []
        i = 0
        for s in sentences:
            j = 0
            for w in s:
                if(w in model.wv):
                    output[i][j] = np.concatenate((vocab_word, model.wv[w])) # word vector of word j in sentence i
                elif(w == "/MISSING_WORD"):
                    output[i][j] = missing_word
                else:
                    output[i][j] = unknown_word
                j += 1
            maxIndex.append(j-1)
            for k in xrange(padding_size - len(s)):
                output[i][j+k] = padding_word
            if((i+1) % 1000 == 0):
                output.flush()
            i += 1
        del output
        
        print("\tEmbedded {} sentences and saved to {}".format(len(sentences), fname))
        print("\tArray shape: ({}, {}, {})".format(len(sentences), padding_size, NCS+embedding_size))
    else:
        print("No valid options selected!")
    
    return 0

if __name__ == '__main__':
    main()

