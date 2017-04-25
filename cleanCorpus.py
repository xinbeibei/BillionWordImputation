# Beibei, Jared and Roelof
# Final Project - Corpus cleaning script
# 04/13/2017 

import sys
import time
import math
from collections import defaultdict
import string
import re
import numpy as np
import argparse
import random

def hasNumbers(inputString):
    if args.numbers:
        return False
    else:
        return any(char.isdigit() for char in inputString)

def onlyKnownWords(inputString):
    for word in inputString:
        if (word not in vocabulary) or (vocabulary[word] < N):
            return False
    return True

# Method to remove a random word from
# a sentence and replace it with the
# missing word symbol: /MISSING_WORD
def removeWord(sentence, FH):
    index = random.randint(1,len(sentence)-2)
    FH.write(sentence[index]+'\n')
    sentence[index] = "/MISSING_WORD"
    return ' '.join(sentence)

parser = argparse.ArgumentParser()

parser.add_argument("input", help="input file name")
parser.add_argument("num", help="minimum number of occurences for a word in the full corpus", type=int)
parser.add_argument("--smin", help="specify minimum length of sentence", type=int, default=3)
parser.add_argument("--smax", help="specify maximum length of sentence", type=int, default=9999)
parser.add_argument("--split", help="specify percentage of corpus to seperate into test set", type=int, default=5)
parser.add_argument("--allow_numbers", help="allow numbers in sentences", action="store_true", dest="numbers")
parser.add_argument("--hist", help="save histogram of sentence lengths", action="store_true")
parser.add_argument("--afr", help="cleaning afrikaans corpus", action="store_true")
args = parser.parse_args()

startTime = time.time()

vocabulary = {} # this is the full vocabulary of words
N = args.num
## ----------------------------------------------
## Read the input files
## ----------------------------------------------
print("\n\n\tCleaning corpus to only include\n\tsentences in which each word\n\tappears at least "+str(N)+" times\n\n\tLoading corpus data...")

maxLen = 0
textFile = open(args.input)
for line in textFile:
    line = line.lower().replace('\n', '')
    if args.afr:
        if re.match("^[^<>]+$",line) is None:
            continue
        line = re.sub(ur"^[a-mo-z]\s",'',line)
    line = re.sub(ur"[^\w\d'\s]+",'',line)
    sentence = line.split()
    # Don't add words to vocab from sentences which won't be included
    if(len(sentence) < args.smin):
        continue
    if(len(sentence) > args.smax):
        continue
    if(hasNumbers(sentence)):
        continue
    # we also need the vocabulary
    for word in sentence:
        if word not in vocabulary:
            #print word
            vocabulary[word] = 1
        else:
            vocabulary[word] = vocabulary[word] + 1
    maxLen = max(maxLen, len(sentence))

textFile.close()
endTime = time.time()
print ("\tLoading corpus data complete\t\t%f seconds\n\n\tCleaning started..." % (endTime - startTime))
startTime = time.time()


textFile = open(args.input)
tr_tar = open("clean_training_target.txt", 'w')
tr_src = open("clean_training_source.txt", 'w')
te_tar = open("clean_testing_target.txt", 'w')
te_src = open("clean_testing_source.txt", 'w')

tr_gt = open("training_ground_truth.txt", "w")
te_gt = open("testing_ground_truth.txt", "w")
count = 0
rejected = 0
if args.hist:
    lengthHist = np.zeros(maxLen+1)
for line in textFile:
    line = line.lower().replace('\n', '')
    if args.afr:
        if re.match("^[^<>]+$",line) is None:
            continue
        line = re.sub(ur"^[a-mo-z]\s",'',line)
    line = re.sub(ur"[^\w\d'\s]+",'',line)
    sentence = line.split()
    
    if len(sentence) < args.smin:
        rejected += 1
        continue 
    if len(sentence) > args.smax:
        rejected += 1
        continue 

    if (onlyKnownWords(sentence)) and not hasNumbers(sentence):
        if( (args.split*count) % 100 == 0):
            te_tar.write(' '.join(sentence)+"\n")
            te_src.write(removeWord(sentence, te_gt)+"\n")
        else:
            tr_tar.write(' '.join(sentence)+"\n")
            tr_src.write(removeWord(sentence, tr_gt)+"\n")
        count += 1
        if args.hist:
            lengthHist[len(sentence)] = lengthHist[len(sentence)] + 1
    else:
        rejected += 1
textFile.close()
tr_gt.close()
te_gt.close()

if args.hist:
    fid = open('histogram.dat', 'w')
    for ii in range(len(lengthHist)):
        fid.write(str(ii)+'\t'+str(lengthHist[ii])+"\n")

endTime = time.time()
print ("\tCorpus saved\t\t\t\t%f seconds\n" % (endTime - startTime))
print ("\tKept %d sentences and rejected %d." %(count, rejected))
