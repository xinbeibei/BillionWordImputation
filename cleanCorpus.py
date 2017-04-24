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

parser = argparse.ArgumentParser()

parser.add_argument("input", help="input file name")
parser.add_argument("num", help="minimum number of occurences in full corpus", type=int)
parser.add_argument("-smin", help="specify minimum length of sentence", type=int,default=2)
parser.add_argument("-smax", help="specify maximum length of sentence", type=int,default=0)
parser.add_argument("-split", help="specify percentage of corpus to seperate into test set", type=float,default=0.0)
parser.add_argument("-num","--numbers", help="allow numbers in sentences", action="store_true")
parser.add_argument("-hist", help="save histogram of sentence lengths", action="store_true")
args = parser.parse_args()

startTime = time.time()

vocabulary = {} # this is the full vocabulary of words
N = args.num
if args.hist:
	lengthHist = np.zeros(1000)

	## ----------------------------------------------
	## Read the input files
	## ----------------------------------------------
print("\n\n\tCleaning corpus to only include\n\tsentences in which each word\n\tappears at least "+str(N)+" times\n\n\tLoading corpus data...")
	
textFile = open(args.input)
	
for line in textFile:
	line = line.lower().replace('\n', '')
	line = re.sub(ur"[^\w\d'\s]+",'',line)
	sentence = line.split()
	# we also need the vocabulary
	for word in sentence:
		if word not in vocabulary:
			#print word
			vocabulary[word] = 1
		else:
			vocabulary[word] = vocabulary[word] + 1

textFile.close()
endTime = time.time()
print ("\tLoading corpus data complete\t\t%f seconds\n\n\tCleaning started..." % (endTime - startTime))
startTime = time.time()

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

textFile = open(args.input)
cleanSentences = []
for line in textFile:
	line = line.lower().replace('\n', '')
	line = re.sub(ur"[^\w\d'\s]+",'',line)
	sentence = line.split()
	if args.hist:
		try:
			lengthHist[len(sentence)] = lengthHist[len(sentence)] + 1
		except:
			continue
	
	if len(sentence) < args.smin:
		continue 
	if args.smax > 0:
		if len(sentence) > args.smax:
			continue 

	if (onlyKnownWords(sentence)) and not hasNumbers(sentence):
		cleanSentences.append(line)

textFile.close()
endTime = time.time()
print ("\tCleaning complete\t\t\t%f seconds\n\n\tSaving corpus..." % (endTime - startTime))
startTime = time.time()

# Method to remove a random word from
# a sentence and replace it with the
# missing word symbol: /MISSING_WORD
def removeWord(inputString):
	sentence = inputString.split()
	sentence[random.randint(1,len(sentence)-1)] = "/MISSING_WORD"
	return ' '.join(sentence)

cut = int(len(cleanSentences) * (100 - args.split)/100)
fid = open("clean_training_target.txt", 'w')
fid2 = open("clean_training_source.txt", 'w')
for ii in range(0, cut):
	fid.write(cleanSentences[ii]+"\n")
	fid2.write(removeWord(cleanSentences[ii])+"\n")
fid.close()
fid2.close()
if args.split > 0.0:
	fid = open("clean_test_target.txt", 'w')
	fid2 = open("clean_test_source.txt", 'w')
	for ii in range(cut, len(cleanSentences)):
		fid.write(cleanSentences[ii]+"\n")
		fid2.write(removeWord(cleanSentences[ii])+"\n")
	fid.close()
	fid2.close()

if args.hist:
	fid = open('histogram.dat', 'w')
	for ii in range(0,500):
		fid.write(str(ii)+'\t'+str(lengthHist[ii])+"\n")

endTime = time.time()
print ("\tCorpus saved\t\t\t\t%f seconds\n" % (endTime - startTime))
