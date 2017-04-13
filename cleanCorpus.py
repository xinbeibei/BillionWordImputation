# Beibei, Jared and Roelof
# Final Project - Corpus cleaning script
# 04/13/2017 

import sys
import time
import math
from collections import defaultdict
import string
import numpy as np

startTime = time.time()

vocabulary = {} # this is the full vocabulary of words
N = int(sys.argv[2])

	## ----------------------------------------------
	## Read the input files
	## ----------------------------------------------

try:
	print("\n\n\tCleaning corpus to only include\n\tsentences in which each word\n\tappears at least "+str(N)+" times\n\n\tLoading corpus data...")
	
	textFile = open(sys.argv[1])
	
	sentences = []
	for line in textFile:
		line = line.replace('\n', '')
		sentences.append(line)

	textFile.close()
except:
	print ('\t'+'*'*56)
	print ('\tError occurred during reading of training data... \n\tProgram terminated')
	print ('\t'+'*'*56)
	exit()

endTime = time.time()
print ("\tLoading corpus data complete\t\t%f seconds\n\n\tCleaning started..." % (endTime - startTime))
startTime = time.time()

for sentence in sentences:
	sentence = sentence.split()
	# we also need the vocabulary
	for word in sentence:
		if word not in vocabulary:
			#print word
			vocabulary[word] = 1
		else:
			vocabulary[word] = vocabulary[word] + 1

def hasNumbers(inputString):
	return any(char.isdigit() for char in inputString)

def onlyKnownWords(inputString):
	for word in inputString:
		if (word not in vocabulary) or (vocabulary[word] < N):
			return False
	return True

for sentence in list(sentences):
	line = sentence.split()
	if (not onlyKnownWords(line)) or hasNumbers(line):
		sentences.remove(sentence)

endTime = time.time()
print ("\tCleaning complete\t\t\t%f seconds\n\n\tSaving corpus..." % (endTime - startTime))
startTime = time.time()
fid = open(sys.argv[3], 'w')
for sentence in sentences:
	fid.write(sentence+"\n")

endTime = time.time()
print ("\tCorpus saved\t\t\t\t%f seconds\n" % (endTime - startTime))

