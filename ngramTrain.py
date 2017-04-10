# Beibei, Jared and Roelof
# Final Project - N-gram Training Algorithm
# 04/06/2017 

import sys
import time
import math
from collections import defaultdict
import string
import numpy as np

startTime = time.time()

phrases = {}	# this is the dictionary to store all the n-gram phrases
vocabulary = [] # this is the full vocabulary of words
N = int(sys.argv[2])

def getNgram(input_list, n):
	return zip(*[input_list[i:] for i in range(n)])

	## ----------------------------------------------
	## Read the input files
	## ----------------------------------------------

try:
	print("\n\n\tN-gram Training Algorithm\n\n\tLoading training data...")
	
	textFile = open(sys.argv[1])
	
	sentences = []
	for line in textFile:
		line = line.translate(None, string.punctuation).lower().replace('\n', '')
		sentences.append(line)

	textFile.close()
except:
	print ('\t'+'*'*56)
	print ('\tError occurred during reading of training data... \n\tProgram terminated')
	print ('\t'+'*'*56)
	exit()

endTime = time.time()
print ("\tLoading training data complete\t\t%f seconds\n\n\tLearning started..." % (endTime - startTime))
startTime = time.time()

for sentence in sentences:
	sentence = sentence.split()
	ngrams = getNgram(sentence,N)
	
	for phrase in ngrams:
		if phrase in phrases:
			phrases[phrase] = phrases[phrase]+1
		else:
			phrases[phrase] = 1

	# we also need the vocabulary
	for word in sentence:
		if word not in vocabulary:
			vocabulary.append(word)

endTime = time.time()
print ("\tLearning complete\t\t\t%f seconds\n\n\tSaving model..." % (endTime - startTime))
startTime = time.time()
total = 1.0*sum(phrases.values())
fid = open('ngramModel.txt', 'w')
fid.write("%i Word Phrase" % (N)+" "*20+"Probability\n")
fid.write('-'*91+'\n')
for phrase in phrases:
	for word in phrase:
  		fid.write("%s " % word)
	fid.write("\t%.5e\n"%(phrases[phrase]/total))

endTime = time.time()
print ("\tModel saved\t\t\t\t%f seconds\n\n\tSaving vocabulary..." % (endTime - startTime))
startTime = time.time()
fid = open('vocabulary.txt', 'w')
fid.write(str(sorted(vocabulary)))
endTime = time.time()
print ("\tVocabulary saved\t\t\t%f seconds\n" % (endTime - startTime))
