# Beibei, Jared and Roelof
# Final Project - N-gram Training Algorithm
# 04/06/2017 

import sys
import time
import math
from collections import defaultdict
import string
import numpy as np
import argparse
parser = argparse.ArgumentParser()
startTime = time.time()

parser.add_argument("input", help="input file name")
parser.add_argument("n", help="specify the value of N", type=int)
parser.add_argument("-out", help="directory to save output", default='')
parser.add_argument("-vocab", help="store vocabulary", action="store_true")
args = parser.parse_args()

phrases = {}	# this is the dictionary to store all the n-gram phrases
vocabulary = [] # this is the full vocabulary of words
N = args.n

def getNgram(input_list, n):
	return zip(*[input_list[i:] for i in range(n)])

	## ----------------------------------------------
	## Read the input files
	## ----------------------------------------------

print("\n\n\tN-gram Training Algorithm\n\n\tLoading training data...")
	
if args.vocab:
	# get the vocabulary
	textFile = open(args.input)
	vocabulary = set(textFile.read().split())
	textFile.close()

	fid = open(args.out+'vocabulary.txt', 'w')
	fid.write(str(sorted(vocabulary)))
	endTime = time.time()
	print ("\tVocabulary saved\t\t\t%f seconds\n\n\tCounting n-grams..." % (endTime - startTime))
	startTime = time.time()

textFile = open(args.input)

for line in textFile:
	line = line.replace('\n', '')
	
	sentence = line.split()
	ngrams = getNgram(sentence,N)
	
	for phrase in ngrams:
		if phrase in phrases:
			phrases[phrase] = phrases[phrase]+1
		else:
			phrases[phrase] = 1
textFile.close()

endTime = time.time()
print ("\tCounting complete\t\t\t%f seconds\n\n\tSaving model..." % (endTime - startTime))
startTime = time.time()

total = 1.0*sum(phrases.values())
fid = open(args.out+'ngramModel.txt', 'w')
fid.write("%i Word Phrase" % (N)+" "*20+"Probability\n")
fid.write('-'*91+'\n')
for phrase in phrases:
	for word in phrase:
  		fid.write("%s " % word)
	fid.write("\t%.5e\n"%(phrases[phrase]/total))
endTime = time.time()
print ("\tModel saved\t\t\t%f seconds\n" % (endTime - startTime))
