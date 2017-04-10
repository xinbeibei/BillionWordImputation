# Beibei, Jared and Roelof
# Final Project - N-gram Training Algorithm
# 04/06/2017 

import sys
import time
import math
from collections import defaultdict
import string
import numpy as np
import random

startTime = time.time()

phrases = {}
vocabulary = []
N = 0
def getNgram(input_list, n):
	return zip(*[input_list[i:] for i in range(n)])

	## ----------------------------------------------
	## Read the input files
	## ----------------------------------------------

try:
	print("\n\n\tN-gram Algorithm\n\n\tLoading vocabulary and \n\tmodel parameters...")

	vocabulary = eval(open('vocabulary.txt').read())

	textFile = open("ngramModel.txt")

	ii = 0
	for line in textFile:
		if ii == 0:
			N = int(line.split()[0])
		if ii > 1:
			line = line.split()
			phrase = getNgram(line[0:N],N)[0]
			phrases[phrase] = float(line[-1])
		ii = ii+1

	textFile.close()
except:
	print ('\t'+'*'*56)
	print ('\tError occurred during reading of training data... \n\tProgram terminated')
	print ('\t'+'*'*56)
	exit()

endTime = time.time()
print ("\tLoading model data complete\t%f seconds\n\n\tReading input file..." % (endTime - startTime))
startTime = time.time()

sentences =[]
try:
	textFile = open(sys.argv[1])
	
	for line in textFile:
		line = line.translate(None, string.punctuation).lower().replace('\n', '')
		sentences.append(line)
	textFile.close()
except:
	print ('\t'+'*'*56)
	print ('\tError occurred during reading of input data... \n\tProgram terminated')
	print ('\t'+'*'*56)
	exit()

endTime = time.time()
print ("\tLoading input data complete\t%f seconds\n\n\tStarting analysis..." % (endTime - startTime))
startTime = time.time()

success = 0

for sentence in sentences:
	sentence = sentence.split()
	pos = random.randrange(0, len(sentence))
	missingWord = sentence[pos]
	maxProb = 0.0
	chosenWord = ''
	for word in vocabulary:
		sentence[pos] = word
		prob = 0.0
		for ii in range(0,N):
			try:
				phrase = getNgram(sentence[pos - ii:pos - ii + N],N)[0]
				prob = prob + phrases[phrase]
			except:
				pass
		if prob > maxProb:
			maxProb = prob
			chosenWord = word
	#print chosenWord, missingWord
	if (chosenWord == missingWord):
		success = success + 1

endTime = time.time()
print ("\tAnalysis complete\t\t%f seconds\n\n\tTotal sentences evaluated:\t%i\n\tSuccess rate:\t\t\t%.1f%%" % (endTime - startTime, len(sentences), (100.0*success/len(sentences))))


