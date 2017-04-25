# Beibei, Jared and Roelof
# Final Project - N-gram Evaluation Algorithm
# 04/06/2017 

import sys
import time
import math
from collections import defaultdict
import string
import numpy as np
import random

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("vocab", help="vocabulary file")
parser.add_argument("model", help="model file")
parser.add_argument("ts", help="test sentences file")
args = parser.parse_args()

startTime = time.time()

phrases = {}
vocabulary = []
N = 0
def getNgram(input_list, n):
	return zip(*[input_list[i:] for i in range(n)])

	## ----------------------------------------------
	## Read the input files
	## ----------------------------------------------

#try:
print("\n\n\tN-gram Algorithm\n\n\tLoading vocabulary and \n\tmodel parameters...")

vocabulary = eval(open(args.vocab).read())
textFile = open(args.model)

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
#except:
#	print ('\t'+'*'*56)
#	print ('\tError occurred during reading of training data... \n\tProgram terminated')
#	print ('\t'+'*'*56)
#	exit()

endTime = time.time()
print ("\tLoading model data complete\t%f seconds\n\n\tStarting analysis..." % (endTime - startTime))
startTime = time.time()

textFile = open(args.ts)
fid = open(str(N)+'-gram_answer.txt', 'w')
	
for line in textFile:
	line = line.replace('\n', '')
	sentence = line.split()
	pos = sentence.index('/MISSING_WORD')
	#missingWord = sentence[pos]
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
	fid.write(chosenWord+'\n')
textFile.close()
fid.close()

endTime = time.time()
print ("\tAnalysis complete\t\t%f seconds\n\n" % (endTime - startTime))
