from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import wvModel
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("labels", help="Known missing words")
parser.add_argument("predictions", help="Predicted missing words.")
parser.add_argument("--embedding_dim", type=int, default=100)
args = parser.parse_args()

def main(model, pred_file, target_file):
    cosine_dist_sum = 0
    p = open(pred_file, 'r')
    t = open(target_file, 'r')
    pred = [line.rstrip() for line in p.readlines()]
    target = [line.rstrip() for line in t.readlines()]
    p.close()
    t.close()
    
    if len(pred) != len(target):
        raise TypeError("Length not matched.")
        
    for i in range(len(pred)):
        pred_v = model.wordLookup(pred[i])
        target_v = model.wordLookup(target[i])
        cosine_dist_sum += model.distance(pred_v, target_v)
        
    print ("Average cosine distance is:" + str(round(cosine_dist_sum/len(pred), 4)))

if __name__ == '__main__':
    #load the model
    model = wvModel.Model(embedding_size=args.embedding_dim)
    main(model, args.labels, args.predictions)    

