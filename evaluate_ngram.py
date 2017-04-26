from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import wvModel

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
        
    print ("Average cosine distance is:" + str(cosine_dist_sum/len(pred)))

if __name__ == '__main__':
    pred_file = sys.argv[1]
    target_file = sys.argv[2]
    #load the model
    model = wvModel.Model()
    main(model, pred_file, target_file)    

