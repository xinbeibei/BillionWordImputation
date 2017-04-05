# CSCI-544 Final Project

# Team members: <br />
Jared Sagendorf<br />
Roelof Groenewald<br />
Beibei Xin<br />

# word_embedding.py
This program trains the word2vec model, and saves the model to a file. To run this, you need to install the package gensim https://radimrehurek.com/gensim/install.html <br />

To use it:
python word_embedding.py /path/to/training/data
<br />
and it will output a file called w2v.mdl, which is the trained model. This can then be loaded in other programs. I included a model trained on the entire Kaggle training data.

# test_embedding.py
This shows how to load the trained model from a file, and evaluates the model using 20,000 semantic tests publised by google. These are contained in the file "questsions-words.txt". This evaluation is only a rough guide to how good the embedding is - it doesn't mean it's good or bad for our particular application. 
