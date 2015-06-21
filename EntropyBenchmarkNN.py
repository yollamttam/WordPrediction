import nltk
import glob
import pickle
import numpy as np
from fann2 import libfann

# obvious
alpha = 0.5
nExamples = 0
fileToEntropy = {}
# load pickle 
wordsInOrder = pickle.load( open( "wordsInOrder.p", "rb" ) )
wordProb = pickle.load( open( "wordProbability.p", "rb" ) )

# load neural network
ann = libfann.neural_net()
ann.create_from_file("NN.net")

nFeatures = np.shape(wordProb)[0]-1

files = glob.glob('reuters/training/*')
files = files[:10]
print files
fileNum = 0
for filename in files:
    entropy = 0
    fileNum += 1
    print "%d of %d" % (fileNum,len(files))
    openfile = open(filename,'r')
    readfile = openfile.read()
    tokens = nltk.word_tokenize(readfile)

    # set up initial state of the whatever
    recentWordProb = np.copy(wordProb)
    nextWord = np.zeros(np.shape(recentWordProb))
 
    

    # loop through tokens
    for token in tokens:
        prediction = np.asarray(ann.run(recentWordProb))
        # print np.sum(prediction)
        prediction /= np.sum(prediction)
        token = token.lower()
        if (token in wordsInOrder):
            tokenIndex = wordsInOrder[token]
        else:
            tokenIndex = nFeatures
    
            
        logProb = np.min((50,-1*np.log(prediction[tokenIndex])))
        entropy += logProb

        nextWord[tokenIndex] = 1
        
        recentWordProb = (1-alpha)*recentWordProb + alpha*nextWord

        nextWord[tokenIndex] = 0

    entropy /= len(tokens)
    print entropy
    fileToEntropy[filename] = entropy

    openfile.close()

avgEntropy = 0
for value in fileToEntropy.itervalues():
    avgEntropy += value

avgEntropy /= len(fileToEntropy)
print avgEntropy

pickle.dump(fileToEntropy,open("fileToEntropy.p", "wb" ))
