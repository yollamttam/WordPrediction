import nltk
import glob
import pickle
import numpy as np

# obvious
alpha = 0.5

# load pickle 
wordsInOrder = pickle.load( open( "wordsInOrder.p", "rb" ) )
wordProb = pickle.load( open( "wordProbability.p", "rb" ) )

nFeatures = np.shape(wordProb)[0]-1

openfile = open('reuters/training/1','r')
readfile = openfile.read()
tokens = nltk.word_tokenize(readfile)

# set up initial state of the whatever
recentWordProb = np.copy(wordProb)
nextWord = np.zeros(np.shape(recentWordProb))

f_handle = file('TrainingData.txt','w',)

# loop through tokens
for token in tokens:
    token = token.lower()
    if (token in wordsInOrder):
        tokenIndex = wordsInOrder[token]
    else:
        tokenIndex = nFeatures
    
    nextWord[tokenIndex] = 1

    np.savetxt(f_handle,np.transpose(recentWordProb),fmt='%.6f',delimiter=' ')
    np.savetxt(f_handle,np.transpose(nextWord),fmt='%.1f',delimiter=' ')

    recentWordProb = (1-alpha)*recentWordProb + alpha*nextWord

    nextWord[tokenIndex] = 0



f_handle.close()

nExamples = len(tokens)

firstLine = "%d %d %d \n" % (nExamples,nFeatures+1,nFeatures+1)

with file('TrainingData.txt', 'r') as original: data = original.read()
with file('TrainingData.txt', 'w') as modified: modified.write(firstLine + data)
