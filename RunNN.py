from fann2 import libfann
import pickle
import numpy as np

# load neural network
ann = libfann.neural_net()
ann.create_from_file("NN.net")

# obvious
alpha = 0.5
numWords = 500

# load pickle 
wordsInOrder = pickle.load( open( "wordsInOrder.p", "rb" ) )
wordProb = pickle.load( open( "wordProbability.p", "rb" ) )
rWordsInOrder = {v: k for k, v in wordsInOrder.items()}


# set up initial state of the whatever
recentWordProb = np.copy(wordProb)
nextWord = np.zeros(np.shape(recentWordProb))
nFeatures = np.shape(recentWordProb)[0]-1

story = ""
for i in range(0,numWords):
    
    prediction = np.asarray(ann.run(recentWordProb))
    prediction -= np.min(prediction)
    prediction /= np.sum(prediction)
    
    randomNumber = np.random.rand()
    
    for i in range(0,nFeatures+1):
        
        randomNumber -= prediction[i]
        if (randomNumber <= 0):
            break

    nextWord[i] = 1

    recentWordProb = (1-alpha)*recentWordProb + alpha*nextWord

    nextWord[i] = 0

    if (i == nFeatures):
        story += "__unknown__ "
    else:
        story += (rWordsInOrder[i] + " ")

print story
