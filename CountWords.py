import nltk
import glob
import operator
import pickle
import numpy as np
# word counts
wordCounts = {}
wordsInOrder = {}
totalWords = 0
topCounts = 0
nFeatures = 5000
wordProbability = np.zeros((nFeatures+1,1))
# loop through the files
files = glob.glob('reuters/training/*')
for file in files:
    openfile = open(file,'r')
    readfile = openfile.read()
    tokens = nltk.word_tokenize(readfile)
    
    # loop through tokens
    for token in tokens:
        token = token.lower()
        if (token not in wordCounts):
            wordCounts[token] = 0
            
        wordCounts[token] += 1
        totalWords += 1

sortedWordCounts = sorted(wordCounts.items(), key=operator.itemgetter(1),reverse=True)


for i in range(0,nFeatures):
    print sortedWordCounts[i]
    topCounts += sortedWordCounts[i][1]
    wordsInOrder[sortedWordCounts[i][0]] = i
    wordProbability[i] = sortedWordCounts[i][1]*1.0/totalWords

wordProbability[nFeatures] = 1 - topCounts*1.0/totalWords

print topCounts*1.0/totalWords

pickle.dump(wordsInOrder,open("wordsInOrder.p", "wb" ))
pickle.dump(wordProbability,open("wordProbability.p", "wb" ))



