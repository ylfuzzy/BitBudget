#!/usr/bin/python3
import numpy as np
import random
import copy

class RandomSelector:
    def getRandomizedMatrix(self, originalMatrix, bIndexes, ii, jj, kk=None):
        randomizedMatrix = copy.deepcopy(bIndexes)
        for i in range(np.shape(randomizedMatrix)[0]):
            if i % 2 == 0:
                w = randomizedMatrix[i]
                for iW in range(np.shape(w)[0]):
                    for jW in range(np.shape(w)[1]):
                        iRandom = random.randint(0,3)
                        w[iW][jW] = w[iW][jW][iRandom]
                        # print(w[iW][jW])
            else:
                bias = randomizedMatrix[i]
                for iB in range(np.shape(bias)[0]):
                    iRandom = random.randint(0,3)
                    bias[iB] = bias[iB][iRandom]
        if kk != None:
            randomizedMatrix[ii][jj][kk] = originalMatrix[ii][jj][kk]
        else:
            randomizedMatrix[ii][jj] = originalMatrix[ii][jj]

        return randomizedMatrix
    

if __name__ == '__main__':
    bIndexes = np.load('./after_GQ_B_0123_H6.npy')
    print(np.shape(bIndexes[3]))
    randomSelector = RandomSelector()
    randomizedMatrix = randomSelector.getRandomizedMatrix(None, bIndexes, 0, 0, 0)
    print(np.shape(randomizedMatrix[1]))
    print(randomizedMatrix[0][0][0])