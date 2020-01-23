#!/usr/bin/python3
import numpy as np
import random
import copy

class RandomSelector:
    def getRandomizedMatrix(self, originalMatrix, bIndexes, iQuantized, ii, jj, kk=None):
        randomizedMatrix = copy.deepcopy(bIndexes)
        for i in range(np.shape(randomizedMatrix)[0]):
            if i % 2 == 0:
                w = randomizedMatrix[i]
                for iW in range(len(w)):
                    for jW in range(len(w[0])):
                        iRandom = random.randint(0,3)
                        w[iW][jW] = w[iW][jW][iRandom]
                        # print(w[iW][jW])
            else:
                bias = randomizedMatrix[i]
                for iB in range(len(bias)):
                    iRandom = random.randint(0,3)
                    bias[iB] = bias[iB][iRandom]

        randomizedMatrixQuantized = copy.deepcopy(randomizedMatrix)
        if kk != None:
            randomizedMatrix[ii][jj][kk] = originalMatrix[ii][jj][kk]
            randomizedMatrixQuantized[ii][jj][kk] = bIndexes[ii][jj][kk][iQuantized]
            # print(randomizedMatrixQuantized[ii][jj][kk])
            # print(bIndexes[ii][jj][kk][iQuantized])
        else:
            randomizedMatrix[ii][jj] = originalMatrix[ii][jj]
            randomizedMatrixQuantized[ii][jj] = bIndexes[ii][jj][iQuantized]

        return randomizedMatrix, randomizedMatrixQuantized
    

if __name__ == '__main__':
    bIndexes = np.load('./after_GQ_B_0123_H6.npy')
    originalMatrix = np.load('./Original_weights_H6_H3_Adam_Relu_Softmax_01_22.npy')
    print(np.shape(bIndexes[3]))
    print(bIndexes)
    # randomSelector = RandomSelector()
    # randomizedMatrix, randomizedMatrixQuanitzed = randomSelector.getRandomizedMatrix(originalMatrix, bIndexes, 0, 0, 0, 0)
    # # print(np.shape(randomizedMatrix[1]))
    # print(randomizedMatrix[0][0][0])
    # print('------')
    # print(randomizedMatrixQuanitzed[0][0][0])