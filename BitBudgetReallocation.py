#!/usr/bin/python3
import numpy as np
import time
import copy
from DataOperator import DataOperator

class BitBudgetReallocation:
    def __init__(self, parAvgBits, cKeras, cIndexes, bIndexes, minimum=False, saveToDisk=False, threshold=300, readThreshold=100):
        self.cKeras = cKeras
        self.cIndexes = cIndexes
        self.bIndexes = bIndexes
        self.minimum = minimum
        self.saveToDisk = saveToDisk
        self.dataOperator = None
        if self.saveToDisk:
            self.dataOperator = DataOperator()
            self.threshold = threshold
            self.readThreshold = readThreshold
        self.list1DTo3D = self.mapOneDToThreeD()
        print('num pars: ', len(self.list1DTo3D))
        self.bitBudget = int(len(self.list1DTo3D) * parAvgBits)
    
        # For dfs
        self.dfsMaxVal = -np.inf
        self.dfsBitPars = []
        self.dfsTempBitPars = []
    
    def mapOneDToThreeD(self):
        list1DTo3D = []
        for i in range(np.shape(self.cIndexes)[0]):
            if i % 2 == 0:
                w = self.cIndexes[i]
                for iW in range(np.shape(w)[0]):
                    for jW in range(np.shape(w)[1]):
                        if self.minimum:
                            # Inverse the value to find the minimum
                            w[iW][jW] = [-v for v in w[iW][jW]]

                        # Don't consider 0th bit's impact
                        #w[iW][jW][0] = 0
                        list1DTo3D.append((i, iW, jW))
            else:
                b = self.cIndexes[i]
                for iB in range(np.shape(b)[0]):
                    if self.minimum:
                        # Inverse the value to find the MININUM
                        b[iB] = [-v for v in b[iB]]

                    # Don't consider 0th bit's impact
                    #b[iB][0] = 0
                    list1DTo3D.append((i, iB))
        return list1DTo3D
    
    def solveByBruteForce(self):
        self.dfsMaxVal = -np.inf
        self.dfsBitPars = []
        self.dfsTempBitPars = []
        self.__dfs(0)
        relocatedMatrix = self.init_origin_w_zero()
        for i in range(len(self.list1DTo3D)):
            numBitsUsed = self.dfsBitPars[i]
            idxTuple = self.list1DTo3D[i]
            i = idxTuple[0]
            j = idxTuple[1]
            if len(idxTuple) == 3:
                k = idxTuple[2]
                relocatedMatrix[i][j][k] = self.bIndexes[i][j][k][numBitsUsed]
            else:
                relocatedMatrix[i][j] = self.bIndexes[i][j][numBitsUsed]
            
        return (self.dfsMaxVal, self.dfsBitPars, relocatedMatrix)
                    

    def __dfs(self, i):
        if i == len(self.list1DTo3D):
            if sum(self.dfsTempBitPars) <= self.bitBudget:
                curVal = 0
                for iP in range(len(self.dfsTempBitPars)):
                    idxTuple = self.list1DTo3D[iP]
                    cPars = self.cIndexes[idxTuple[0]][idxTuple[1]]
                    weightDim = 3
                    if len(idxTuple) == weightDim:
                        cPars = self.cIndexes[idxTuple[0]][idxTuple[1]][idxTuple[2]]
                    curVal += cPars[self.dfsTempBitPars[iP]]
                if curVal > self.dfsMaxVal:
                    self.dfsMaxVal = curVal
                    self.dfsBitPars = copy.deepcopy(self.dfsTempBitPars)
            return None
        for iB in range(4):
            self.dfsTempBitPars.append(iB)
            self.__dfs(i + 1)
            self.dfsTempBitPars = self.dfsTempBitPars[:-1]

    def buildUpDPTableSaveToDisk(self):
        # Initialize the base row
        dpChunk = [[0] * (self.bitBudget + 1)]
        for j in range(self.bitBudget + 1):
            firstIdxTuple = self.list1DTo3D[0]
            cPars = self.cIndexes[firstIdxTuple[0]][firstIdxTuple[1]][firstIdxTuple[2]]
            maxVal = cPars[0]
            numBitsMaxVal = 0
            for numBits in range(len(cPars)):
                if j >= numBits and cPars[numBits] > maxVal:
                    maxVal = cPars[numBits]
                    numBitsMaxVal = numBits
            dpChunk[0][j] = (maxVal, numBitsMaxVal)
        
        # Write the base case to the file
        self.dataOperator.writeDPChunkToFile(dpChunk)
        
        # Build up the table
        threshold = self.threshold
        numLines = 0
        numSegments = 0

        # For debug
        numLinesWrite = 1
        for i in range(1, len(self.list1DTo3D)):
            # Generate a new row
            dpChunk.append([0] * (self.bitBudget + 1))
            idxTuple = self.list1DTo3D[i]
            cPars = self.cIndexes[idxTuple[0]][idxTuple[1]]
            weightDim = 3
            if len(idxTuple) == weightDim:
                cPars = self.cIndexes[idxTuple[0]][idxTuple[1]][idxTuple[2]]
            
            # Calculate the offset
            offset = threshold * numSegments
            iOffset = i - offset
        
            for j in range(self.bitBudget + 1):
                maxVal = dpChunk[iOffset - 1][j][0] + cPars[0]
                numBitsMaxVal = 0
                for numBits in range(len(cPars)):
                    newVal = cPars[numBits] + dpChunk[iOffset - 1][j - numBits][0]
                    if j >= numBits and newVal > maxVal:
                        maxVal = newVal
                        numBitsMaxVal = numBits
                dpChunk[iOffset][j] = (maxVal, numBitsMaxVal)
            
            # Calculate current number of lines
            # If it already equals to the threshold, save them to the disk
            numLines += 1
            if numLines == threshold or i == len(self.list1DTo3D) - 1:
                numLines = 0
                numSegments += 1
                self.dataOperator.writeDPChunkToFile(dpChunk[1:])
                print('dpChunk len: ', len(dpChunk[1:]))
                numLinesWrite += len(dpChunk[1:])
                lastRow = dpChunk[-1]
                dpChunk = [lastRow]
                print('numSegments: ', numSegments)

        # Close the file after writing the entire dp table
        self.dataOperator.closeFile()
        print('numLineWrite: ', numLinesWrite)
        return dpChunk

    def buildUpDPTable(self):
        dp = [[0] * (self.bitBudget + 1) for i in range(len(self.list1DTo3D))]

        # Initialize the base case
        for j in range(self.bitBudget + 1):
            firstIdxTuple = self.list1DTo3D[0]
            cPars = self.cIndexes[firstIdxTuple[0]][firstIdxTuple[1]][firstIdxTuple[2]]
            maxVal = cPars[0]
            numBitsMaxVal = 0
            for numBits in range(len(cPars)):
                if j >= numBits and cPars[numBits] > maxVal:
                    maxVal = cPars[numBits]
                    numBitsMaxVal = numBits
            dp[0][j] = (maxVal, numBitsMaxVal)

        # Build up the dp table
        for i in range(1, len(self.list1DTo3D)):
            idxTuple = self.list1DTo3D[i]
            cPars = self.cIndexes[idxTuple[0]][idxTuple[1]]
            weightDim = 3
            if len(idxTuple) == weightDim:
                cPars = self.cIndexes[idxTuple[0]][idxTuple[1]][idxTuple[2]]

            for j in range(self.bitBudget + 1):
                maxVal = dp[i - 1][j][0] + cPars[0]
                numBitsMaxVal = 0
                for numBits in range(len(cPars)):
                    newVal = cPars[numBits] + dp[i - 1][j - numBits][0]
                    if j >= numBits and newVal > maxVal:
                        maxVal = newVal
                        numBitsMaxVal = numBits
                dp[i][j] = (maxVal, numBitsMaxVal)
        return dp

    def getRelocatedPars(self):
        dp = self.buildUpDPTable()
        relocatedPars = []
        i = len(dp) - 1
        j = len(dp[0]) - 1
        while i >= 0:
            numBitsUsed = dp[i][j][1]
            relocatedPars.append((self.list1DTo3D[i], numBitsUsed))
            i -= 1
            j -= numBitsUsed
        return relocatedPars
    
    def genRelocatedParsFromFile(self):
        # self.dataOperator.changeFilename('testCase.dat')
        # print(len(self.dataOperator.lineOffset))
        relocatedPars = []
        iPar = len(self.list1DTo3D) - 1
        numLines = self.readThreshold
        jChunk = self.bitBudget

        # For debug
        numLinesRead = 0
        while iPar >= 0:
            dpChunk = self.dataOperator.readAChunkFromFile(iPar, numLines)
            print('num dpChunk read: ', len(dpChunk))
            numLinesRead += len(dpChunk)
            for iChunk in range(len(dpChunk)):
                numBitsUsed = dpChunk[iChunk][jChunk][1]
                relocatedPars.append((self.list1DTo3D[iPar], numBitsUsed))
                jChunk -= numBitsUsed
                iPar -= 1
        
        # Close the file after reading the entire dp table
        self.dataOperator.closeFile()
        self.dataOperator.deleteFile()
        print('Total Lines Read: ', numLinesRead)
        return relocatedPars        


    def cal2ndBitInpact(self):
        all2ndBitInpact = 0
        for p in self.list1DTo3D:
            i = p[0]
            j = p[1]
            if len(p) == 3:
                k = p[2]
                all2ndBitInpact += self.cIndexes[i][j][k][2]
            else:
                all2ndBitInpact += self.cIndexes[i][j][2]
        return all2ndBitInpact
            

    def init_origin_w_zero(self):
        init_val_matrix = []
        for i in range(len(self.cKeras)):
            arr_mask = self.cKeras[i]
            if i%2 == 0:
                arr_append = [[0 for j in i] for i in arr_mask]
            else:
                arr_append = [0 for i in arr_mask]
            init_val_matrix.append(arr_append)
        return init_val_matrix
    
    def genReallocatedMatrix(self):
        relocatedPars = None
        if self.saveToDisk:
            self.buildUpDPTableSaveToDisk()
            relocatedPars = self.genRelocatedParsFromFile()
        else:
            relocatedPars = self.getRelocatedPars()
        relocatedMatrix = self.init_origin_w_zero()
        for p in relocatedPars:
            idxMultiD = p[0]
            numBitsUsed = p[1]
            i = idxMultiD[0]
            j = idxMultiD[1]
            if len(idxMultiD) == 3:
                k = idxMultiD[2]
                relocatedMatrix[i][j][k] = self.bIndexes[i][j][k][numBitsUsed]
            else:
                relocatedMatrix[i][j] = self.bIndexes[i][j][numBitsUsed]
        return relocatedMatrix
                

if __name__ == '__main__':
    cIndexesPath = './c_results_GQ_B_0123_Relocation_valid.npy'
    bIndexesPath = './b_after_GQ_B_0123_Relocation_No_Hidden.npy'
    cKerasPath = './Original_weights_ES_SGD_LogReg_Softmax_11_18.npy'
    test = True
    bigFile = True
    if test:
        cIndexes = [list([[[0, -1, 2, 3],[1, 5, -2, 3]]]), list([[10, 1, 2, 3], [2, -7, 0, 4]])]
        bIndexes = [list([[[0, -1, 2, 3],[1, 5, -2, 3]]]), list([[10, 1, 2, 3], [2, -7, 0, 4]])]
        cKeras = [np.array([[0.1, -0.5]]), np.array([-3, -5])]
    else:
        cIndexes = np.load(cIndexesPath)
        bIndexes = np.load(bIndexesPath)
        cKeras = np.load(cKerasPath)
    if bigFile:
        cIndexes = np.load('./H32_32_ED/cIndexes_results_ED_GQ_B_Relocation_valid_32_32_Hidden_12_11.npy')
        bIndexes = np.load('./H32_32_ED/bIndexes_after_GQ_H32_32_B_0123_Relocation_12_11.npy')
        cKeras = np.load('./H32_32_ED/cKeras_Original_weights_H32_32_ES_SGD_12_11.npy')

    print(np.shape(cIndexes[0]))
    print(np.shape(bIndexes[0]))
    print(np.shape(cKeras[0]))
    # bitBudget = 2 * 4
    tStart = time.time()
    reallocator = BitBudgetReallocation(0.5, cKeras, cIndexes, bIndexes, minimum=True, saveToDisk=True)
    matrix = reallocator.genReallocatedMatrix()
    #print(matrix)
    np.save('testNowFileExpected.npy', matrix)
    # reallocator = BitBudgetReallocation(2, cKeras, cIndexes, bIndexes, saveToDisk=False)
    # print(reallocator.genReallocatedMatrix())
    # dpChunk = reallocator.buildUpDPTableSaveToDisk()
    # print(dpChunk[0][-1])
    # dp = reallocator.buildUpDPTable()
    # print(dp[-1][-1])
    #print(reallocator.cIndexes)
    #print(reallocator.cal2ndBitInpact())
    #dp = reallocator.buildUpDPTable()
    #print(dp[-1][-1][0])
    #relocatedMatrix = reallocator.genRelacatedMatrix()
    #print(relocatedMatrix)
    #np.save('BitReallocatedWeight.npy', relocatedMatrix)
    tEnd = time.time()
    tTotal = tEnd - tStart
    print('Total Time: ', tTotal)