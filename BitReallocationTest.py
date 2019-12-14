#!/usr/bin/python3
import unittest
import random
import numpy as np
from BitBudgetReallocation import BitBudgetReallocation

class TestCaseGenerator:
    def genTestCase(self, numRow, numCol):
        numParCandidates = 4
        weight = []
        for i in range(numRow):
            row = []
            for j in range(numCol):
                row.append([random.uniform(-10, 10) for i in range(numParCandidates)])
            weight.append(row)
        bias = []
        for j in range(numCol):
            bias.append([random.uniform(-10, 10) for i in range(numParCandidates)])
        return [weight, bias]
        #print(np.shape(cIndexes))
        #dummy = [list([[[0, -1, 2, 3],[1, 5, -2, 3]]]), list([[10, 1, 2, 3], [2, -7, 0, 4]])]
        #print(np.shape(dummy))
    
    def getCKeras(self, numRow, numCol):
        numParCandidates = 4
        weight = []
        for i in range(numRow):
            row = []
            for j in range(numCol):
                row.append(0)
            weight.append(row)
        bias = []
        for j in range(numCol):
            bias.append(0)
        return [weight, bias]


class BitReallocationTest(unittest.TestCase):
    def testNumLines87Vs300(self):
        cIndexes = np.load('./H32_32_ED/cIndexes_results_ED_GQ_B_Relocation_valid_32_32_Hidden_12_11.npy')
        bIndexes = np.load('./H32_32_ED/bIndexes_after_GQ_H32_32_B_0123_Relocation_12_11.npy')
        cKeras = np.load('./H32_32_ED/cKeras_Original_weights_H32_32_ES_SGD_12_11.npy')

        reallocator = BitBudgetReallocation(0.5, cKeras, cIndexes, bIndexes, minimum=True, saveToDisk=True, threshold=123, readThreshold=100)
        t1 = reallocator.genReallocatedMatrix()
        reallocator = BitBudgetReallocation(0.5, cKeras, cIndexes, bIndexes, minimum=True, saveToDisk=True, threshold=300, readThreshold=170)
        t2 = reallocator.genReallocatedMatrix()
        self.assertEqual(t1, t2)
    #def testR1C2Avg2(self):
    #    numRow = 3
    #    numCol = 10
    #    avgBits = 3
    #    tcg = TestCaseGenerator()
    #    for i in range(1):
    #        cIndexes = tcg.genTestCase(numRow, numCol)
    #        bIndexes = tcg.genTestCase(numRow, numCol)
    #        cKeras = tcg.getCKeras(numRow, numCol)
    #        reallocator = BitBudgetReallocation(avgBits, cKeras, cIndexes, bIndexes)
    #        (valBF, bitParsBF, relocatedMatrixBF) = reallocator.solveByBruteForce()
    #        dp = reallocator.buildUpDPTable()
    #        print(relocatedMatrixBF)
    #        relocatedMatrixDP = reallocator.genReallocatedMatrix()
    #        print()
    #        print(relocatedMatrixDP)
    #        self.assertEqual(valBF, dp[-1][-1][0])
    #
    #def testR1C2Avg3(self):
    #    numRow = 1
    #    numCol = 2
    #    avgBits = 3
    #    tcg = TestCaseGenerator()
    #    cIndexes = tcg.genTestCase(numRow, numCol)
    #    bIndexes = tcg.genTestCase(numRow, numCol)
    #    cKeras = tcg.getCKeras(numRow, numCol)
    #    reallocator = BitBudgetReallocation(avgBits, cKeras, cIndexes, bIndexes)
    #    valDFS = reallocator.solveByBruteForce()
    #    dp = reallocator.buildUpDPTable()
    #    self.assertEqual(valDFS, dp[-1][-1][0])

    #def testR2C3Avg2(self):
    #    numRow = 2
    #    numCol = 3
    #    avgBits = 2
    #    tcg = TestCaseGenerator()
    #    for i in range(100):
    #        cIndexes = tcg.genTestCase(numRow, numCol)
    #        bIndexes = tcg.genTestCase(numRow, numCol)
    #        cKeras = tcg.getCKeras(numRow, numCol)
    #        reallocator = BitBudgetReallocation(avgBits, cKeras, cIndexes, bIndexes)
    #        valDFS = reallocator.solveByBruteForce()
    #        dp = reallocator.buildUpDPTable()
    #        self.assertEqual(valDFS, dp[-1][-1][0])

    #def testR3C3Avg3(self):
    #    numRow = 3
    #    numCol = 3
    #    avgBits = 3
    #    tcg = TestCaseGenerator()
    #    cIndexes = tcg.genTestCase(numRow, numCol)
    #    bIndexes = tcg.genTestCase(numRow, numCol)
    #    cKeras = tcg.getCKeras(numRow, numCol)
    #    reallocator = BitBudgetReallocation(avgBits, cKeras, cIndexes, bIndexes)
    #    valDFS = reallocator.solveByBruteForce()
    #    dp = reallocator.buildUpDPTable()
    #    self.assertEqual(valDFS, dp[-1][-1][0])

if __name__ == '__main__':
    unittest.main()