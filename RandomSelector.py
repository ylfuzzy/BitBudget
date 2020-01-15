#!/usr/bin/python3
import numpy as np
import random

class RandomSelector:
    def randomSelect(self, bIndexes):
        for i in range(np.shape(bIndexes)[0]):
            if i % 2 == 0:
                w = bIndexes[i]
                for iW in range(np.shape(w)[0]):
                    for jW in range(np.shape(w)[1]):
                        iRandom = random.randint(0,3)
                        w[iW][jW] = w[iW][jW][iRandom]
                        # print(w[iW][jW])
            else:
                bias = bIndexes[i]
                for iB in range(np.shape(bias)[0]):
                    iRandom = random.randint(0,3)
                    bias[iB] = bias[iB][iRandom]
            # else:
            #     b = self.cIndexes[i]
            #     for iB in range(np.shape(b)[0]):
            #         if self.minimum:
            #             # Inverse the value to find the MININUM
            #             b[iB] = [-v for v in b[iB]]

            #         # Don't consider 0th bit's impact
            #         #b[iB][0] = 0
            #         list1DTo3D.append((i, iB))
    

bIndexes = np.load('./H32_32_ED/bIndexes_after_GQ_H32_32_B_0123_Relocation_12_11.npy')
print(np.shape(bIndexes[0]))
randomSelector = RandomSelector()
randomSelector.randomSelect(bIndexes)
print(np.shape(bIndexes[0]))