#!/usr/bin/python3
import csv
import re
import os

class DataOperator:
    def __init__(self):
        self.filename = 'test2222.dat'
        self.offset = 0
        self.lineOffset = []
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def genOutputTupleSeqStr(self, rowTuples):
        outputStr = ''
        for t in rowTuples:
            outputStr += '{:.20f}'.format(t[0]) + ' ' + str(t[1]) + ','
        outputStr = outputStr[:-1] + '\n'
        return outputStr

    def writeDPChunkToFile(self, rows):
        with open(self.filename, 'a') as file:
            for r in rows:
                line = self.genOutputTupleSeqStr(r)
                file.write(line)
                self.lineOffset.append(self.offset)
                self.offset += len(line)
            file.close()

    def convertTextRowToDPRow(self, textRow):
        textRow = textRow.split(',')
        dpRow = [[float(entity.split(' ')[0]), int(entity.split(' ')[1])] for entity in textRow]
        return dpRow

    def readRowByLineOffset(self, file, iLine):
        textRow = ''
        file.seek(self.lineOffset[iLine])
        textRow = file.readline()
        return self.convertTextRowToDPRow(textRow)
    
    def readAChunkFromFile(self, iStart, numLines, reverse=True):
        tick = 1
        if reverse:
            tick = -1
            numLines = -numLines
        dpRows = []
        with open(self.filename, 'r') as file:
            for iLine in range(iStart, iStart + numLines, tick):
                dpRows.append(self.readRowByLineOffset(file, iLine))

                # Break the loop
                if reverse and iLine == 0 or not reverse and iLine == len(self.lineOffset) - 1:
                    break
            file.close()

        return dpRows

if __name__ == '__main__':
    dataOperator = DataOperator()
    t = [[(3.141582738245324080139653233346, 2), (1., 345), (5, 6)], [(1, 2), (1, 3), (5, 6)], [(1, 2), (1, 3), (5, 6)]]
    #print(dataOperator.genOutputTupleSeqStr(t[0]))
    dataOperator.writeDPChunkToFile(t)
    dataOperator.writeDPChunkToFile(t)
    # dataOperator.readRowsFromFile()
    #print(dataOperator.readAChunkFromFile(2, 10))