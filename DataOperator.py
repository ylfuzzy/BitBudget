#!/usr/bin/python3
import csv
import re
import os

class DataOperator:
    def __init__(self):
        self.filename = 'dpTable2222.dat'
        self.offset = 0
        self.lineOffset = []
        self.deleteFile()
        self.file = None
    
    def __initFileWriter(self):
        if self.file == None:
            self.file = open(self.filename, 'w')
    
    def __initFileReader(self):
        if self.file == None:
            self.file = open(self.filename, 'r')
    
    def deleteFile(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
    
    def closeFile(self):
        if self.file != None:
            self.file.close()
            self.file = None

    def changeFilename(self, filename):
        self.filename = filename

    def genOutputTupleSeqStr(self, rowTuples):
        outputStr = ''
        for t in rowTuples:
            outputStr += str(t[0]) + ' ' + str(t[1]) + ','
        outputStr = outputStr[:-1] + '\n'
        return outputStr

    def writeDPChunkToFile(self, rows):
        try:
            self.__initFileWriter()
            for r in rows:
                line = self.genOutputTupleSeqStr(r)
                self.file.write(line)
                self.lineOffset.append(self.offset)
                self.offset += len(line)
        except:
            self.closeFile()
            self.deleteFile()
            raise

    def convertTextRowToDPRow(self, textRow):
        textRow = textRow.split(',')
        dpRow = [[float(value.split(' ')[0]), int(value.split(' ')[1])] for value in textRow]
        return dpRow

    def readRowByLineOffset(self, iLine):
        textRow = ''
        self.file.seek(self.lineOffset[iLine])
        textRow = self.file.readline()
        return self.convertTextRowToDPRow(textRow)
    
    def readAChunkFromFile(self, iStart, numLines, reverse=True):
        try:
            self.__initFileReader()
            tick = 1
            if reverse:
                tick = -1
                numLines = -numLines

            dpRows = []
            for iLine in range(iStart, iStart + numLines, tick):
                dpRows.append(self.readRowByLineOffset(iLine))

                # Break the loop
                if reverse and iLine == 0 or not reverse and iLine == len(self.lineOffset) - 1:
                    break
        except:
            self.closeFile()
            self.deleteFile()
            raise
        
        return dpRows

if __name__ == '__main__':
    dataOperator = DataOperator()
    t = [[(3.141582738245324080139653233346, 2), (1., 345), (5, 6)], [(1, 2), (1, 3), (5, 6)], [(1, 2), (1, 3), (5, 6)]]
    #print(dataOperator.genOutputTupleSeqStr(t[0]))
    dataOperator.writeDPChunkToFile(t)
    dataOperator.writeDPChunkToFile(t)
    # dataOperator.readRowsFromFile()
    #print(dataOperator.readAChunkFromFile(2, 10))