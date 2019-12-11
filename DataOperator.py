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
            outputStr += '{:.20f}'.format(t[0]) + ' ' + '{:.20f}'.format(t[1]) + ','
        outputStr = outputStr[:-1] + '\n'
        return outputStr

    def writeRowsToFile(self, rows):
        with open(self.filename, 'a') as file:
            for r in rows:
                line = self.genOutputTupleSeqStr(r)
                file.write(line)
                self.lineOffset.append(self.offset)
                self.offset += len(line)
            file.close()
    
    def readRowsFromFile(self):
        l = []
        with open(self.filename, 'r') as file:
            for i, row in enumerate(file):
                if i == 0:
                    l = row.split(',')
            file.close()
        
        l = [re.sub('[()]', '', t) for t in l]
        l = [[float(t.split(' ')[0]), float(t.split(' ')[1])] for t in l]
        print(l)

    def convertTextRowToDPRow(self, textRow):
        textRow = textRow.split(',')
        dpRow = [[float(entity.split(' ')[0]), float(entity.split(' ')[1])] for entity in textRow]
        return dpRow


    def readRowByLineOffset(self, iLine):
        textRow = ''
        with open(self.filename, 'r') as file:
            file.seek(self.lineOffset[iLine])
            textRow = file.readline()
            file.close()
        return self.convertTextRowToDPRow(textRow)

dataOperator = DataOperator()
t = [[(3.141582738245324080139653233346, 2), (1., 345), (5, 6)], [(1, 2), (1, 3), (5, 6)], [(1, 2), (1, 3), (5, 6)]]
#print(dataOperator.genOutputTupleSeqStr(t[0]))
dataOperator.writeRowsToFile(t)
# dataOperator.readRowsFromFile()
print(dataOperator.readRowByLineOffset(0))