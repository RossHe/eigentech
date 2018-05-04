# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:43:51 2018

@author: Harold
"""
import numpy as np
import sys

def blocks(file, size=65536):
    while True:
        b = file.read(size)
        if not b: break
        yield b

def read_1000lines(filename):
    linesCount = 0
    with open(filename, "r") as f:
        for bl in blocks(f):
            linesCount += bl.count('\n')
    mask = np.random.choice(linesCount,1000)
    i = -1
    lines = []
    with open(filename,'r') as f:
        for line in f:
            i += 1
            if i in mask:
                lines.append(line)
    return lines

if __name__ == '__main__':
    filename = sys.argv[1]
    lines = read_1000lines(filename)