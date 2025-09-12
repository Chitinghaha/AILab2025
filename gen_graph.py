
import os
import src.management.management as mngt
import argparse
import csv

import numpy as np
import cv2

parser = argparse.ArgumentParser()

parser.add_argument('-c','--csv', required=True, help='csv file')
parser.add_argument('-o','--output', required=True, help='output image filePath')

args = parser.parse_args()

second = int(0)
maxAddr = 0


tensor = {}
with open(args.csv, 'r', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        # second += int(round(float(row[0]),2))
        second += 1
        for i in range(1,len(row),3):
            if row[i] != 'empty':
                startAddr = int(row[i+1])
                endAddr = int(row[i+2])
                if row[i] in tensor.keys():
                    tensor[row[i]]["end"] = second
                else:
                    tensor[row[i]] = {"start":second-1,"end":second,'startAddr':startAddr,'endAddr':endAddr}
                if endAddr > maxAddr: maxAddr = endAddr
            else: continue


print(f"Memory Requirement : {maxAddr/(1024)}KB")


img = np.ones((512,1060,3), np.uint8)*255

yStart = 50
yEnd = 450

xStart = 50
xEnd = 1050


pivotX = second / (xEnd - xStart) 
pivotY = maxAddr / (yEnd - yStart) 
memMax = int(maxAddr/pivotY)

cv2.line(img, (xStart, yStart - 10), (xStart, yEnd), (0, 0, 0), 1)
cv2.line(img, (xEnd, yEnd), (xStart, yEnd), (0, 0, 0), 1)
for tensorName in tensor:
    X = [int(tensor[tensorName]['start']),int(tensor[tensorName]['end']) ]
    X[0], X[1] = int(X[0] / pivotX), int(X[1] / pivotX)
    Y = [int(tensor[tensorName]['startAddr']),int(tensor[tensorName]['endAddr']) ] 
    Y[0], Y[1] = int(Y[0] / pivotY), int(Y[1] / pivotY + 1)
    cv2.rectangle(img, (xStart + X[0] , yEnd - Y[1]), (xStart + X[1] , yEnd - Y[0] - 1), (0, 0, 0), 1)
cv2.line(img, (xEnd, yEnd - memMax - 1), (xStart, yEnd - memMax - 1), (0, 0, 255), 1)
cv2.putText(img, f"{(maxAddr/(1024*1024))}MB", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
cv2.imwrite(os.path.join(args.output), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])