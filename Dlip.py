import xlsxwriter
import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt

workbook = xlsxwriter.Workbook(
    'Distance10.xlsx')  # name of the excel sheet would be saved in the save folder of the code
worksheet = workbook.add_worksheet()
row = 1
DLipR = []
DLipL = []
Dlip = []
worksheet.write(0, 0, 'D_lipR')
worksheet.write(0, 1, 'D_lipL')
worksheet.write(0, 2, 'D_lip')
df = pd.read_excel(r'.\smiles\pair3id5N.xlsx')
# print(df)

RightX1 = df['X point Right'][0] - df['X point Nose tip'][0]
RightY1 = df['Y point Right'][0] - df['Y point Nose tip'][0]
Right1 = (RightX1, RightY1)
LeftX1 = df['X point Left'][0] - df['X point Nose tip'][0]
LeftY1 = df['Y point Left'][0] - df['Y point Nose tip'][0]
Left1 = (LeftX1, LeftY1)
CenterX1 = (RightX1 + LeftX1) / 2
CenterY1 = (RightY1 + LeftY1) / 2
Center1 = (CenterX1, CenterY1)
for i in range(0, len(df['Time']) - 1):
    # Nomred right and Left based on the distance from nose tip (Landmark 1 on mediapipe)

    RightX = df['X point Right'][i] - df['X point Nose tip'][i]
    RightY = df['Y point Right'][i] - df['Y point Nose tip'][i]
    Right = (RightX, RightY)

    LeftX = df['X point Left'][i] - df['X point Nose tip'][i]
    LeftY = df['Y point Left'][i] - df['Y point Nose tip'][i]
    Left = (LeftX, LeftY)

    # calculate the Central point from left + right /2

    CenterX = (RightX + LeftX) / 2
    CenterY = (RightY + LeftY) / 2
    Center = (CenterX, CenterY)

    # dstRight = distance.euclidean(Right, Center)
    # dstLeft = distance.euclidean(Left, Center)
    D_lipR = distance.euclidean(Center, Right)
    D_lipL = distance.euclidean(Center, Left)
    D_lipR_Temp = distance.euclidean(Center1, Right)
    D_lipL_Temp = distance.euclidean(Center1, Left)
    D_lip_first = 2 * distance.euclidean(Right1, Left1)
    D_lip = (D_lipR_Temp + D_lipL_Temp) / D_lip_first
    # D_lip = (dstRight + dstLeft) / 2 / dst
    worksheet.write(row, 0, D_lipR)
    worksheet.write(row, 1, D_lipL)
    worksheet.write(row, 2, D_lip)
    row = row + 1
    DLipR.append(D_lipR)
    DLipL.append(D_lipL)
    # print(dstRight)
    # print(dstLeft)
    # print(dst)
    # print(D_lip)
    Dlip.append(D_lip)
plt.plot(Dlip)
plt.ylabel('Distance R as pixels')
plt.xlabel('Frames')
plt.show()

workbook.close()
