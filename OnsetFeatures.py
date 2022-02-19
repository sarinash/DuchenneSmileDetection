import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import xlsxwriter
from moviepy.editor import VideoFileClip
import autograd.numpy as npp
import sympy as sym
from autograd import grad
from scipy.signal import savgol_filter

# Set some reasonable parameters to start with
w = 5
p = 2
ww = 1
clip = VideoFileClip('.\smiles\pair3id5N.mp4')
print('Time', clip.duration)
Time = clip.duration
cam = cv2.VideoCapture('.\smiles\pair3id5N.mp4')
df = pd.read_excel(r'Distance10.xlsx')
Distance = df['D_lip']
Distance1 = np.array(Distance)

MinValue = min(Distance1)

Distance1 = Distance1 - MinValue
MaxValue = max(Distance1)
Distance1 = Distance.rolling(3 * ww).mean()
onset_index = []
Onset_part = []
Apex_part = []


def LogestIncSubArr(arr, n):
    m = 1
    l = 1
    maxIndex = 0
    for i in range(1, n):
        if (arr[i] >= arr[i - 1]):
            l = l + 1
        else:
            if (m < l):
                m = l
                maxIndex = i - m
            l = 1
    if (m < l):
        m = l
        maxIndex = n - m
    for i in range(maxIndex, (m + maxIndex)):
        # print(arr[i], end=" ")
        onset_index.append(i)
        Onset_part.append(arr[i])


print(LogestIncSubArr(Distance1, len(Distance1)))
print(onset_index)
onset_start_index = onset_index[0]
onset_finish_index = onset_index[-1]

Distance2 = Distance1[onset_index[-1]:len(Distance1)]
offset_index = []
offset_part = []


def LogestDecSubArr(arr1, n1):
    k = 1
    u = 1
    maxIndex = 0
    for h in range(onset_index[-1] - 1, n1):
        if (arr1[h] <= arr1[h - 1]):
            u = u + 1
        else:
            if (k < u):
                k = u
                maxIndex = h - k
            u = 1
    if (k < u):
        k = u
        maxIndex = n1 - k
    for h in range(maxIndex, (k + maxIndex)):
        offset_index.append(h)
        offset_part.append(arr1[h])


LogestDecSubArr(Distance1, len(Distance1))
print(onset_index[-1])
print(offset_index[0])
Apex_part = Distance1[onset_index[-1]: offset_index[0]]
Apex_part = np.array(Apex_part)
print("%%%%%%%%%%%%%555" , Apex_part)
offset_start_index = offset_index[0]
offset_finish_index = offset_index[-1]
print("____________________")
plt.plot(Distance1)

plt.ylabel('Displacement')
plt.xlabel('Frames')
plt.axvline(x=onset_index[0], color='r')
plt.axvline(x=onset_index[-1], color='r')
plt.axvline(x=offset_index[0], color='y')
plt.axvline(x=offset_index[-1], color='y')
plt.show()
# print(Distance1)
# print(Onset_part)
# print(Apex_part)
# print(offset_part)

Increase = []
Apex = []
Decrease = []

for index in range(0, len(Apex_part)):
    if Apex_part[index] > Apex_part[index - 1]:
        Increase.append(index)
    elif Apex_part[index] < Apex_part[index - 1]:
        Decrease.append(index)
    else:
        Apex.append(index)
IncreasePart = []
DecreasePart = []
ApexPart = []
for i in range(1, len(Apex_part)):
    IncreasePart = Apex_part[Increase]
    DecreasePart = Apex_part[Decrease]
    ApexPart = Apex_part[Apex]

n = len(Apex_part)
x = np.arange(n + 1)  # resampledTime
y = Apex_part

# set up colors
c = []
col = []
for j in range(1, len(Apex_part)):
    if j in Increase:
        c.append('red')
    else:
        c.append('blue')

# convert time series to line segments
lines = [((x0, y0), (x1, y1)) for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:])]
colored_lines = LineCollection(lines, colors=c, linewidths=(2,))

# plot data
fig, ax = plt.subplots(1)
ax.add_collection(colored_lines)
ax.autoscale_view()
plt.show()
print(Onset_part, "__________________________________________")
# Features Onset Part
###__________________________________#
fps = cam.get(cv2.CAP_PROP_FPS)
print('fps', fps)
FS = len(Onset_part)
FSP = len(Onset_part)
print('FS', FS)
print("FSP", FSP)
DurationP = FSP / fps
print('DurationP:', DurationP)
DurationRatioP = FSP / FS
print('DurationRatioP:', DurationRatioP)
MaximumAmplitude = max(Onset_part)
print('MaximumAmplitude:', MaximumAmplitude)
MeanAmplitudeP = np.nansum(Onset_part) / FSP
print('MeanAmplitudeP:', MeanAmplitudeP)
STDofAmplitude = np.std(Onset_part)
print('STDofAmplitude:', STDofAmplitude)
TotalAmplitudeP = np.nansum(Onset_part)
print('TotalAmplitudeP:', TotalAmplitudeP)
NetAmplitude = TotalAmplitudeP - 0
print('NetAmplitude:', NetAmplitude)
AmplitudeRatioP = TotalAmplitudeP / (TotalAmplitudeP + 0)
print('AmplitudeRatioP:', AmplitudeRatioP)

if len(Onset_part) >= 2:
    OnsetSeries = pd.Series(Onset_part)
    SpeedP = sym.diff(OnsetSeries)
    MaximumSpeedP = np.nanmax(SpeedP)
    print(SpeedP)
    print('MaximumSpeedP:', MaximumSpeedP)
    MeanSpeedP = np.nansum(SpeedP)/ len(SpeedP)
    print('MeanSpeedP:', MeanSpeedP)
    if len(SpeedP) != 0:
        AccelerationP = sym.diff(SpeedP)
        MaximumAccelerationP = np.nanmax(AccelerationP)
        print(AccelerationP)
        MeanAccelerationP = np.nansum(AccelerationP) / len(AccelerationP)
        print('MaximumAccelerationP:', MaximumAccelerationP)
        print('MeanAccelerationP:', MeanAccelerationP)
    else:
        AccelerationP = 0
        MaximumAccelerationP = 0
        MeanAccelerationP = 0
        print('MaximumAccelerationP:', MaximumAccelerationP)
        print('MeanAccelerationP:', MeanAccelerationP)

else:
    SpeedP = 0
    MaximumSpeedP = 0
    MeanSpeedP = 0
    print('MeanSpeedP:', MeanSpeedP)
    print('MaximumSpeedP:', 0)
    AccelerationP = 0
    MaximumAccelerationP = 0
    MeanAccelerationP = 0
    print('MaximumAccelerationP:', MaximumAccelerationP)
    print('MeanAccelerationP:', MeanAccelerationP)




NetAmpDurationRatio = (sum(Onset_part) - 0) * fps / FS
print('NetAmpDurationRatio:', NetAmpDurationRatio)
DistanceL = df['D_lipL']
DistanceR = df['D_lipR']
DistanceL = np.array(DistanceL)
DistanceR = np.array(DistanceR)
LeftRightAmpDifference = np.absolute(np.nansum(DistanceL) - np.nansum(DistanceR)) / FS

print("LeftRightAmpDifference:", LeftRightAmpDifference)
workbook = xlsxwriter.Workbook(
    '25FeaturesOnset10.xlsx')  # name of the excel sheet would be saved in the save folder of the code
worksheet = workbook.add_worksheet()

worksheet.write(0, 0, 'DurationP')
worksheet.write(1, 0, 'DurationRatioP')
worksheet.write(2, 0, 'MaximumAmplitude')
worksheet.write(3, 0, 'MeanAmplitudeP')
worksheet.write(4, 0, 'STDofAmplitude')
worksheet.write(5, 0, 'TotalAmplitudeP')
worksheet.write(6, 0, 'NetAmplitude')
worksheet.write(7, 0, 'AmplitudeRatioP')
worksheet.write(8, 0, 'MaximumSpeedP')
worksheet.write(9, 0, 'MeanSpeedP')
worksheet.write(10, 0, 'MaximumAccelerationP')
worksheet.write(11, 0, 'MeanAccelerationP')
worksheet.write(12, 0, 'NetAmpDurationRatio')
worksheet.write(13, 0, 'LeftRightAmpDifference')

worksheet.write(0, 1, DurationP)
worksheet.write(1, 1, DurationRatioP)
worksheet.write(2, 1, MaximumAmplitude)
worksheet.write(3, 1, MeanAmplitudeP)
worksheet.write(4, 1, STDofAmplitude)
worksheet.write(5, 1, TotalAmplitudeP)
worksheet.write(6, 1, NetAmplitude)
worksheet.write(7, 1, AmplitudeRatioP)
worksheet.write(8, 1, MaximumSpeedP)
worksheet.write(9, 1, MeanSpeedP)
worksheet.write(10, 1, MaximumAccelerationP)
worksheet.write(11, 1, MeanAccelerationP)
worksheet.write(12, 1, NetAmpDurationRatio)
worksheet.write(13, 1, LeftRightAmpDifference)
workbook.close()
# -----------------------------------------------------------------------------------------
