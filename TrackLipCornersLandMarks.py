import mediapipe as mp
import cv2
import xlsxwriter

workbook = xlsxwriter.Workbook('.\smiles\pair3id5N.xlsx') # name of the excel sheet would be saved in the save folder of the code
worksheet = workbook.add_worksheet()
row = 1
column = 0

cap = cv2.VideoCapture(".\smiles\pair3id5N.mp4")  # //just put the video in the same folder as the python code
facmesh = mp.solutions.face_mesh
face = facmesh.FaceMesh(static_image_mode=True, min_tracking_confidence=0.6, min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils
time = 0

while True:

    ret, frm = cap.read()
    if frm is None:
        workbook.close()


    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    op = face.process(rgb)
    if op.multi_face_landmarks:
        for i in op.multi_face_landmarks:
            print(time)
            print(i.landmark[291].x * 1080)
            print(i.landmark[291].y * 1440)
            draw.draw_landmarks(frm, i, landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 255), circle_radius=1))

            # Write code into excel sheet
            worksheet.write(0, 0, 'Time')
            worksheet.write(0, 1, 'X point Right')
            worksheet.write(0, 2, 'Y point Right')
            worksheet.write(0, 3, 'X point Left')
            worksheet.write(0, 4, 'Y point Left')
            worksheet.write(0, 5, 'X point Center 1')
            worksheet.write(0, 6, 'Y point Center 1')
            worksheet.write(0, 7, 'X point Center 2')
            worksheet.write(0, 8, 'Y point Center 2')
            worksheet.write(0, 9, 'X point Nose tip')
            worksheet.write(0, 10, 'Y point Nose tip')

            worksheet.write(row, 0, time)
            worksheet.write(row, 1, i.landmark[291].x * 1080)
            worksheet.write(row, 2, i.landmark[291].y * 1440)
            worksheet.write(row, 3, i.landmark[61].x * 1080)
            worksheet.write(row, 4, i.landmark[61].y * 1440)
            worksheet.write(row, 5, i.landmark[13].x * 1080)
            worksheet.write(row, 6, i.landmark[13].y * 1440)
            worksheet.write(row, 7, i.landmark[14].x * 1080)
            worksheet.write(row, 8, i.landmark[14].y * 1440)
            worksheet.write(row, 9, i.landmark[1].x * 1080)
            worksheet.write(row, 10, i.landmark[1].y * 1440)

            time = time + 1
            row = row + 1




    cv2.imshow("window", frm)



    if cv2.waitKey(1) == 27:

        cap.release()
        cv2.destroyAllWindows()
        break




