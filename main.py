import cv2
import dlib
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import random
import threading
from imutils import face_utils
import time
import signal

import math



# グローバル領域で各種変数を定義
height = 0
width = 0
cap = cv2.VideoCapture(0)
if (not cap.isOpened()):
    print("cannot open the camera")
    exit()

cascadeFile = 'haarcascade_frontalface_alt2.xml'
cascade = cv2.CascadeClassifier(cascadeFile)


face_part_data = "shape_predictor_68_face_landmarks.dat"
face_parts_detector = dlib.shape_predictor(face_part_data)


scale = 1

mh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(mh / scale)
width = int(mw / scale)
mask = np.full((height, width, 3), (70, 110, 0), dtype=np.uint8)

res = np.zeros((height, width, 3), dtype=np.uint8)
catched_frame = res

detector = dlib.get_frontal_face_detector()

loop_flg = True

rec_mode = False

arg = 0


class FaceThread(threading.Thread):
	def __init__(self, frame):
            super(FaceThread, self).__init__()
            self._frame = frame
            self.fight = 0

	def run(self):
            frame = self._frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = cascade.detectMultiScale(
                gray,
                minSize=(height // 10, width // 10)
            )

            tmp = cv2.addWeighted(frame, 0.8, mask, 0.9, 0).copy()

            global rec_mode

        

            if len(faces) > 0:
                for fx, fy, fw, fh in faces:
                    center_x = fx + fw // 2
                    center_y = fy + fh // 2
                    center = (center_x, center_y)
                    radius =  int((((fw//2) * (fw//2) + (fh//2) * (fh//2)) ** 0.5) * 0.7)
                    # cv2.rectangle(tmp, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
                    textColor = (0, 244, 243)

                    #顔のランドマークの取得
        
                    face_range = dlib.rectangle(fx, fy, fx + fw, fy + fh)
                    face_landmarks = face_parts_detector(gray, face_range)
                    face_landmarks = face_utils.shape_to_np(face_landmarks)


                    if rec_mode:

                        #上の三角
                        p1 = (center_x, fy)
                        p2 = (center_x - fw//10, fy - fh//10)
                        p3 = (center_x + fw//10, fy - fh//10)
                        triangle = np.array([p1, p2, p3])
                        cv2.drawContours(tmp, [triangle], 0, textColor, -1)

                        #下の三角
                        p1 = (center_x, fy + fh)
                        p2 = (center_x - fw//10, fy + fh + fh//10)
                        p3 = (center_x + fw//10, fy + fh + fh//10)
                        triangle = np.array([p1, p2, p3])
                        cv2.drawContours(tmp, [triangle], 0, textColor, -1)

                        #左の三角
                        p1 = (int(fx - fw//12)), int(fy + (fh * 0.5) + fh//12)
                        p2 = (int(fx - fw//5), int(fy + (fh * 0.5)))
                        p3 = (int(fx - fw//5), int(fy + (fh * 0.5) + fh//6))
                        triangle = np.array([p1, p2, p3])
                        cv2.drawContours(tmp, [triangle], 0, textColor, -1)

                        #右の三角
                        p1 = (int(fx + fw + fw//12)), int(fy + (fh * 0.5) + fh//12)
                        p2 = (int(fx + fw + fw//5), int(fy + (fh * 0.5)))
                        p3 = (int(fx + fw + fw//5), int(fy + (fh * 0.5) + fh//6))
                        triangle = np.array([p1, p2, p3])
                        cv2.drawContours(tmp, [triangle], 0, textColor, -1)

                        for i, ((x, y)) in enumerate(face_landmarks[:]):
                            cv2.circle(tmp, (x, y), 1, (0, 255, 0), -1)
                            cv2.putText(tmp, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                        
                        if len(face_landmarks) != 0:
                            # cv2.rectangle(tmp, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
                            global arg
                            cv2.ellipse(tmp, (center_x, center_y), (radius, radius), angle=0, startAngle=arg, endAngle=40+arg, color=(255, 255, 255), thickness=7)
                            # print(arg)
                            arg += 5

                            strongness = 53000


                            fontSize = fh//6
                            font = ImageFont.FreeTypeFont('DSEG7Modern-Light.ttf', fontSize)
                            word = "status"
                            img_pil = Image.fromarray(tmp)
                            draw = ImageDraw.Draw(img_pil)
                            x = fx + (fw / 2) - (font.getsize(word)[0] / 2)
                            y = fy - (fontSize * 1.5)

                            draw.text((x, y - 30), word, font = font, fill = textColor)
                            x = fx + (fw / 2) - (font.getsize(str(strongness))[0] / 2)
                            y = fy - (fontSize * 0.5)
                            draw.text((x, y), str(strongness), font = font, fill = textColor)
                            tmp = np.array(img_pil)
                        
                        else:
                            arg = 0
                            # global rec_mode
                            rec_mode = False
                            #360度回らなかったら矯正終了する


            global res,catched_frame
            res = tmp
            catched_frame = tmp


def main():

    global rec_mode, catched_frame

    while loop_flg:
        global arg
        if arg == 360:
            rec_mode = False
            arg = 0
        ret, frame = cap.read()
        if not ret:
            continue
        if(threading.activeCount() == 1):
            th = FaceThread(frame)
            th.start()

        k = cv2.waitKey(10)
        if k == 27 or k == ord('q'):
            break

        if k == ord('r'):
            rec_mode = not rec_mode
            if (rec_mode):
                arg = 0
        
        if (rec_mode):
            cv2.imshow("scouter", catched_frame)
        else:
            cv2.imshow("scouter", res); 

    cap.release()

if __name__ == "__main__":
    main()