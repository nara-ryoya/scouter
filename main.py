from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import random
import threading
import queue


# グローバル領域で各種変数を定義
height = 0
width = 0
cap = cv2.VideoCapture(0)
if (not cap.isOpened()):
    print("cannot open the camera")
    exit()

cascadeFile = 'haarcascade_frontalface_alt2.xml'
cascade = cv2.CascadeClassifier(cascadeFile)

scale = 1

mh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(mh / scale)
width = int(mw / scale)
mask = np.full((height, width, 3), (70, 110, 0), dtype=np.uint8)

captured_image = queue.Queue()
processed_image = queue.Queue()


#戦闘力（乱数）
fight = random.randint(5, 530000)

loop_flg = True

def main():

    while loop_flg:
        capture()
        detect_person_and_draw()
        k = cv2.waitKey(10)
        #Escキー or'q' を押されたら終了
        if k == 27 or k == ord('q'):
            break
    cap.release()

def capture():
    ret, frame = cap.read()
    if not ret:
        global loop_flg
        loop_flg = False
    captured_image.put(frame)

def detect_person_and_draw():
    if captured_image:
        frame = captured_image.get()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        faceRect = cascade.detectMultiScale(gray)
        frame = cv2.addWeighted(frame, 0.8, mask, 0.9, 0)
        if len(faceRect) > 0:
            for fx, fy, fw, fh in faceRect:
                textColor = (0, 244, 243)
                fontSize = 64
                font = ImageFont.FreeTypeFont('DSEG7Modern-Light.ttf', fontSize)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                x = fx + (fw / 2) - (font.getsize('FIGHT')[0] / 2)
                y = fy - (fontSize * 1.5)
                draw.text((x, y - 30), 'FIGHT', font = font, fill = textColor)
                x = fx + (fw / 2) - (font.getsize(str(fight))[0] / 2)
                y = fy - (fontSize * 0.5)
                draw.text((x, y), str(fight), font = font, fill = textColor)
                frame = np.array(img_pil)

                #真ん中三角
                p1 = (int(fx + (fw / 2)), int(fy + (fh * 1.0)))
                p2 = (int(fx + (fw / 2) - 40), int(fy + (fh * 1.0) + 60))
                p3 = (int(fx + (fw / 2) + 40), int(fy + (fh * 1.0) + 60))
                triangle = np.array([p1, p2, p3])
                cv2.drawContours(frame, [triangle], 0, textColor, -1)

                #左三角
                p1 = (int(fx - 35)), int(fy + (fh * 0.5) + 30)
                p2 = (int(fx - 95), int(fy + (fh * 0.5)))
                p3 = (int(fx - 95), int(fy + (fh * 0.5) + 60))
                triangle = np.array([p1, p2, p3])
                cv2.drawContours(frame, [triangle], 0, textColor, -1)

                #右三角
                p1 = (int(fx + fw + 35)), int(fy + (fh * 0.5) + 30)
                p2 = (int(fx + fw + 95), int(fy + (fh * 0.5)))
                p3 = (int(fx + fw + 95), int(fy + (fh * 0.5) + 60))
                triangle = np.array([p1, p2, p3])
                cv2.drawContours(frame, [triangle], 0, textColor, -1)

        cv2.imshow("scouter", frame)

if __name__ == "__main__":
    main()