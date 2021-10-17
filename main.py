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

res = np.zeros((height, width, 3), dtype=np.uint8)

captured_image = queue.Queue()
processed_image = queue.Queue()


#戦闘力（乱数）
fight = random.randint(5, 530000)

loop_flg = True

class FaceThread(threading.Thread):
	def __init__(self, frame):
            super(FaceThread, self).__init__()
            self._frame = frame

	def run(self):
            frame = self._frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = cascade.detectMultiScale(
                gray,
                minSize=(height // 10, width // 10)
            )

            tmp = cv2.addWeighted(frame, 0.8, mask, 0.9, 0).copy()

            if len(faces) == 1:
                # fx, fy, fw, fh = faces[0][0], faces[0][1], faces[0][2], faces[0][3]
                for fx, fy, fw, fh in faces:
                    textColor = (0, 244, 243)
                    fontSize = 64

                    word = "status"
                    font = ImageFont.FreeTypeFont('DSEG7Modern-Light.ttf', fontSize)
                    img_pil = Image.fromarray(tmp)
                    draw = ImageDraw.Draw(img_pil)
                    x = fx + (fw / 2) - (font.getsize(word)[0] / 2)
                    y = fy - (fontSize * 1.5)
                    draw.text((x, y - 30), word, font = font, fill = textColor)
                    x = fx + (fw / 2) - (font.getsize(str(fight))[0] / 2)
                    y = fy - (fontSize * 0.5)
                    draw.text((x, y), str(fight), font = font, fill = textColor)
                    tmp = np.array(img_pil)

                    #真ん中三角
                    p1 = (int(fx + (fw / 2)), int(fy + (fh * 1.0)))
                    p2 = (int(fx + (fw / 2) - 40), int(fy + (fh * 1.0) + 60))
                    p3 = (int(fx + (fw / 2) + 40), int(fy + (fh * 1.0) + 60))
                    triangle = np.array([p1, p2, p3])
                    cv2.drawContours(tmp, [triangle], 0, textColor, -1)

                    #左三角
                    p1 = (int(fx - 35)), int(fy + (fh * 0.5) + 30)
                    p2 = (int(fx - 95), int(fy + (fh * 0.5)))
                    p3 = (int(fx - 95), int(fy + (fh * 0.5) + 60))
                    triangle = np.array([p1, p2, p3])
                    cv2.drawContours(tmp, [triangle], 0, textColor, -1)

                    #右三角
                    p1 = (int(fx + fw + 35)), int(fy + (fh * 0.5) + 30)
                    p2 = (int(fx + fw + 95), int(fy + (fh * 0.5)))
                    p3 = (int(fx + fw + 95), int(fy + (fh * 0.5) + 60))
                    triangle = np.array([p1, p2, p3])
                    cv2.drawContours(tmp, [triangle], 0, textColor, -1)

            global res
            res = tmp

def main():

    while loop_flg:
        global captured_image, processed_image
        ret, frame = cap.read()
        if not ret:
            continue
        if(threading.activeCount() == 1):
            th = FaceThread(frame)
            th.start()

        cv2.imshow("scouter", res)
        
        # if (captured_image.qsize() - processed_image.qsize() >= 10):
        #     captured_image = queue.Queue()
        #     processed_image = queue.Queue()

        k = cv2.waitKey(10)
        if k == 27 or k == ord('q'):
            break
    cap.release()

if __name__ == "__main__":
    main()