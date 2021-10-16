from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import random

cap = cv2.VideoCapture(0)

scale = 1.0

#顔検出のカスケード分類器を生成※環境によりパスが変わります
cascadeFile = 'haarcascade_frontalface_alt2.xml'
cascade = cv2.CascadeClassifier(cascadeFile)

#スカウターっぽく背景を緑色にするためのマスク画像
mh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(mh / scale)
width = int(mw / scale)
mask = np.full((height, width, 3), (70, 110, 0), dtype=np.uint8)


#戦闘力（乱数）
fight = random.randint(5, 530000)

# print(cap.size())
# print(mask.shape)

def init():
    glutInitWindowPosition(0, 0)
    #glutInitWindowSizeは(w, h)の順番
    # glutInitWindowSize(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    glutInitWindowSize(width, height)
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
    glutCreateWindow("Scouter")

def draw():
    _, curFrame = cap.read()

    # gray = np.full((height, width, 1), 0, dtype=np.uint8)


    #顔検出
    # curFrame = cv2.resize(curFrame, (height, width), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(curFrame, cv2.COLOR_BGRA2GRAY)
    faceRect = cascade.detectMultiScale(gray)

    # 全体を緑色っぽく
    curFrame = cv2.addWeighted(curFrame, 0.8, mask , 0.9, 0)

    # print(len(faceRect))
    if len(faceRect) > 0:
        for fx, fy, fw, fh in faceRect:
            #顔にモザイク

            #戦闘力表示
            textColor = (0, 244, 243)
            fontSize = 64
            font = ImageFont.truetype('SFNSMonoItalic.ttf', fontSize)
            img_pil = Image.fromarray(curFrame)
            draw = ImageDraw.Draw(img_pil)
            x = fx + (fw / 2) - (font.getsize('FIGHT')[0] / 2)
            y = fy - (fontSize * 1.5) + 5
            draw.text((x, y), 'FIGHT', font = font, fill = textColor)
            x = fx + (fw / 2) - (font.getsize(str(fight))[0] / 2)
            y = fy - (fontSize * 0.5)
            draw.text((x, y), str(fight), font = font, fill = textColor)
            curFrame = np.array(img_pil)

            #真ん中三角
            p1 = (int(fx + (fw / 2)), int(fy + (fh * 1.0)))
            p2 = (int(fx + (fw / 2) - 40), int(fy + (fh * 1.0) + 60))
            p3 = (int(fx + (fw / 2) + 40), int(fy + (fh * 1.0) + 60))
            triangle = np.array([p1, p2, p3])
            cv2.drawContours(curFrame, [triangle], 0, textColor, -1)

            #左三角
            p1 = (int(fx - 35)), int(fy + (fh * 0.5) + 30)
            p2 = (int(fx - 95), int(fy + (fh * 0.5)))
            p3 = (int(fx - 95), int(fy + (fh * 0.5) + 60))
            triangle = np.array([p1, p2, p3])
            cv2.drawContours(curFrame, [triangle], 0, textColor, -1)

            #右三角
            p1 = (int(fx + fw + 35)), int(fy + (fh * 0.5) + 30)
            p2 = (int(fx + fw + 95), int(fy + (fh * 0.5)))
            p3 = (int(fx + fw + 95), int(fy + (fh * 0.5) + 60))
            triangle = np.array([p1, p2, p3])
            cv2.drawContours(curFrame, [triangle], 0, textColor, -1)

    #フレーム描画
    curFrame = cv2.cvtColor(curFrame, cv2.COLOR_BGR2RGB)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, curFrame.shape[1], curFrame.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, curFrame)

    glEnable(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    glBegin(GL_QUADS) 
    glTexCoord2d(0.0, 1.0)
    glVertex3d(-1.0, -1.0, 0.0)
    glTexCoord2d(1.0, 1.0)
    glVertex3d(1.0, -1.0, 0.0)
    glTexCoord2d(1.0, 0.0)
    glVertex3d(1.0,  1.0, 0.0)
    glTexCoord2d(0.0, 0.0)
    glVertex3d(-1.0, 1.0, 0.0)
    glEnd()

    glFlush()
    glutSwapBuffers()

def idle():
    glutPostRedisplay()

def keyboard(key, x, y):
    key = key.decode('utf-8')
    if key == 'q':
        exit()

if __name__ == "__main__":
    init()
    glutDisplayFunc(draw)
    glutKeyboardFunc(keyboard)
    glutIdleFunc(idle)
    glutMainLoop()

    cap.release()