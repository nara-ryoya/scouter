import cv2
import dlib
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import random
import threading
from imutils import face_utils
import time
import signal
import random
import math
import subprocess

import simpleaudio

#諸々のファイル
cascadeFile = 'haarcascade_frontalface_alt2.xml'
face_part_data = "shape_predictor_68_face_landmarks.dat"
font_path = "DSEG7Modern-Light.ttf"
roop_sound_file = "roop_music.wav"
end_sound_file = "end_music.wav"

# グローバル領域で各種変数を定義

#カメラのキャプチャ
cap = cv2.VideoCapture(0)
if (not cap.isOpened()):
    print("cannot open the camera")
    exit()

#画面の大きさの定義
scale = 1
mh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(mh / scale)
width = int(mw / scale)

#緑色にマスクするための画面
mask = np.full((height, width, 3), (70, 110, 0), dtype=np.uint8)
#赤色にマスクするための画面
danger_mask = np.full((height, width, 3), (70, 110, 0), dtype=np.uint8)

#cascade分類器(顔の判定)
cascade = cv2.CascadeClassifier(cascadeFile)
#顔のランドマークの判定器
face_parts_detector = dlib.shape_predictor(face_part_data)

#マスク・検出・描画などの処理した後の画面
res = np.zeros((height, width, 3), dtype=np.uint8)
catched_frame = res

loop_flg = True #loopを続けるかどうかを制御するためのbool変数
rec_mode = False #rec_mode=Trueのときは描画を行う
arg = 0 #arg < 360のとき顔認識を行い、arg<540のとき描画を行う
is_roop_sound_played = False #計測中の効果音を流すかどうかを制御するためのbool変数

strongness_list = [0] #目の開き具合、口の下がり具合で判定する

#効果音のオブジェクト
roop_sound_obj = simpleaudio.WaveObject.from_wave_file(roop_sound_file)
end_sound_obj = simpleaudio.WaveObject.from_wave_file(end_sound_file)

#計測中の効果音の再生オブジェクト、すべての関数から止められるようにグローバル領域で定義
roop_sound_play_obj = roop_sound_obj.play()
roop_sound_play_obj.stop()

def roop_sound():
    global roop_sound_play_obj
    is_roop_sound_played = True
    roop_sound_play_obj = roop_sound_obj.play()
    is_roop_sound_played = False

def stop_roop_sound():
    global roop_sound_play_obj
    if roop_sound_play_obj.is_playing():
        roop_sound_play_obj.stop()


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class FaceThread(threading.Thread):
	def __init__(self, frame):
            super(FaceThread, self).__init__()
            self._frame = frame
            self.fight = 0
            self.fontSize = 40
            self.strongness = -1
            self.speed = 10
            self.alpha = 10000.0
            self.beta = 1.0

	def run(self):
            frame = self._frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            tmp = cv2.addWeighted(frame, 0.8, mask, 0.9, 0).copy()

            global rec_mode, arg, speed

            textColor = (0, 230, 230)

            font = ImageFont.FreeTypeFont(font_path, self.fontSize)

            
            if arg < 360:
                #角度が360度未満だったら計測しよう
                #検出処理 & 描画処理
                #途中で検出できなかったら終わらせる
                faces = cascade.detectMultiScale(
                    gray,
                    minSize=(height // 10, width // 10)
                )

                if rec_mode:              #計測中はグリッド線を引く
                    tmp[:height:height // 20, :, 1:] = 240
                    tmp[:, :width:width // 20, 1:] = 240


                if len(faces) == 1:
                    for fx, fy, fw, fh in faces:

                        center_x = fx + fw // 2
                        center_y = fy + fh // 2
                        center = (center_x, center_y)
                        radius =  int((((fw//2) * (fw//2) + (fh//2) * (fh//2)) ** 0.5) * 0.7)

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

                            # for i, ((x, y)) in enumerate(face_landmarks[:]):
                            #     cv2.circle(tmp, (x, y), 1, (0, 255, 0), -1)
                            #     cv2.putText(tmp, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                            
                            if len(face_landmarks) != 0:
                                cv2.ellipse(tmp, (center_x, center_y), (radius, radius), angle=0, startAngle=arg, endAngle=40+arg, color=textColor, thickness=7)
                                eye_right_w = distance(face_landmarks[36], face_landmarks[39])
                                eye_right_h =  (distance(face_landmarks[37], face_landmarks[41]) + distance(face_landmarks[38], face_landmarks[40])) / 2
                                eye_left_w = distance(face_landmarks[42], face_landmarks[44])
                                eye_left_h = (distance(face_landmarks[43], face_landmarks[47]) + distance(face_landmarks[44], face_landmarks[46])) // 2
                                eye_right_ratio = eye_right_h / eye_right_w
                                eye_left_ratio = eye_left_h / eye_left_w
                                eye_ratio = self.alpha * (eye_left_ratio + eye_right_ratio) / 2

                                mouth_w = distance(face_landmarks[48], face_landmarks[54])
                                mouth_h = (face_landmarks[51][1] + face_landmarks[57][1] - face_landmarks[48][1] - face_landmarks[54][1]) / 2
                                mouth_ratio = self.alpha * mouth_h / mouth_w

                                strongness_list.append(int(mouth_ratio + eye_ratio))
                                

                                word = "calculating...".upper()
                                self.strongness = random.randint(0, 999999)

                                img_pil = Image.fromarray(tmp)
                                draw = ImageDraw.Draw(img_pil)
                                x = width//2 - (font.getsize(word)[0] // 2)
                                y = height//2 - int(self.fontSize)
                                y = max(y, 0)
                                draw.text((x, y), word, font = font, fill = textColor)

                                x = width // 2 - (font.getsize(str(self.strongness))[0] // 2)
                                y +=  int(self.fontSize * 1.2)
                                draw.text((x, y), str(self.strongness), font = font, fill = textColor)
                                tmp = np.array(img_pil)

                                arg += self.speed
            
                            
            else:
                if arg - self.speed < 360:
                    end_sound_obj.play()
                    
                stop_roop_sound()
                if self.strongness == -1:
                    self.strongness = int(10 ** ( (sum(strongness_list) * 10/ (self.alpha * len(strongness_list)))))
                
                if self.strongness >= 530000:
                    tmp
                word = "complete!".upper()
                img_pil = Image.fromarray(tmp)
                draw = ImageDraw.Draw(img_pil)
                x = width//2 - (font.getsize(word)[0] // 2)
                y = height//2 - int(self.fontSize)
                y = max(y, 0)
                draw.text((x, y), word, font = font, fill = textColor)

                x = width//2 - (font.getsize(str(self.strongness))[0] // 2)
                y +=  int(self.fontSize * 1.2)
                draw.text((x, y), str(self.strongness), font = font, fill = textColor)
                tmp = np.array(img_pil)

                arg += self.speed // 5


            
            global res, catched_frame
            res = tmp
            catched_frame = tmp



def main():

    global rec_mode, catched_frame

    while loop_flg:
        rec_pre_mode = rec_mode
        global arg, strongness_list
        if arg >= 540:
            rec_mode = False
            arg = 0
            strongness_list = [0]
        ret, frame = cap.read()
        if not ret:
            continue
        if(threading.activeCount() == 1):
            th = FaceThread(frame)
            th.start()
        
        if rec_mode and threading.activeCount() == 2 and not roop_sound_play_obj.is_playing() and arg < 360:
            roop_sound_thread = threading.Thread(target=roop_sound)
            roop_sound_thread.setDaemon(True)
            roop_sound_thread.start()

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
            stop_roop_sound()
            cv2.imshow("scouter", res); 

    cap.release()

if __name__ == "__main__":
    main()