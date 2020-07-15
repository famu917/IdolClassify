import numpy as np
import cv2
from keras.models import load_model
import tkinter
from tkinter import filedialog
import os
from PIL import ImageFont, ImageDraw, Image


# フォルダ指定
def dirdialog_clicked():
    root = tkinter.Tk()
    root.withdraw()
    iDir = os.path.abspath(os.path.dirname(__file__))
    iDirPath = filedialog.askdirectory(initialdir=iDir)
    root.destroy()
    return iDirPath


def select_image():
    root = tkinter.Tk()
    root.withdraw()
    fTyp = [("", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    filePath = filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    root.destroy()
    return filePath


# パスに日本語が含まれる場合の対策
# np.profileとcv2.imdecodeに分解した
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


# cv2.imencode + np.ndarray.tofile に分解
def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def puttext(cv_image, text, point, font_path, font_size, color=(0, 0, 0)):
    font = ImageFont.truetype(font_path, font_size)

    cv_rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_rgb_image)

    draw = ImageDraw.Draw(pil_image)
    #draw.text(point, text, fill=color, font=font)
    draw.text(point, text, fill=color, font=font)

    cv_rgb_result_image = np.asarray(pil_image)
    cv_bgr_result_image = cv2.cvtColor(cv_rgb_result_image, cv2.COLOR_RGB2BGR)

    return cv_bgr_result_image


def detect_face(image):
    # opencvを使って顔抽出
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
    # 顔認識の実行
    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))
    # 顔が１つ以上検出された時
    if len(face_list) > 0:
        for rect in face_list:
            x, y, width, height = rect
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), (255, 0, 0), thickness=3)
            img = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            if image.shape[0] < 64:
                print("too small")
                continue
            img = cv2.resize(image, (64, 64))
            img = np.expand_dims(img, axis=0)
            name = detect_who(img)
            fontpath = 'C:\Windows\Fonts\meiryo.ttc'
            image = puttext(image, name, (x, y + height + 20), fontpath, 24, (255, 0, 0))
            #cv2.putText(image, name, (x, y + height + 20), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
    # 顔が検出されなかった時
    else:
        print("no face")
    return image


def detect_who(img):
    # image_dir = dirdialog_clicked()
    # name_list = os.listdir(path=image_dir)
    name_list = os.listdir(path="./face")
    # 予測
    name = ""
    print(model.predict(img))
    print(name_list)
    nameNumLabel = np.argmax(model.predict(img))
    print(nameNumLabel)
    for num in range(len(name_list)):
        if nameNumLabel == num:
            name = name_list[num]
            print(name,name_list[num])
    return name


if __name__ == '__main__':
    model = load_model('./my_model.h5')
    image = imread(select_image())
    if image is None:
        print("Not open:")
    whoImage = detect_face(image)

    cv2.imshow("result", whoImage)
    cv2.waitKey(0)
