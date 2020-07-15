import glob
import os
import tkinter
from tkinter import filedialog
import cv2
import numpy as np
import sys

"""
dataディレクトリから画像を読み込んで顔を切り取ってfaceディレクトリに保存.
"""
out_dir = "./face"

# フォルダ指定
def dirdialog_clicked():
    root = tkinter.Tk()
    root.withdraw()
    iDir = os.path.abspath(os.path.dirname(__file__))
    iDirPath = filedialog.askdirectory(initialdir = iDir)
    root.destroy()
    return iDirPath


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
#cv2.imencode + np.ndarray.tofile に分解
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


if __name__ == "__main__":
    #各フォルダ指定
    image_dir = dirdialog_clicked()
    img_lists=os.listdir(path=image_dir)
    out_dir="."

    for lists in range(len(img_lists)):
        save_num = 0
        # 元画像を取り出して顔部分を正方形で囲み、64×64pにリサイズ、別のファイルにどんどん入れてく
        in_dir = image_dir +"/"+img_lists[lists] + "/*.png"
        print(in_dir)
        in_jpg = glob.glob(in_dir)
        print(in_jpg)
        os.makedirs("./face/"+ os.path.basename(img_lists[lists]), exist_ok=True)
        for num in in_jpg:
            image = imread(num)
            print(num)
            if image is None:
                print("Not open:", num)
                continue

            image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
            # 顔認識の実行
            face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))
            # 顔が１つ以上検出された時
            if len(face_list) > 0:
                for rect in face_list:
                    x, y, width, height = rect
                    image = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                    if image.shape[0] < 64:
                        continue
                    image = cv2.resize(image, (64, 64))
                    # 保存
                    fileName = "./face/"+ os.path.basename(img_lists[lists])+"/"+str(save_num)+".png"
                    save_num+=1
                    imwrite(str(fileName), image)
                    print(fileName)
                    print(img_lists[lists]+"/"+str(save_num)+".pngを保存しました.")
            # 顔が検出されなかった時
            else:
                print("no face")
                continue
            print(image.shape)
