import os
import cv2
import glob
from scipy import ndimage
import tkinter
from tkinter import filedialog
import numpy as np
"""
faceディレクトリから画像を読み込んで回転、ぼかし、閾値処理をしてtrainディレクトリに保存する.
"""

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



if __name__ == '__main__':
    os.makedirs("./train", exist_ok=True)

    # 画像フォルダ指定
    image_dir = dirdialog_clicked()
    img_lists = os.listdir(path=image_dir)

    for lists in range(len(img_lists)):
        in_dir = image_dir +"/"+img_lists[lists] + "/*"
        out_dir = "./train/" + img_lists[lists]+"/"
        os.makedirs(out_dir, exist_ok=True)
        in_jpg=glob.glob(in_dir)
        save_num=0
        for i in in_jpg:
            img = imread(i)
            # 左右反転

            img_flip = cv2.flip(img, 1)
            fileName = os.path.join(out_dir, str(save_num) + "_flip.jpg")
            imwrite(str(fileName), img_flip)
            # 回転
            for ang in [-10,0,10]:
                img_rot = ndimage.rotate(img,ang)
                img_rot = cv2.resize(img_rot,(64,64))
                fileName=os.path.join(out_dir,str(save_num)+"_"+str(ang)+".jpg")
                imwrite(str(fileName),img_rot)
                # 閾値
                img_thr = cv2.threshold(img_rot, 100, 255, cv2.THRESH_TOZERO)[1]
                fileName=os.path.join(out_dir,str(save_num)+"_"+str(ang)+"thr.jpg")
                imwrite(str(fileName),img_thr)
                # ぼかし
                img_filter = cv2.GaussianBlur(img_rot, (5, 5), 0)
                fileName=os.path.join(out_dir,str(save_num)+"_"+str(ang)+"filter.jpg")
                imwrite(str(fileName),img_filter)
            save_num += 1