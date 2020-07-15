# 2割をテストデータに移行
import shutil
import random
import glob
import os
import tkinter
from tkinter import filedialog
os.makedirs("../test", exist_ok=True)

# フォルダ指定
def dirdialog_clicked():
    root = tkinter.Tk()
    root.withdraw()
    iDir = os.path.abspath(os.path.dirname(__file__))
    iDirPath = filedialog.askdirectory(initialdir = iDir)
    root.destroy()
    return iDirPath


if __name__ == '__main__':
    # 画像フォルダ指定
    image_dir = dirdialog_clicked()
    img_lists = os.listdir(path=image_dir)
    for lists in range(len(img_lists)):
        in_dir = image_dir +"/"+img_lists[lists]+"/*"
        in_jpg=glob.glob(in_dir)
        img_file_name_list=os.listdir(image_dir +"/"+img_lists[lists]+"/")
        #img_file_name_listをシャッフル、そのうち2割をtest_imageディテクトリに入れる
        random.shuffle(in_jpg)
        os.makedirs('./test/' + os.path.basename(img_lists[lists]), exist_ok=True)
        for t in range(len(in_jpg)//5):
            shutil.move(str(in_jpg[t]), "./test/"+ os.path.basename(img_lists[lists]))