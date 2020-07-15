import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter
from tkinter import filedialog

# フォルダ指定
def dirdialog_clicked():
    root = tkinter.Tk()
    root.withdraw()
    iDir = os.path.abspath(os.path.dirname(__file__))
    iDirPath = filedialog.askdirectory(initialdir = iDir)
    root.destroy()
    return iDirPath


image_dir = dirdialog_clicked()
name = os.listdir(path=image_dir)

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


# 教師データのラベル付け
X_train = []
Y_train = []
for i in range(len(name)):
    img_file_name_list=os.listdir("./train/"+name[i])
    print(len(img_file_name_list))
    for j in range(0,len(img_file_name_list)-1):
        n=os.path.join("./train/"+name[i]+"/",img_file_name_list[j])
        img = imread(n)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        X_train.append(img)
        Y_train.append(i)

# テストデータのラベル付け
X_test = [] # 画像データ読み込み
Y_test = [] # ラベル（名前）
for i in range(len(name)):
    img_file_name_list=os.listdir("./test/"+name[i])
    print(len(img_file_name_list))
    for j in range(0,len(img_file_name_list)-1):
        n=os.path.join("./test/"+name[i]+"/",img_file_name_list[j])
        img = imread(n)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        X_test.append(img)
        # ラベルは整数値
        Y_test.append(i)
X_train=np.array(X_train)
X_test=np.array(X_test)

from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

# モデルの定義
model = Sequential()
model.add(Conv2D(input_shape=(64, 64, 3), filters=32,kernel_size=(3, 3),
                 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("sigmoid"))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(len(name))) # 人数決めるところ
model.add(Activation('softmax'))

# コンパイル
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 学習
history = model.fit(X_train, y_train, batch_size=32,
                    epochs=50, verbose=1, validation_data=(X_test, y_test))

# 汎化制度の評価・表示
score = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

#acc, val_accのプロット
plt.plot(history.history["accuracy"], label="accuracy", ls="-", marker="o")
print("1")
print(history.history)
plt.plot(history.history["val_accuracy"], label="val_accuracy", ls="-", marker="x")
print(history.history)
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

#モデルを保存
model.save("my_model.h5")