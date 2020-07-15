import image_collect2
import face_detect

if __name__ == '__main__':
    a = []
    num = int(input("人数"))
    for i in range(num):
        a.append(input("検索ワード"))
    b = int(input("取得枚数"))
    for i in range(len(a)):
        image_collect2.main(a[i], b)
