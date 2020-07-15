import requests
from bs4 import BeautifulSoup
from pathlib import Path
import urllib
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def main(search_word,maxcount):
    # 検索URL準備
    load_url="https://search.yahoo.co.jp/image/search?p="+search_word+"&ei=UTF-8&b="


    # 保存用フォルダを作る
    data = Path("./data")
    data.mkdir(exist_ok=True)
    out_folder=Path("data/"+search_word)
    out_folder.mkdir(exist_ok=True)

    # すべてのhtmlタグを検索し、リンクを取得する
    count=1

    for i in range(1,maxcount,20):

        # ページ切り替え
        html = requests.get(load_url+str(i))
        soup = BeautifulSoup(html.content, "html.parser")

        for element in soup.find_all("img"):
            src=element.get("src")

            # 絶対URLから画像を取得する
            image_url=urllib.parse.urljoin(load_url,src)
            imgdata=requests.get(image_url)

            # URLから最後のファイル名を取り出して保存先のファイル名とつなげる
            filename=str(count)+".png"
            out_path=out_folder.joinpath(filename)

            # 画像データをファイルに書き出す
            with open(out_path,mode="wb") as f:
                f.write(imgdata.content)

            # 枚数が十分だったらやめる
            if maxcount<=count:
                print(search_word+"完了")
                break
            else:
                count+=1


            # 0.2秒待つ
            time.sleep(0.2)
        print(str(count) + "枚取得")

if __name__ == '__main__':
    member = []
    num = int(input("人数"))
    for i in range(num):
        member.append(input("検索ワード"))
    b = int(input("取得枚数"))
    for i in range(len(member)):
        main(member[i], b)