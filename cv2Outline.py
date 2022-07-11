import numpy as np
import cv2
import glob
from datetime import datetime
from time import sleep

RESIZE_SIZE = 4 # 画像を処理する時に使うサイズ(1/n)
CAMERA_SIZE = 2 # カメラ画像の縮小サイズ(1/n)

# カメラフラグでTrueにするとWebカメラの変換になる
CAMERA_FLG = False

IMG_FOLDER_PATH = "./img/*"     # 画像フォルダ
SAVE_FOLDER_PATH = "./result/"  # 出力保存フォルダ


# メイン関数
def main():
    print("--- start ---")

    # カメラの場合との場合分け
    if (CAMERA_FLG):
        # VideoCaptureのインスタンスを作成(引数でカメラを選択できる)
        cap = cv2.VideoCapture(0)
        changeCameraImage(cap)
        closeWindows(cap) # ウインドウを全て閉じる
    else:
        changeLoadImages(IMG_FOLDER_PATH, SAVE_FOLDER_PATH)

    print("--- end ---")


# カメラの映像を変換する関数
def changeCameraImage(cap):
    while True:
        ret, frame = cap.read() # 戻り値のframeがimg
        resultImg = changeImage(frame) # 画像変換

        # 結果をリサイズ
        fx = int(resultImg.shape[1]/CAMERA_SIZE)
        fy = int(resultImg.shape[0]/CAMERA_SIZE)
        resultImg = cv2.resize(resultImg, (fx, fy))

        # 文字を追加
        text = 'Exit is [Q] key'
        cv2.putText(resultImg, text, (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0), 3, cv2.LINE_AA)

        # 加工した画像を表示
        cv2.imshow('resultImg', resultImg)

        # キー入力を1ms待って、keyが「q」だったらbreak
        key = cv2.waitKey(1)&0xff
        if key == ord('q'):
            break


# 画像を読み込んで変換する関数
def changeLoadImages(imgFolderPath, saveFolderPath):
    images = glob.glob(imgFolderPath)

    for fname in images:
        frame = cv2.imread(fname) # 画像取得
        resultImg = changeImage(frame) # 画像変換

        # 画像保存
        saveImgByTime(saveFolderPath, resultImg)
        sleep(1)


# 画像を変換する関数
def changeImage(colorImg):
    gray = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY) # グレースケール
    gray = cv2.GaussianBlur(gray, (15, 15), 15) # ぼかす

    # 輪郭線処理
    img_diff = outine(gray) # 輪郭線抽出
    img_diff = cv2.GaussianBlur(img_diff, (11, 11), 8) # ぼかす
    ret, img_diff = cv2.threshold(img_diff, 170, 240, cv2.THRESH_BINARY) # 閾値で2値化

    # 影部分の処理
    gray = cv2.GaussianBlur(gray, (31, 31), 30) # ぼかす
    ret, gray = cv2.threshold(gray, 40, 220, cv2.THRESH_BINARY) # 閾値で2値化
    gray = lowContrast(gray) # コントラストを落とす
    
    # 輪郭線と影部分の画像を合成
    resultImg = cv2.bitwise_and(img_diff, gray)

    return resultImg


# 画像の輪郭線を抽出する関数
def outine(grayImg):
    # リサイズ
    fx = int(grayImg.shape[1]/RESIZE_SIZE)
    fy = int(grayImg.shape[0]/RESIZE_SIZE)

    grayChangeImg = cv2.resize(grayImg, (fx, fy))
    # 輪郭線抽出
    result = 255 - cv2.Canny(grayChangeImg, 110, 70)
    # リサイズして元に戻す
    result = cv2.resize(result, (grayImg.shape[1], grayImg.shape[0]))
    return result


# コントラストを落とす関数
def lowContrast(img):
    # ルックアップテーブルの生成
    min_table = 50
    max_table = 230
    diff_table = max_table - min_table
    look_up_table = np.arange(256, dtype = 'uint8' )
 
    for i in range(0, 255):
        look_up_table[i] = min_table + i * (diff_table) / 255
 
    # コントラストを低減
    result = cv2.LUT(img, look_up_table)
    return result


# 画像を時刻で保存する関数
def saveImgByTime(dirPath, img):
    # 時刻を取得
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = dirPath + date + ".png"
    cv2.imwrite(path, img) # ファイル保存


# キャプチャをリリースして、ウィンドウをすべて閉じる関数
def closeWindows(cap):
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()