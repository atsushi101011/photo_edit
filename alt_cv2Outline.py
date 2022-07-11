import numpy as np
import cv2
import glob
from datetime import datetime
from time import sleep

CAMERA_SIZE = 2 # カメラ画像の縮小サイズ(1/n)

BLUR_VALUE = 25000 # ブラーをかけるための定数
PIXEL_VALUE = 300000 # 拡大/縮小の基準となるピクセル

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

    # ピクセル数からぼかす値を計算
    allPixel = colorImg.shape[1] * colorImg.shape[0]
    bokashi = calcBlurValue(allPixel)
    gray = cv2.GaussianBlur(gray, (bokashi, bokashi), bokashi) # ぼかす

    # 輪郭線処理
    img_diff = outine(gray, allPixel) # 輪郭線抽出

    # 輪郭線用のぼかし計算
    bokashiOutline = bokashi
    if bokashi > 4:
        bokashiOutline = bokashi - 4

    img_diff = cv2.GaussianBlur(img_diff, (bokashiOutline, bokashiOutline), bokashiOutline) # ぼかす
    ret, img_diff = cv2.threshold(img_diff, 170, 240, cv2.THRESH_BINARY) # 閾値で2値化

    # 影部分の処理
    gray = cv2.GaussianBlur(gray, (bokashi, bokashi), bokashi) # ぼかす
    ret, gray = cv2.threshold(gray, 40, 220, cv2.THRESH_BINARY) # 閾値で2値化
    gray = lowContrast(gray) # コントラストを落とす
    
    # 輪郭線と影部分の画像を合成
    resultImg = cv2.bitwise_and(img_diff, gray)

    return resultImg


# ブラーをかける値を計算する関数
def calcBlurValue(allPixel):
    result = int(np.sqrt(allPixel/BLUR_VALUE))
    if (result%2 == 0):
        result = result + 1
    return result


# 画像の輪郭線を抽出する関数
def outine(grayImg, allPixel):
    # リサイズ
    z =  np.sqrt(PIXEL_VALUE / (grayImg.shape[1] * grayImg.shape[0]))
    if (z > 1):
        z = 1

    fx = int(grayImg.shape[1] * z)
    fy = int(grayImg.shape[0] * z)

    grayChangeImg = cv2.resize(grayImg, (fx, fy))
    # 輪郭線抽出
    result = 255 - cv2.Canny(grayChangeImg, 100, 50)
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
    print("saved: " + date + ".png")


# キャプチャをリリースして、ウィンドウをすべて閉じる関数
def closeWindows(cap):
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
