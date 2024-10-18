from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import pytesseract
import imutils    
from imutils.object_detection import non_max_suppression
import os
import gdown
import numpy as np
import urllib.request

def pre_processamento(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #maior = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
  valor, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  return otsu


def download_detector(url, output):
    # Verifica se o arquivo já existe antes de baixá-lo
    if not os.path.exists(output):
        urllib.request.urlretrieve(url, output)
        print("Download concluído!")
    else:
        print(f"Arquivo {output} já existe, não é necessário fazer o download.")
   
def tesseract_setup(config_tesseract = "--tessdata-dir tessdata --psm 7"):
    # Garante que o diretório tessdata existe
    os.makedirs('./tessdata', exist_ok=True)
    # URL do arquivo do detector em PORTUGUÊS e caminho para salvar
    url = 'https://github.com/tesseract-ocr/tessdata/blob/main/por.traineddata?raw=true'
    output = './tessdata/por.traineddata'
    download_detector(url, output)
    # URL do arquivo do detector em INGLÊS e caminho para salvar
    url = 'https://github.com/tesseract-ocr/tessdata/blob/main/eng.traineddata?raw=true'
    output = './tessdata/eng.traineddata'
    download_detector(url, output)
    

def tesseract_OCR(roi, config_tesseract, lang='por'):
    # Processar a ROI com a função pre_processamento
    preprocess_roi = pre_processamento(roi)
    texto = pytesseract.image_to_string(preprocess_roi, lang=lang, config=config_tesseract)
    return texto

def net_create(detector = './Modelos/frozen_east_text_detection.pb'):
    # Criar o diretório
    os.makedirs('./Modelos', exist_ok=True)
    # Baixar o arquivo da rede neural armazenado do Google Drive
    if not os.path.exists(detector):
        url = 'https://drive.google.com/uc?id=1-RbGz-8K7kC_Fve6J0eLtcRZQhmKS3UQ'
        gdown.download(url, detector, quiet=False)
    else:
        print("Arquivo frozen_east_text_detection.pb já existe, não é necessário fazer o download.")
    return cv2.dnn.readNet(detector)

def net_forward(img, rede_neural, min_confianca = 0.90,
                nomes_camadas = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']):
    blob = cv2.dnn.blobFromImage(img, 1.0, (img.shape[1], img.shape[0]), swapRB = True, crop = False)
    rede_neural.setInput(blob)
    scores, geometry = rede_neural.forward(nomes_camadas)
    linhas, colunas = scores.shape[2:4]
    caixas = []
    confiancas = []    
    for y in range(0, linhas):
        data_scores = scores[0, 0, y]
        data_angulos, xData0, xData1, xData2, xData3 = dados_geometricos(geometry, y)
        for x in range(0, colunas):
            if data_scores[x] < min_confianca:
                continue
            inicioX, inicioY, fimX, fimY = calculos_geometria(x, y, data_angulos, xData0, xData1, xData2, xData3)
            confiancas.append(data_scores[x])
            caixas.append((inicioX, inicioY, fimX, fimY))    
    return non_max_suppression(np.array(caixas), probs=confiancas)    

# Funções para decodificação de saída de rede neural
def dados_geometricos(geometry, y):
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    data_angulos = geometry[0, 4, y]
    return data_angulos, xData0, xData1, xData2, xData3

def calculos_geometria(x, y, data_angulos, xData0, xData1, xData2, xData3):
    (offsetX, offsetY) = (x * 4.0, y * 4.0)
    angulo = data_angulos[x]
    cos = np.cos(angulo)
    sin = np.sin(angulo)
    h = xData0[x] + xData2[x]
    w = xData1[x] + xData3[x]

    fimX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
    fimY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))

    inicioX = int(fimX - w)
    inicioY = int(fimY - h)
    return inicioX, inicioY, fimX, fimY

def main():
    ## ESCREVA SEU CÓDIGO AQUI

if __name__ == "__main__":
    main()
