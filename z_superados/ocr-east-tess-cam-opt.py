import cv2
import pytesseract
import numpy as np
import os
import gdown
import time
from imutils.object_detection import non_max_suppression
import urllib.request

def pre_processamento(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    valor, preprocess_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return preprocess_img

def download_detector(url, output):
    if not os.path.exists(output):
        urllib.request.urlretrieve(url, output)
        print("Download concluído!")
    else:
        print(f"Arquivo {output} já existe, não é necessário fazer o download.")

def tesseract_setup(config_tesseract="--tessdata-dir tessdata --psm 7"):
    os.makedirs('./tessdata', exist_ok=True)
    urls = [
        ('https://github.com/tesseract-ocr/tessdata/blob/main/por.traineddata?raw=true', './tessdata/por.traineddata'),
        ('https://github.com/tesseract-ocr/tessdata/blob/main/eng.traineddata?raw=true', './tessdata/eng.traineddata')
    ]
    for url, output in urls:
        download_detector(url, output)

def tesseract_OCR(roi, config_tesseract, lang='por'):
    preprocess_roi = pre_processamento(roi)
    texto = pytesseract.image_to_string(preprocess_roi, lang=lang, config=config_tesseract)
    return texto

def net_create(detector='./Modelos/frozen_east_text_detection.pb'):
    os.makedirs('./Modelos', exist_ok=True)
    if not os.path.exists(detector):
        url = 'https://drive.google.com/uc?id=1-RbGz-8K7kC_Fve6J0eLtcRZQhmKS3UQ'
        gdown.download(url, detector, quiet=False)
    return cv2.dnn.readNet(detector)

def net_forward(img, rede_neural, min_confianca=0.90, nomes_camadas=['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']):
    blob = cv2.dnn.blobFromImage(img, 1.0, (img.shape[1], img.shape[0]), swapRB=True, crop=False)
    rede_neural.setInput(blob)
    scores, geometry = rede_neural.forward(nomes_camadas)
    linhas, colunas = scores.shape[2:4]
    caixas = []
    confiancas = []
    
    for y in range(linhas):
        data_scores = scores[0, 0, y]
        data_angulos, xData0, xData1, xData2, xData3 = dados_geometricos(geometry, y)
        for x in range(colunas):
            if data_scores[x] < min_confianca:
                continue
            inicioX, inicioY, fimX, fimY = calculos_geometria(x, y, data_angulos, xData0, xData1, xData2, xData3)
            confiancas.append(data_scores[x])
            caixas.append((inicioX, inicioY, fimX, fimY))
    return non_max_suppression(np.array(caixas), probs=confiancas)

def dados_geometricos(geometry, y):
    return geometry[0, 4, y], geometry[0, 0, y], geometry[0, 1, y], geometry[0, 2, y], geometry[0, 3, y]

def calculos_geometria(x, y, data_angulos, xData0, xData1, xData2, xData3):
    offsetX, offsetY = x * 4.0, y * 4.0
    angulo, cos, sin = data_angulos[x], np.cos(data_angulos[x]), np.sin(data_angulos[x])
    h, w = xData0[x] + xData2[x], xData1[x] + xData3[x]
    fimX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
    fimY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
    inicioX, inicioY = int(fimX - w), int(fimY - h)
    return inicioX, inicioY, fimX, fimY

def main():
    config_tesseract = "--tessdata-dir tessdata --psm 7"
    tesseract_setup(config_tesseract)
    rede_neural = net_create()

    camera = cv2.VideoCapture(0)  # Use 0 para a webcam padrão
    camera.set(cv2.CAP_PROP_FPS, 1)  # Define o FPS para 1
    last_detection_time = time.time()
    detection_interval = 5  # Intervalo de 5 segundos
    detections = []  # Lista para armazenar detecções

    while True:
        ret, img = camera.read()
        if not ret:
            break
        
        # Redimensiona a imagem para a entrada da rede
        img_resized = cv2.resize(img, (320, 320))

        # Verifica se é hora de realizar a detecção
        if time.time() - last_detection_time >= detection_interval:
            deteccoes = net_forward(img_resized, rede_neural)
            detections = []  # Reinicia a lista de detecções
            for (inicioX, inicioY, fimX, fimY) in deteccoes:
                roi = img[inicioY:fimY, inicioX:fimX]
                texto = tesseract_OCR(roi, config_tesseract)
                detections.append((inicioX, inicioY, fimX, fimY, texto))
            last_detection_time = time.time()  # Atualiza o tempo da última detecção

        # Desenha todas as detecções na imagem
        for (inicioX, inicioY, fimX, fimY, texto) in detections:
            cv2.rectangle(img, (inicioX, inicioY), (fimX, fimY), (0, 255, 0), 2)
            cv2.putText(img, texto, (inicioX, inicioY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Video Stream", img)
        
        # Controla a taxa de quadros para 1 FPS
        if cv2.waitKey(1000) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
