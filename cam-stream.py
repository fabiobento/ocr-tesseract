import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

def main():
    # Inicializa a câmera
    camera = PiCamera()
    camera.resolution = (320, 240)  # Define a resolução
    camera.framerate = 30  # Define a taxa de quadros

    rawCapture = PiRGBArray(camera, size=(320, 240))
    time.sleep(0.1)  # Tempo para a câmera iniciar

    # Loop para capturar o vídeo
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        img = frame.array  # Obtém a imagem do frame

        # Exibe o vídeo em uma janela
        cv2.imshow("Video Stream", img)

        rawCapture.truncate(0)  # Limpa o buffer de captura

        # Sai do loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Limpeza
    camera.close()  # Fecha a câmera
    cv2.destroyAllWindows()  # Fecha todas as janelas

if __name__ == "__main__":
    main()
