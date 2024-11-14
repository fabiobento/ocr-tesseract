import cv2
from picamera2 import Picamera2, Preview
import time

def main():
    # Inicializa a câmera Picamera2
    camera = Picamera2()
    camera.preview_configuration.main.size = (320, 240)  # Define a resolução
    camera.preview_configuration.main.format = "RGB888"  # Define o formato da imagem para RGB
    camera.preview_configuration.controls.FrameRate = 10  # Define a taxa de quadros
    camera.configure("preview")
    
    camera.start()  # Inicia a câmera
    time.sleep(0.1)  # Tempo para a câmera iniciar

    # Loop para capturar o vídeo
    while True:
        # Captura o frame como um array NumPy (compatível com OpenCV)
        img = camera.capture_array()
        
        # Exibe o vídeo em uma janela
        cv2.imshow("Video Stream", img)
        
        # Sai do loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Limpeza
    camera.stop()  # Fecha a câmera
    cv2.destroyAllWindows()  # Fecha todas as janelas

if __name__ == "__main__":
    main()
