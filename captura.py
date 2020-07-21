import cv2
import numpy as np

classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classificadorOlho = cv2.CascadeClassifier("haarcascade_eye.xml")
camera = cv2.VideoCapture(1) #faz a captura do vídeo # 0 indica a propria webcam do computador
amostra = 1
numeroAmostras = 25;
id = input('Digite seu identificador: ')
largura, altura = 220, 220
print("Capturando as faces...")

while(True):
    conectado, imagem = camera.read() #leirura da imagem
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) #converte a imagem para escalas de cinza
    print(np.average(imagemCinza))
    facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(100,100)) #scaleFactor indica a escala da webcam

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,0,255),2)
        regiao = imagem[y:y + a,x:x + 1]
        #convertendo a regiao da escala de cinza
        regiaoCinzaolho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaolho)
        for (ox, oy, ol, oa) in olhosDetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0,255,0), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if np.average(imagemCinza)  > 110: #imagemCinza é a capturada pela webcam
                    imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + 1], (largura, altura))
                    cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace) #grava a imahem na pasta
                    print("[Foto " + str(amostra) + " capturada com sucesso]")
                    amostra += 1

    cv2.imshow("Face", imagem) #imshow exibe a imagem
    cv2.waitKey(1) #display a frame por 1 ms
    if (amostra >= numeroAmostras + 1):
        break

print("Faces capturadas com sucesso")
camera.release() #libera a memória
cv2.destroyAllWindows()
