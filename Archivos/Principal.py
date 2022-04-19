from Interfaz.Interfaz import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import cv2
import imutils
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import *
from matplotlib import pyplot as plt  
from PIL import Image  
import random
class Principal(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btn_Abrir.clicked.connect(self.Cargar_Imagen)
        self.btn_Limpiar.clicked.connect(self.close)
        self.btn_Sumar.clicked.connect(self.suma)
        self.btn_Restar.clicked.connect(self.resta)
        self.btn_Histo.clicked.connect(self.histograma)
        self.btn_Umbral.clicked.connect(self.Binario)
        self.Com()
        self.btn_Umbral_2.clicked.connect(self.Borde)
        self.btn_Cuadrado.clicked.connect(self.Mult_Imagenes)
        self.btn_Gris.clicked.connect(self.Practica_3)

    #Configuramos los conboxs
    def Com(self):
        Suma = ["Suma", "Suma Trunca", "Super Escalar"]       
        self.Box_Suma.addItems(Suma) 
        self.Box_Suma.setMouseTracking(True)         
        self.Box_Suma.setStyleSheet("border : 1px solid red;") 
        Resta = ["Resta","Resta Trunca","Resta Escalar","Negativo"]
        self.Box_Resta.addItems(Resta) 
        self.Box_Resta.setMouseTracking(True)         
        self.Box_Resta.setStyleSheet("border : 1px solid red;") 
        Binario = ["Binary","Binary_Inv","Trunc","Tozero","Tozero_Inv","Otsu"]
        self.Box_Binario.addItems(Binario) 
        self.Box_Binario.setMouseTracking(True)
        self.Box_Binario.setStyleSheet("border : 1px solid red;") 

        Borde = ["Borde","Borde Horizontal","Borde Vertical"]
        self.Box_Borde.addItems(Borde) 
        self.Box_Borde.setMouseTracking(True)
        self.Box_Borde.setStyleSheet("border : 1px solid red;") 

        Cuadrado = ["Multiplicar Imagen","Restar Imagen","Sumar Imagen","Restar Imagen Color"]
        self.Box_Cuadrado.addItems(Cuadrado) 
        self.Box_Cuadrado.setMouseTracking(True)
        self.Box_Cuadrado.setStyleSheet("border : 1px solid red;") 

        Gris = ["Suma 0 y 255","Suma 255 y 0","Restar 0 y 255","Restar 255 y 0","Suma 130 y fondo aleatorio","Resta Aleatorio y fondo 0"]
        self.Box_Gris.addItems(Gris) 
        self.Box_Gris.setMouseTracking(True)
        self.Box_Gris.setStyleSheet("border : 1px solid red;") 

    #Creamos variables Globales
    filename = None
    final_image = None
    #Metodo para obtener la imagen
    def Cargar_Imagen(self):
        #Variablle donde guardaremos la ruta de la imagen
        global filename
        #Obtehemos la rita
        filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        #Leemos la imagen
        imagen = cv2.imread(filename)
        #Metodo donde cargara la imagen
        self.setPhoto(imagen)

    #Funcion para poner la image el en label
    def setPhoto(self,image):
        #Redimensionamos la imagen 
        image = imutils.resize(image,width=340) 
        #Convertimos a RGB porque la interfaz lo detecta de esa forma
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        #image = image.scaled(500, 380, Qt.KeepAspectRatio)
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))
    
    #Suma de una imagen
    def suma(self):
        global filename,final_image    
        final_image = cv2.imread(filename)
        (alto,ancho,canales) = final_image.shape
        #Obetenemos el numero de la interfaz
        pixeles = int(self.txt_Datos.text())
        Suma = ["Suma", "Suma Trunca", "Super Escalar"]       
        opcion = self.Box_Suma.currentText()
        suma = np.zeros(final_image.shape, dtype=np.int16)
        im12 = final_image
        i = 0
        while i < alto:
            j = 0
            while j < ancho:
                k=0
                while k<canales:
                    valor = im12[i,j,k]            
                    if opcion == Suma[0]:
                        valor = valor + pixeles
                        if valor>= 255:
                            valor = 255
                        else:
                            valor = valor
                        im12[i,j,k] = valor                    
                    k+=1
                j+=1
            i+=1    
        #Redimensionamos la imagen 
        image = imutils.resize(im12,width=340) 
        #Convertimos a RGB porque la interfaz lo detecta de esa forma
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        #image = image.scaled(500, 380, Qt.KeepAspectRatio)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(image))

    #Suma de una Resta
    def resta(self):
        global filename,final_image    
        final_image = cv2.imread(filename)
        (alto,ancho,canales) = final_image.shape
        #Obetenemos el numero de la interfaz
        pixeles = int(self.txt_Datos.text())
        im12 = final_image
        i = 0

        while i < alto:
            j = 0
            while j < ancho:
                k=0
                while k<canales:
                    valor = im12[i,j,k]            
                    #Restamos los pixeles
                    valor = valor - pixeles
                    if valor <0:
                        valor = 0
                    else:
                        valor = valor
                    im12[i,j,k] = valor
                    k+=1
                j+=1
            i+=1
        #Redimensionamos la imagen 
        image = imutils.resize(im12,width=340) 
        #Convertimos a RGB porque la interfaz lo detecta de esa forma
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        #image = image.scaled(500, 380, Qt.KeepAspectRatio)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(image))



    #Suma de una Resta
    def multi(self):
        global filename,final_image    
        final_image = cv2.imread(filename)
        (alto,ancho,canales) = final_image.shape
        #Obetenemos el numero de la interfaz
        pixeles = int(self.txt_Datos.text())
        im12 = final_image
        i = 0

        while i < alto:
            j = 0
            while j < ancho:
                k=0
                while k<canales:
                    valor = im12[i,j,k]            
                    valor = valor * pixeles
                    if valor>= 255:
                        valor = 255
                    else:
                        valor = valor
                    im12[i,j,k] = valor
                    k+=1
                j+=1
            i+=1

        #Redimensionamos la imagen 
        image = imutils.resize(im12,width=340) 
        #Convertimos a RGB porque la interfaz lo detecta de esa forma
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        #image = image.scaled(500, 380, Qt.KeepAspectRatio)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(image))

    #Suma de una Resta
    def div(self):
        global filename,final_image    
        final_image = cv2.imread(filename)
        (alto,ancho,canales) = final_image.shape
        #Obetenemos el numero de la interfaz
        pixeles = int(self.txt_Datos.text())
        im12 = final_image
        i = 0


        while i < alto:
            j = 0
            while j < ancho:
                k=0
                while k<canales:
                    valor = im12[i,j,k]            
                    valor = valor / pixeles
                    valor = int(valor)
                    if valor<= 0:
                        valor = abs(valor)
                    else:
                        valor = valor
                    im12[i,j,k] = valor
                    k+=1
                j+=1
            i+=1

        #Redimensionamos la imagen 
        image = imutils.resize(im12,width=340) 
        #Convertimos a RGB porque la interfaz lo detecta de esa forma
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        #image = image.scaled(500, 380, Qt.KeepAspectRatio)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(image))


    def histograma(self):
        """
        global filename,final_image    
        final_image = Image.open(filename)

        #Obetenemos el numero de la interfaz
        #pixeles = int(self.txt_Datos.text())
        global filename,final_image    
        #final_image = Image.open(filename)
        imagen = cv2.imread(filename)
        histo = self.histo(imagen)
        plt.subplot(312)
        plt.plot(histo,color='g')
        plt.show()

        imagen_2 = cv2.imread(filename)
        histo2 = self.histo(imagen_2)
        plt.subplot(313)
        plt.plot(histo2,color='r')
        plt.show()
        """
        imagen = cv2.imread(filename)
        canales = cv2.split(imagen)
        colores = ('b','g','r')
        plt.figure()
        plt.title("Histograma de Colores")
        plt.xlabel("Bits")
        plt.ylabel("#Pixel")
        for (canal,color) in zip(canales,colores):
            hist = cv2.calcHist([canal],[0],None,[256],[0,256])
            plt.plot(hist,color=color)
            plt.xlim([0,256])
        plt.show()
        """
        
        final_image = Image.open(filename)
        im = final_image.convert('L')
        im9 = im
        [ren,col] = im9.size
        total = ren * col
        a = np.asarray(im9, dtype = np.float32)
        a = a.reshape(1, total)
        a = a.astype(int)
        a = max(a)
        valor = 0
        maxd = max(a)
        grises = maxd
        vec = np.zeros(grises + 1)
        for i in range(total - 1):
            valor = a[i]
            vec[valor] = vec[valor] + 1    
        plt.plot(vec)
        plt.show()
"""        

    def histo(self,imagen):
        h = np.zeros([256]);
        img = np.reshape(imagen,-1)
        for p in img:
            h[p] +=1
        return h





    def Binario(self):
        global filename,final_image    
        final_image = cv2.imread(filename)
        (alto,ancho,canales) = final_image.shape
        #Optenemos la opcion obtenida en en ComboBox
        opcion = self.Box_Binario.currentText()
        Binario = ["Binary","Binary_Inv","Trunc","Tozero","Tozero_Inv","Otsu"]
        #Obetenemos el numero registrado de la interfaz
        pixeles = int(self.txt_Datos.text())
        im12 = final_image
        i = 0
        while i < alto:
            j = 0
            while j < ancho:
                k=0
                while k<canales:
                    valor = im12[i,j,k]  
                    if opcion == Binario[0]:
                        if valor <  pixeles:
                            valor = 0
                        else:
                            valor = 255
                    elif opcion == Binario[1]:
                        if valor >  pixeles:
                            valor = 0
                        else:
                            valor = 255
                    elif opcion == Binario[2]:
                        if valor >  pixeles:
                            valor = pixeles 
                        else:
                            valor = valor
                    elif opcion == Binario[3]:
                        if valor >  pixeles:
                            valor = valor 
                        else:
                            valor = 0
                    elif opcion == Binario[4]:
                        if valor <  pixeles:
                            valor = valor  
                        else:
                            valor = 0
                    elif opcion == Binario[5]:
                        pass
                    im12[i,j,k] = valor
                    k+=1
                j+=1
            i+=1

        #Redimensionamos la imagen 
        image = imutils.resize(im12,width=340) 
        #Convertimos a RGB porque la interfaz lo detecta de esa forma
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        #image = image.scaled(500, 380, Qt.KeepAspectRatio)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(image))


    def Binario_Inv(self):
        global filename,final_image    
        final_image = cv2.imread(filename)
        (alto,ancho,canales) = final_image.shape
        #Obetenemos el numero de la interfaz
        pixeles = int(self.txt_Datos.text())
        im12 = final_image
        i = 0
        while i < alto:
            j = 0
            while j < ancho:
                k=0
                while k<canales:
                    valor = im12[i,j,k]            
                    if valor >  pixeles:
                        valor = 0
                    else:
                        valor = 255
                    im12[i,j,k] = valor
                    k+=1
                j+=1
            i+=1

        #Redimensionamos la imagen 
        image = imutils.resize(im12,width=340) 
        #Convertimos a RGB porque la interfaz lo detecta de esa forma
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        #image = image.scaled(500, 380, Qt.KeepAspectRatio)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(image))


    def Trunc(self):
        global filename,final_image    
        final_image = cv2.imread(filename)
        (alto,ancho,canales) = final_image.shape
        #Obetenemos el numero de la interfaz
        pixeles = int(self.txt_Datos.text())
        im12 = final_image
        i = 0
        while i < alto:
            j = 0
            while j < ancho:
                k=0
                while k<canales:
                    valor = im12[i,j,k]            
                    #Cuando los pixeles son mayores al valor del Umbral, toman el valor del umbral
                    if valor >  pixeles:
                        valor = pixeles 
                    else:
                        valor = valor

                    im12[i,j,k] = valor
                    k+=1
                j+=1
            i+=1

        #Redimensionamos la imagen 
        image = imutils.resize(im12,width=340) 
        #Convertimos a RGB porque la interfaz lo detecta de esa forma
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        #image = image.scaled(500, 380, Qt.KeepAspectRatio)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(image))


    def Tozero(self):
        global filename,final_image    
        final_image = cv2.imread(filename)
        (alto,ancho,canales) = final_image.shape
        #Obetenemos el numero de la interfaz
        pixeles = int(self.txt_Datos.text())
        im12 = final_image
        i = 0
        while i < alto:
            j = 0
            while j < ancho:
                k=0
                while k<canales:
                    valor = im12[i,j,k]            
                    #Cuando los pixeles son mayores al valor del Umbral, toman el valor del umbral
                    if valor >  pixeles:
                        valor = valor 
                    else:
                        valor = 0
                    im12[i,j,k] = valor
                    k+=1
                j+=1
            i+=1

        #Redimensionamos la imagen 
        image = imutils.resize(im12,width=340) 
        #Convertimos a RGB porque la interfaz lo detecta de esa forma
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        #image = image.scaled(500, 380, Qt.KeepAspectRatio)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(image))


    def Tozero_inv(self):
        #Variable Global
        global filename,final_image    
        #Abrimos la imagen
        final_image = cv2.imread(filename)
        #Optenemos el alto y ancho de la imagen
        (alto,ancho,canales) = final_image.shape
        #Obetenemos el numero de la interfaz
        pixeles = int(self.txt_Datos.text())
        im12 = final_image
        i = 0
        while i < alto:
            j = 0
            while j < ancho:
                k=0
                while k<canales:
                    valor = im12[i,j,k]            
                    #Cuando los pixeles son mayores al valor del Umbral, toman el valor del umbral
                    if valor <  pixeles:
                        valor = 0  
                    else:
                        valor = valor
                    im12[i,j,k] = valor
                    k+=1
                j+=1
            i+=1

        #Redimensionamos la imagen 
        image = imutils.resize(im12,width=340) 
        #Convertimos a RGB porque la interfaz lo detecta de esa forma
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        #image = image.scaled(500, 380, Qt.KeepAspectRatio)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(image))

    def Borde(self):
        #Variable Global
        global filename,final_image    
        #Abrimos la imagen
        final_image = cv2.imread(filename)
        #Optenemos el alto y ancho de la imagen
        (alto,ancho,canales) = final_image.shape
        #Obetenemos el numero de la interfaz

        Borde = ["Borde","Borde Horizontal","Borde Vertical"]
        opcion = self.Box_Borde.currentText()
        Imag_Procesada = final_image
        if opcion == Borde[0]:
            pixeles = int(self.txt_Datos.text())
            #Arriba
            Imag_Procesada[0:pixeles,:,:] = [255,255,0]
            #Abajp
            Imag_Procesada[Imag_Procesada.shape[1]-pixeles:Imag_Procesada.shape[1],:,:] = [255,255,0]
            #Extremos
            #M = np.array([[1,0,Imag_Procesada[0]],[0,1,Imag_Procesada.shape[1]]])
            Imag_Procesada[:,-pixeles:Imag_Procesada.shape[0]-1,:]  = [255,255,0]
            Imag_Procesada[:,-Imag_Procesada.shape[0]-1:pixeles,:]  = [255,255,0]

        elif opcion == Borde[1]:
        
            for i in range(0,1000,40):
                colos = random.randint(0,255)
                #Imag_Procesada[i:i+10,:,:] = [i+20,i+40,0]
                Imag_Procesada[i:i+10,:,:] = [colos,colos,0]

            #Extremos
            #M = np.array([[1,0,Imag_Procesada[0]],[0,1,Imag_Procesada.shape[1]]])
            #Imag_Procesada[:,-pixeles:Imag_Procesada.shape[0]-1,:]  = [255,255,0]
            #Imag_Procesada[:,-Imag_Procesada.shape[0]-1:pixeles,:]  = [255,255,0]
        elif opcion == Borde[2]:
        
            Imag_Procesada[:,-Imag_Procesada.shape[0]+2**3:2**4,:]  = [255,255,0]
            Imag_Procesada[:,-Imag_Procesada.shape[0]+2**5:2**6,:]  = [255,255,0]
            Imag_Procesada[:,-Imag_Procesada.shape[0]+2**7:2**8,:]  = [255,255,0]
            #Imag_Procesada[:,-Imag_Procesada.shape[0]+2**9:513,:]  = [255,255,0]

            """              
            k = 2
            for i in range(0,50,2):
                Imag_Procesada[:,-Imag_Procesada.shape[0]+i**2:k,:]  = [255,255,0]
                k = k**2 + 4
            """
            #for i in range(0,500,20):
                #for j in range(10,500,30):

                    #Imag_Procesada[:,-Imag_Procesada.shape[0]+i:j,:]  = [255,255,0]



        """
        #Redimensionamos la imagen 
        Border = cv2.copyMakeBorder(
            final_image, pixeles, pixeles, pixeles, pixeles, cv2.BORDER_CONSTANT, value=[255, 255, 0])
        """
        image = imutils.resize(Imag_Procesada,width=340) 
        #Convertimos a RGB porque la interfaz lo detecta de esa forma
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        #image = image.scaled(500, 380, Qt.KeepAspectRatio)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(image))


    def Mult_Imagenes(self):
        global filename,final_image    
        final_image = cv2.imread(filename)
        final_image = imutils.resize(final_image,width=340) 
        Cuadrado = cv2.imread("./Imagens/Cuadrado.PNG")
        Cuadrado = imutils.resize(Cuadrado,width=340) 
        Medio = cv2.imread("./Imagens/Medio.PNG")
        Medio = imutils.resize(Medio,width=340) 
        Color = cv2.imread("./Imagens/Color.PNG")
        Color = imutils.resize(Color,width=340) 
        Colores = cv2.imread("./Imagens/Colores.PNG")
        Colores = imutils.resize(Colores,width=340) 
        (alto,ancho,canales) = final_image.shape
        #Obetenemos el numero de la interfaz
        #pixeles = int(self.txt_Datos.text())
        Cuadrados = ["Multiplicar Imagen","Restar Imagen","Sumar Imagen","Restar Imagen Color"]
        opcion = self.Box_Cuadrado.currentText()
        im12 = final_image
        i = 0

        while i < alto:
            j = 0
            while j < ancho:     
                if opcion == Cuadrados[0]:              
                    if i < Cuadrado.shape[0] and j<Cuadrado.shape[1]:      
                        valor = im12[i,j]    
                        valor = valor * Cuadrado[i,j]
                        if valor.all()>255:
                            valor = 255
                        else:
                            valor = valor 
                        im12[i,j] = valor

                if opcion == Cuadrados[1]:
                    if i < Medio.shape[0] and j<Medio.shape[1]:      
                        valor = im12[i,j]    
                        valor = valor - Medio[i,j]
                        if valor.all() <0:
                            valor = 0
                        else:
                            valor = valor
                        im12[i,j] = valor
                if opcion == Cuadrados[2]:
                    if i < Color.shape[0] and j<Color.shape[1]:      
                        valor = im12[i,j]    
                        valor = valor + Color[i,j]
                        if valor.all() >255:
                            valor = 255
                        else:
                            valor = valor
                        im12[i,j] = valor
                if opcion == Cuadrados[3]:
                    if i < Colores.shape[0] and j<Colores.shape[1]:      
                        valor = im12[i,j]    
                        valor = valor - Colores[i,j]
                        if valor.all() <0:
                            valor = 0
                        else:
                            valor = valor
                        im12[i,j] = valor

                j+=1
            i+=1
        #Redimensionamos la imagen 
        image = imutils.resize(im12,width=340) 
        #Convertimos a RGB porque la interfaz lo detecta de esa forma
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        #image = image.scaled(500, 380, Qt.KeepAspectRatio)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(image))


    def Practica_3(self):
        global filename,final_image    
        final_image = cv2.imread(filename)
        final_image = imutils.resize(final_image,width=340) 
        #Imagen cpn un cuadrado negro en medio y fondo blanco
        Cuadrado = cv2.imread("./Imagens/Practica_3/Medio_Negro.PNG")
        Cuadrado = imutils.resize(Cuadrado,width=340) 
        #Imagen cpn un cuadrado blanco en medio y fondo negro
        Medio = cv2.imread("./Imagens/Practica_3/Medio.PNG")
        Medio = imutils.resize(Medio,width=340) 
        #Cuadrado con nivel 130 y fondo aleatorio
        Color = cv2.imread("./Imagens/Practica_3/Fondo0.PNG")
        Color = imutils.resize(Color,width=340) 
        #Cuadrao con colores diefrenetes y fondo 0
        Colores = cv2.imread("./Imagens/Practica_3/Fondo_Aleatorio.PNG")
        Colores = imutils.resize(Colores,width=340) 



        (alto,ancho,canales) = final_image.shape
        #Obetenemos el numero de la interfaz
        #pixeles = int(self.txt_Datos.text())
        Cuadrados = ["Suma 0 y 255","Suma 255 y 0","Suma 0 y 255","Restar 255 y 0","Suma 130 y fondo aleatorio","Resta Aleatorio y fondo 0"]
        opcion = self.Box_Gris.currentText()
        im12 = final_image
        i = 0

        while i < alto:
            j = 0
            while j < ancho:     
                if opcion == Cuadrados[0]:              
                    if i < Cuadrado.shape[0] and j<Cuadrado.shape[1]:      
                        valor = im12[i,j]    
                        valor = valor + Cuadrado[i,j]
                        if valor.all()>255:
                            valor = 255
                        else:
                            valor = valor 
                        im12[i,j] = valor

                if opcion == Cuadrados[1]:
                    if i < Medio.shape[0] and j<Medio.shape[1]:      
                        valor = im12[i,j]    
                        valor = valor + Medio[i,j]
                        if valor.all() > 255:
                            valor = 255
                        else:
                            valor = valor
                        im12[i,j] = valor
                if opcion == Cuadrados[2]:
                    if i < Cuadrado.shape[0] and j<Cuadrado.shape[1]:      
                        valor = im12[i,j]    
                        valor = valor - Cuadrado[i,j]
                        if valor.all() <0:
                            valor = 0
                        else:
                            valor = valor
                        im12[i,j] = valor
                if opcion == Cuadrados[3]:
                    if i < Medio.shape[0] and j<Medio.shape[1]:      
                        valor = im12[i,j]    
                        valor = valor - Medio[i,j]
                        if valor.all() <0:
                            valor = 0
                        else:
                            valor = valor
                        im12[i,j] = valor

                if opcion == Cuadrados[4]:
                    if i < Color.shape[0] and j<Color.shape[1]:      
                        valor = im12[i,j]    
                        valor = valor + Color[i,j]
                        if valor.all() >255:
                            valor = 255
                        else:
                            valor = valor
                        im12[i,j] = valor

                if opcion == Cuadrados[5]:
                    if i < Colores.shape[0] and j<Colores.shape[1]:      
                        valor = im12[i,j]    
                        valor = valor - Colores[i,j]
                        if valor.all() <0:
                            valor = 0
                        else:
                            valor = valor
                        im12[i,j] = valor
                j+=1
            i+=1
        #Redimensionamos la imagen 
        image = imutils.resize(im12,width=340) 
        #Convertimos a RGB porque la interfaz lo detecta de esa forma
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        #image = image.scaled(500, 380, Qt.KeepAspectRatio)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(image))






























