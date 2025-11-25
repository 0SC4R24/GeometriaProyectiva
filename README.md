# Laboratorio Final de Geometría Proyectiva

Este repositorio contiene el código y los recursos utilizados para el laboratorio final del curso de Geometría Proyectiva.
El proyecto se centra en la aplicación de conceptos de geometría proyectiva para resolver problemas relacionados con la 
visión por computadora, utilizando imágenes y datos proporcionados. Se incluyen scripts para cada parte del laboratorio final,
así como recursos adicionales como imágenes y datos necesarios para la ejecución del proyecto.

## 📖 Descripción

El sistema consta de varios scripts de Python que implementan diferentes aspectos de la geometría proyectiva aplicada a imágenes,
como el cálculo de la homografía, la proyección de puntos 3D en imágenes 2D, y los parametros extrínsecos e intrínsecos de la cámara.

El proyecto utiliza bibliotecas populares de Python como OpenCV, NumPy y Matplotlib para el procesamiento de imágenes y cálculos matemáticos.

$P = K[R, t]$ donde:
- $P$ es la matriz de proyección de la cámara.
- $K$ es la matriz de parámetros intrínsecos de la cámara.
- $R$ es la matriz de rotación que representa la orientación de la cámara.
- $t$ es el vector de traslación que representa la posición de la cámara en el espacio 3D.

$pixel = P \cdot world point$ donde:
- $pixel$ es la coordenada del píxel en la imagen 2D.
- $world point$ es la coordenada del punto en el espacio 3D.

## 📂 Estructura del Repositorio

```
GeometriaProyectiva/
├── geometria_proyectiva/                      # Código fuente del proyecto
│   ├── __init__.py                            # Inicializador del paquete
│   └── main.py                                # Archivo principal para ejecutar el proyecto
├── resources/                                 # Recursos adicionales (imágenes, datos, etc.)
│   ├── datos/                                 # Datos utilizados en el proyecto
│   │   ├── imagenes/                          # Imágenes utilizadas en el proyecto
│   │   └── K.txt                              # Archivo de datos K
│   ├── frames/                                # Carpeta con frames del video de futbol
│   ├── GeometriaProyectiva_ProyectoFinal.pdf  # Documento del proyecto final
│   ├── pts_porteria.txt                       # Archivo de puntos de la portería
│   └── u-tad-logo.jpg                         # Logo de U-tad
├── resultados/                                # Resultados generados por el proyecto
├── laboratorio-2.py                           # Archivo del laboratorio 2
├── laboratorio-final-parte-1.py               # Archivo del laboratorio final parte 1
├── laboratorio-final-parte-2.py               # Archivo del laboratorio final parte 2
├── laboratorio-final-parte-3.py               # Archivo del laboratorio final parte 3
├── laboratorio-final-parte-4.py               # Archivo del laboratorio final parte 4
├── README.md                                  # Archivo de documentación del proyecto
└── requirements.txt                           # Archivo de dependencias del proyecto
```

## ▶️ Cómo usar este repositorio

1. Clona este repositorio en tu máquina local:

```bash
git clone https://github.com/0SC4R24/GeometriaProyectiva.git GeometriaProyectiva
```

2. Navega al directorio del proyecto:

```bash
cd GeometriaProyectiva
```

3. Ejecuta los scripts del laboratorio final.

```bash
python laboratorio-final-parte-1.py
```
```bash
python laboratorio-final-parte-2.py
```
```bash
python laboratorio-final-parte-3.py
```
```bash
python laboratorio-final-parte-4.py
```

## 🚀 Requisitos

Asegúrate de tener Python instalado en tu sistema. Puedes descargarlo desde [python.org](https://www.python.org/).
También es posible que necesites instalar algunas bibliotecas adicionales. Puedes hacerlo usando pip:

```bash
pip install -r requirements.txt
```

## 🧑‍💻 Autor

- Nombre: Oscar Viudez
- Email: [oscarviudez24@gmail.com](mailto:oscarviudez24@gmail.com)

## 🎥 Video Example

https://github.com/user-attachments/assets/a0352f1c-1331-44bb-a296-b47c07110ab4
