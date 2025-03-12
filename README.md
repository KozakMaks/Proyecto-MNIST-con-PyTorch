# Proyecto MNIST con PyTorch

Este proyecto implementa una red neuronal convolucional para el reconocimiento de dígitos utilizando PyTorch. Se incluye un script para entrenar el modelo y una aplicación con interfaz gráfica que permite predecir dígitos escritos a mano.

## Contenido

- `train.py`: Entrena la red neuronal utilizando el dataset MNIST y guarda el modelo en `model.pth`.
- `app.py`: Interfaz gráfica construida con Tkinter para dibujar un dígito y predecir su valor con el modelo entrenado.

## Requisitos

- Python 3.x
- PyTorch
- torchvision
- Pillow
- Tkinter (incluido en la mayoría de las distribuciones de Python)

## Instalación

Instala las dependencias ejecutando:

```bash
pip install torch torchvision pillow
