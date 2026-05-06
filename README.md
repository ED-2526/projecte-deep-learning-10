# XN-breed-dogs: Clasificación de Razas de Perros 🐶

Este proyecto tiene como objetivo desarrollar un modelo de Deep Learning capaz de clasificar correctamente diferentes razas de perros a partir de imágenes. El proyecto ha sido desarrollado como parte de la asignatura de **Xarxes Neuronals i Aprenentatge Profund**.

## 📁 Estructura del Código

El repositorio está organizado de la siguiente manera para separar la lógica de entrenamiento, los modelos y las utilidades:

* `models/`: Contiene las definiciones de las arquitecturas de las redes neuronales utilizadas.
* `utils/`: Scripts auxiliares para la carga de datos (dataloaders), preprocesamiento de imágenes y otras funciones de ayuda.
* `test/`: Scripts y recursos destinados exclusivamente a la evaluación del modelo.
* `main.py` / `main2.py`: Puntos de entrada principales para ejecutar flujos completos (entrenamiento y validación).
* `train.py`: Script dedicado específicamente al bucle de entrenamiento del modelo.
* `test.py`: Script para ejecutar inferencias y calcular métricas de rendimiento sobre el conjunto de test.
* `test_trained_model.ipynb`: Jupyter Notebook interactivo ideal para visualizar predicciones, analizar errores y probar el modelo ya entrenado de forma visual.
* `environment.yml`: Archivo de configuración con todas las dependencias necesarias para ejecutar el proyecto.

## 🛠️ Instalación y Configuración

Antes de ejecutar el código, es necesario crear un entorno virtual local utilizando Conda. El archivo `environment.yml` incluye todas las dependencias requeridas (PyTorch, librerías de visión artificial, etc.).

1. Clona el repositorio y navega hasta la carpeta:
   ```bash
   git clone [https://github.com/ED-2526/projecte-deep-learning-10.git](https://github.com/ED-2526/projecte-deep-learning-10.git)
   cd projecte-deep-learning-10

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate xnap-example
```

To run the example code:
```
python main.py
```

## Próximos Pasos
* descargar  nuevo dataset (preguntar ramon)
* más modelos
* mas epochs?


## Contributors
Martina Vitagliano
Guillem Batlle
Carla Martinez Vidal
Marcel Izquierdo

Xarxes Neuronals i Aprenentatge Profund
Grau dÉnginyeria de Dades, 
UAB, 2026
