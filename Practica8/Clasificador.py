import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict
import re



# Función para cargar y entrenar el clasificador
def entrenar_clasificador(nombre_archivo):
    # Cargar los datos desde el archivo CSV
    data = pd.read_csv(nombre_archivo)

    # Dividir los datos en características (X) y variable objetivo (y)
    X = data[['petallength', 'petalwidth']]
    y = data['class']

    # Asignar nombres a las características
    X.columns = ['petallength', 'petalwidth']

    # Inicializar el clasificador
    classifier = LogisticRegression()

    # Entrenar el clasificador
    classifier.fit(X, y)

    return classifier, X, y





# Función para clasificar una nueva muestra
def clasificar_nueva_muestra(classifier, X):
    # Pedir al usuario que ingrese los valores para petallength y petalwidth de las nuevas muestras
    valores = input("Ingrese los valores de petallength y petalwidth para las nuevas muestras separados por coma y entre paréntesis (por ejemplo, (1.2,0.3),(1.5,0.6),(4.2,0.5)): ")
    
    # Parsear los valores ingresados por el usuario
    try:
        muestras = []
        for muestra_str in re.findall(r'\([^)]*\)', valores):
            muestra_str = muestra_str.strip('()')
            petallength_nueva, petalwidth_nueva = map(float, muestra_str.split(','))
            muestras.append([petallength_nueva, petalwidth_nueva])
    except ValueError:
        print("Error: Ingrese los valores en el formato correcto.")
        return
    
    # Hacer la predicción utilizando el clasificador entrenado
    for muestra in muestras:
        prediccion = classifier.predict([muestra])
        print("Clase predicha para la muestra:", prediccion[0])





# Función para mostrar las muestras posibles encontradas para cada clase
def mostrar_muestras_posibles(X, y):
    muestras_por_clase = defaultdict(list)
    for i in range(len(X)):
        clase = y[i]
        muestra = X.iloc[i].values
        muestras_por_clase[clase].append(muestra)

    print("\nMuestras posibles encontradas para cada clase:")
    for clase, muestras in muestras_por_clase.items():
        print(f"Clase '{clase}':")
        for muestra in muestras:
            print(f"   {muestra}")

if __name__ == "__main__":
    # Nombre del archivo CSV de entrenamiento
    nombre_archivo_entrenamiento = "train.csv"

    # Entrenar el clasificador
    classifier, X, y = entrenar_clasificador(nombre_archivo_entrenamiento)

    # Clasificar una nueva muestra
    clasificar_nueva_muestra(classifier, X)
    
    # Mostrar las muestras posibles encontradas para cada clase
    #mostrar_muestras_posibles(X, y)
