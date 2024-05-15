import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

def opciones(value):
    if value == 1:
        # Cargar el dataset de Iris
        iris = load_iris()
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        irisvalue = int(input("Escoge una acción  1. mostrar datos iris, 2. HoldOut , 3. 10-Fold"))
        if irisvalue == 1:
            # Configurar pandas para mostrar más filas
            pd.set_option('display.max_rows', None)  # None significa que no hay límite en el número de filas mostradas
            # Imprimir el DataFrame completo
            print(iris_df)
        elif irisvalue == 2:
            # Dividir el dataset en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(iris_df[iris.feature_names], iris.target, test_size=0.3, random_state=42, stratify=iris.target)

            # Crear el clasificador k-NN y entrenarlo con el conjunto de entrenamiento
            knn = KNeighborsClassifier(n_neighbors=5)  # Puedes ajustar el número de vecinos si es necesario
            knn.fit(X_train, y_train)

            # Predecir las clases para el conjunto de prueba
            y_pred = knn.predict(X_test)

            # Crear un DataFrame para mostrar los datos de prueba y las predicciones
            results_df = pd.DataFrame(X_test, columns=iris.feature_names)
            results_df['Predicted Species'] = pd.Categorical.from_codes(y_pred, iris.target_names)
            results_df['True Species'] = pd.Categorical.from_codes(y_test, iris.target_names)

            # Imprimir los resultados
            print(results_df)
        elif irisvalue == 3:
            # Configurar la validación cruzada de 10-Fold estratificada
            skf = StratifiedKFold(n_splits=10)
            # Crear el clasificador, k-NN en este caso
            knn = KNeighborsClassifier(n_neighbors=5)
            # Almacenar las puntuaciones de cada fold para luego calcular un promedio
            scores = []

            # Ejecutar la validación cruzada
            for train_index, test_index in skf.split(iris.data, iris.target):
                X_train, X_test = iris.data[train_index], iris.data[test_index]
                y_train, y_test = iris.target[train_index], iris.target[test_index]

                # Entrenar el modelo con el conjunto de entrenamiento del fold actual
                knn.fit(X_train, y_train)
                
                # Predecir el conjunto de prueba del fold actual
                y_pred = knn.predict(X_test)
                
                # Calcular la precisión del modelo en el fold actual
                accuracy = accuracy_score(y_test, y_pred)
                scores.append(accuracy)

            # Calcular la precisión media de todos los folds
            average_accuracy = np.mean(scores)

            print("Precisión promedio con 10-Fold Cross-Validation estratificado:", average_accuracy)
        else:
            print("No se ha escogido ninguna opción")
    elif value == 2:
        # Cargar el dataset de Wine
        wine = load_wine()
        wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
        
        # Añadir una columna con las etiquetas de las clases (tipo de vino)
        wine_df['target'] = wine.target
        wine_df['target_name'] = pd.Categorical.from_codes(wine.target, wine.target_names)

        winevalue = int(input("Escoge una acción  1. mostrar datos wine, 2. HoldOut , 3. 10-Fold"))
        if winevalue == 1:
            #pd.set_option('display.max_rows', None)  # None significa que no hay límite en el número de filas mostradas
            # Imprimir el DataFrame completo
            print(wine_df)
        elif winevalue == 2:
            # Dividir el dataset en conjuntos de entrenamiento y prueba
            X = wine_df.drop(['target', 'target_name'], axis=1)
            y = wine_df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

            # Crear el clasificador k-NN
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train, y_train)

            # Predecir las clases para el conjunto de prueba
            y_pred = knn.predict(X_test)

            # Calcular la precisión del modelo
            accuracy = accuracy_score(y_test, y_pred)
            print("Precisión del modelo en el conjunto de prueba:", accuracy)

            # Añadir predicciones al DataFrame de prueba para comparar
            X_test['Predicted Species'] = pd.Categorical.from_codes(y_pred, wine.target_names)
            X_test['True Species'] = pd.Categorical.from_codes(y_test, wine.target_names)

            # Imprimir los resultados
            print(X_test)
        elif winevalue == 3:
            # Configurar la validación cruzada de 10-Fold estratificada
            skf = StratifiedKFold(n_splits=10)

            # Crear el clasificador, k-NN en este caso
            knn = KNeighborsClassifier(n_neighbors=5)

            # Almacenar las puntuaciones de cada fold para luego calcular un promedio
            scores = []

            # Ejecutar la validación cruzada
            for train_index, test_index in skf.split(wine.data, wine.target):
                X_train, X_test = wine.data[train_index], wine.data[test_index]
                y_train, y_test = wine.target[train_index], wine.target[test_index]

                # Entrenar el modelo con el conjunto de entrenamiento del fold actual
                knn.fit(X_train, y_train)
                
                # Predecir el conjunto de prueba del fold actual
                y_pred = knn.predict(X_test)
                
                # Calcular la precisión del modelo en el fold actual
                accuracy = accuracy_score(y_test, y_pred)
                scores.append(accuracy)

            # Calcular la precisión media de todos los folds
            average_accuracy = np.mean(scores)

            print("Precisión promedio con 10-Fold Cross-Validation estratificada:", average_accuracy)
                    
        else:
            print("No se ha escogido ninguna opción")
    elif value == 3:
        cancer = load_breast_cancer()
        cancer_df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)

        # Añadir una columna con las etiquetas de las clases (benigno o maligno)
        cancer_df['target'] = cancer.target
        cancer_df['target_name'] = pd.Categorical.from_codes(cancer.target, cancer.target_names)

        cwvalue = int(input("Escoge una acción  1. mostrar datos cancer winsconsin, 2. HoldOut , 3. 10-Fold"))
        if cwvalue == 1:
            # Configurar pandas para mostrar más filas
            #pd.set_option('display.max_rows', None)  # None significa que no hay límite en el número de filas mostradas
            # Imprimir el DataFrame completo
            print(cancer_df)
        elif cwvalue == 2:
            # Dividir el dataset en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(
                cancer_df[cancer.feature_names], cancer.target, test_size=0.3, random_state=42, stratify=cancer.target)

            # Crear el clasificador k-NN y entrenarlo con el conjunto de entrenamiento
            knn = KNeighborsClassifier(n_neighbors=5)  # Puedes ajustar el número de vecinos si es necesario
            knn.fit(X_train, y_train)

            # Predecir las clases para el conjunto de prueba
            y_pred = knn.predict(X_test)

            # Crear un DataFrame para mostrar los datos de prueba y las predicciones
            results_df = pd.DataFrame(X_test, columns=cancer.feature_names)
            results_df['Predicted Class'] = pd.Categorical.from_codes(y_pred, cancer.target_names)
            results_df['True Class'] = pd.Categorical.from_codes(y_test, cancer.target_names)

            # Imprimir los resultados
            print(results_df)
        elif cwvalue == 3:
            # Configurar la validación cruzada de 10-Fold estratificada
            skf = StratifiedKFold(n_splits=10)

            # Crear el clasificador, k-NN en este caso
            knn = KNeighborsClassifier(n_neighbors=5)

            # Almacenar las puntuaciones de cada fold para luego calcular un promedio
            scores = []

            # Ejecutar la validación cruzada
            for train_index, test_index in skf.split(cancer.data, cancer.target):
                X_train, X_test = cancer.data[train_index], cancer.data[test_index]
                y_train, y_test = cancer.target[train_index], cancer.target[test_index]

                # Entrenar el modelo con el conjunto de entrenamiento del fold actual
                knn.fit(X_train, y_train)
                
                # Predecir el conjunto de prueba del fold actual
                y_pred = knn.predict(X_test)
                
                # Calcular la precisión del modelo en el fold actual
                accuracy = accuracy_score(y_test, y_pred)
                scores.append(accuracy)

            # Calcular la precisión media de todos los folds
            average_accuracy = np.mean(scores)

            print("Precisión promedio con 10-Fold Cross-Validation estratificada:", average_accuracy)
        else:
            print("No se ha escogido ninguna opción")
    else:
        print("No se ha escogido ninguna opción")
    

value = int(input("Escoja una de las opciones de Clases 1. Iris , 2. Vino , 3.Cancer Wisconsin"))
opciones(value)