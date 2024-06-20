import bnlearn as bn
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Función para cargar y discretizar los datos
def load_and_discretize_data(load_function, target_name, n_bins=5):
    data = load_function()
    X = data.data
    y = data.target

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X_discretized = discretizer.fit_transform(X)

    feature_names = [f'feature{i+1}' for i in range(X_discretized.shape[1])]
    df = pd.DataFrame(X_discretized, columns=feature_names)
    df[target_name] = y
    return df, feature_names

# Cargar y discretizar los conjuntos de datos
iris_data, iris_features = load_and_discretize_data(load_iris, 'target')
wine_data, wine_features = load_and_discretize_data(load_wine, 'target')
cancer_data, cancer_features = load_and_discretize_data(load_breast_cancer, 'target')

datasets = {
    "Iris": (iris_data, iris_features),
    "Wine": (wine_data, wine_features),
    "Breast Cancer": (cancer_data, cancer_features)
}

# Función para evaluar la red bayesiana
def evaluate_bayesian_network(name, data, features):
    # Dividir el conjunto de datos en entrenamiento y prueba
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['target'])

    # Crear la estructura del DAG con el número adecuado de características
    dag = bn.make_DAG([(feature, 'target') for feature in features])

    # Entrenar el modelo con los datos de entrenamiento
    model = bn.parameter_learning.fit(dag, train_data)

    # Realizar predicciones para los datos de prueba
    y_pred = []
    for _, row in test_data.iterrows():
        evidence = row.drop('target').to_dict()
        query_result = bn.inference.fit(model, variables=['target'], evidence=evidence)
        y_pred.append(query_result.values.argmax())

    # Evaluar el desempeño
    y_true = test_data['target']
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"{name} - Bayesian Network")
    print(f"Accuracy: {accuracy}")
    ConfusionMatrixDisplay(conf_matrix, display_labels=np.unique(data['target'])).plot()
    plt.title(f"{name} - Bayesian Network - Confusion Matrix")
    plt.show()

# Evaluar la red bayesiana en cada conjunto de datos
for name, (data, features) in datasets.items():
    evaluate_bayesian_network(name, data, features)

# Función para realizar validación cruzada con Redes Bayesianas
def cross_validate_bayesian_network(name, data, features):
    skf = StratifiedKFold(n_splits=10)
    accuracies = []
    conf_matrices = []

    for train_index, test_index in skf.split(data.drop(columns=['target']), data['target']):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        # Crear la estructura del DAG con el número adecuado de características
        dag = bn.make_DAG([(feature, 'target') for feature in features])

        # Entrenar el modelo con los datos de entrenamiento
        model = bn.parameter_learning.fit(dag, train_data)
        infer = bn.inference.fit(model)

        # Realizar predicciones
        y_pred = []
        for _, row in test_data.iterrows():
            evidence = row.drop('target').to_dict()
            query_result = infer.map_query(variables=['target'], evidence=evidence)
            y_pred.append(query_result.values.argmax())

        # Evaluar el desempeño
        y_true = test_data['target']
        accuracies.append(accuracy_score(y_true, y_pred))
        conf_matrices.append(confusion_matrix(y_true, y_pred))

    # Promedio de precisión
    mean_accuracy = np.mean(accuracies)
    print(f"{name} - Mean Accuracy: {mean_accuracy}")

    # Promedio de la matriz de confusión
    mean_conf_matrix = np.mean(conf_matrices, axis=0)
    print(f"{name} - Mean Confusion Matrix:")
    print(mean_conf_matrix)
    ConfusionMatrixDisplay(mean_conf_matrix, display_labels=np.unique(data['target'])).plot()
    plt.title(f"{name} - Bayesian Network - Mean Confusion Matrix")
    plt.show()

# Realizar validación cruzada en cada conjunto de datos
for name, (data, features) in datasets.items():
    cross_validate_bayesian_network(name, data, features)
