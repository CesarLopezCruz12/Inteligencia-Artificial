import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Función para obtener el mejor K para K-NN
def get_best_k(X, y, k_values, cv):
    best_k = 0
    best_score = 0
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
    return best_k, best_score

# Función para entrenar y evaluar el modelo
def evaluate_model(X, y, classifier, cv):
    # Hold-Out 70/30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy_holdout = accuracy_score(y_test, y_pred)
    confusion_holdout = confusion_matrix(y_test, y_pred)

    # 10-Fold Cross Validation
    scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
    accuracy_cv = scores.mean()

    return accuracy_holdout, confusion_holdout, accuracy_cv

# Cargar los conjuntos de datos
datasets = {
    'Iris': load_iris(),
    'Wine': load_wine(),
    'Breast Cancer': load_breast_cancer()
}

results = []

# Parámetros
k_values = range(1, 31)
cv = StratifiedKFold(n_splits=10)

for name, dataset in datasets.items():
    X, y = dataset.data, dataset.target
    
    # Clasificador 1-NN
    classifier_1nn = KNeighborsClassifier(n_neighbors=1)
    acc_holdout_1nn, conf_holdout_1nn, acc_cv_1nn = evaluate_model(X, y, classifier_1nn, cv)
    
    # Clasificador K-NN
    best_k, _ = get_best_k(X, y, k_values, cv)
    classifier_knn = KNeighborsClassifier(n_neighbors=best_k)
    acc_holdout_knn, conf_holdout_knn, acc_cv_knn = evaluate_model(X, y, classifier_knn, cv)
    
    # Guardar resultados
    results.append({
        'Dataset': name,
        'Classifier': '1-NN',
        'Holdout Accuracy': acc_holdout_1nn,
        'Holdout Confusion Matrix': conf_holdout_1nn,
        'CV Accuracy': acc_cv_1nn
    })
    
    results.append({
        'Dataset': name,
        'Classifier': f'K-NN (K={best_k})',
        'Holdout Accuracy': acc_holdout_knn,
        'Holdout Confusion Matrix': conf_holdout_knn,
        'CV Accuracy': acc_cv_knn
    })

# Crear un DataFrame para mostrar los resultados
df_results = pd.DataFrame(results)

# Mostrar resultados con los mejores desempeños resaltados
best_results = df_results.loc[df_results.groupby(['Dataset', 'Classifier'])['Holdout Accuracy'].idxmax()]
best_results_cv = df_results.loc[df_results.groupby(['Dataset', 'Classifier'])['CV Accuracy'].idxmax()]

# Imprimir los resultados en la consola
print("Resultados Hold-Out:")
print(best_results)

print("\nResultados 10-Fold Cross Validation:")
print(best_results_cv)

# Graficar el desempeño del modelo en función de K para uno de los datasets
dataset_name = 'Iris'
dataset = datasets[dataset_name]
X, y = dataset.data, dataset.target
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
    cv_scores.append(scores.mean())

plt.plot(k_values, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Cross-Validated Accuracy')
plt.title(f'KNN Varying number of neighbors (Dataset: {dataset_name})')
plt.show()
