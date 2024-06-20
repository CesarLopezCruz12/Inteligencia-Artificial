import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Cargar los conjuntos de datos
datasets = {
    "Iris": load_iris(),
    "Wine": load_wine(),
    "Breast Cancer": load_breast_cancer()
}

# Inicializar el clasificador Naïve Bayes
naive_bayes = GaussianNB()

# Función para evaluar el modelo
def evaluate_model(dataset_name, data, target):
    # Hold-Out 70/30
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, stratify=target, random_state=42)
    
    naive_bayes.fit(X_train, y_train)
    y_pred = naive_bayes.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"{dataset_name} - Naïve Bayes - Hold-Out 70/30")
    print(f"Accuracy: {accuracy}")
    ConfusionMatrixDisplay(conf_matrix, display_labels=np.unique(target)).plot()
    
    # 10-Fold Cross Validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(naive_bayes, data, target, cv=skf)
    
    print(f"{dataset_name} - Naïve Bayes - 10-Fold Cross Validation")
    print(f"Accuracy: {cv_scores.mean()} (+/- {cv_scores.std() * 2})")

# Evaluar los modelos en cada conjunto de datos
for name, dataset in datasets.items():
    evaluate_model(name, dataset.data, dataset.target)
