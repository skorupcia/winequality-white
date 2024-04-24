#Author: Krzysztof Skorupski WZINIS
#Nr albumu: 71064

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report

df_wine = pd.read_csv("winequality-white.csv")

def informations():
    df_wine_row_count, df_wine_column_count=df_wine.shape
    print('Liczba wierszy:', df_wine_row_count)
    print('Liczba kolumn:', df_wine_column_count)

    print("\n", df_wine.info())
    print("\n", df_wine.head())

    #Brakujące wartości w pliku CSV
    print("\nBrakujące wartości: \n", df_wine.isna().sum())

    #Unikalne wartości
    print ("\nUnikalne wartości:\n", df_wine.nunique())

    #Typy danych
    print("\nTypy danych:\n", df_wine.dtypes)

def analyze():
    #Mapa korelacji danych
    f,ax = plt.subplots(figsize=(9, 9))
    ax.set_title('Mapa korelacji danych')
    sns.heatmap(df_wine.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax,cmap="PuRd")
    plt.show()

    #Ilość win podzielona na ocene jakosci
    sns.countplot(data=df_wine, x='jakosc')
    plt.show()

    #Dokładna ilość win podzielona na ocene jakosci
    print("\nPodział jakości na punkty: ", df_wine.jakosc.unique())
    print("\nIlość win w poszczególnych ocenach jakości: ", df_wine.jakosc.value_counts())

    #Korelacja
    plt.figure(figsize=(6,4))
    sns.heatmap(df_wine.corr(),cmap='Blues',annot=False)
    plt.show()

    #Korelacja jakości
    cols = df_wine.corr().nlargest(12, 'jakosc')['jakosc'].index
    cm = df_wine[cols].corr()
    plt.figure(figsize=(10,6))
    sns.heatmap(cm, annot=True, cmap = 'viridis')
    plt.show()

    ###### 2 #####
    # Przygotowanie danych do analizy oraz podział danych na zbiór uczący i testowy
    # Podział danych na cechy i etykiety
    X = df_wine.drop(columns=['jakosc'])
    y = df_wine['jakosc']

    # Podział danych (80% uczący, 20% testowy)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Wyświetlenie rozmiaru zbiorów uczącego i testowego
    print("\nRozmiar zbioru uczącego:", X_train.shape[0])
    print("\nRozmiar zbioru testowego:", X_test.shape[0])

    # Wykresy
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(df_wine.columns[:-1]):
        plt.subplot(3, 4, i+1)
        sns.histplot(df_wine[col], kde=True)
        plt.title(col)
    plt.tight_layout()
    plt.show()

    ##### 3 #####
    # Model decyzyjny i uczący
    # Inicjalizacja modelu drzewa decyzyjnego
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Obliczanie dokładności modelu
    accuracy = accuracy_score(y_test, y_pred)
    print("\nDokładność modelu na zbiorze testowym:", accuracy)

    ##### 4 #####
    # Model regresji liniowej
    # Inicjalizacja modelu regresji liniowej
    model_regresji = LinearRegression()

    # Trenowanie modelu na danych uczących
    model_regresji.fit(X_train, y_train)

    # Przewidywanie jakości wina na zbiorze testowym
    y_pred_regresja = model_regresji.predict(X_test)

    # Zaokrąglanie przewidywanych wartości
    y_pred_regresja_rounded = np.round(y_pred_regresja)

    # Obliczanie błędu średniokwadratowego
    mse = mean_squared_error(y_test, y_pred_regresja)
    print("Błąd średniokwadratowy (MSE):", mse)

    # Ocena jakości modelu
    accuracy_regresja = accuracy_score(y_test, y_pred_regresja_rounded)
    print("Dokładność modelu na zbiorze testowym (zaokrąglona):", accuracy_regresja)

    ##### 5 #####
    # Model klasyfikacyjny
    print("Model klasyfikacyjny:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Model regresji
    print("Model regresji:")
    print(classification_report(y_test, y_pred_regresja_rounded, zero_division=0))

    # Klasyfikator
    class_names = np.unique(y_test)
    report_clf = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    precision_clf = [report_clf[str(cls)]['precision'] for cls in class_names]
    recall_clf = [report_clf[str(cls)]['recall'] for cls in class_names]
    f1_score_clf = [report_clf[str(cls)]['f1-score'] for cls in class_names]

    # Regresja
    report_reg = classification_report(y_test, y_pred_regresja_rounded, zero_division=0, output_dict=True)
    precision_reg = [report_reg[str(cls)]['precision'] for cls in class_names]
    recall_reg = [report_reg[str(cls)]['recall'] for cls in class_names]
    f1_score_reg = [report_reg[str(cls)]['f1-score'] for cls in class_names]

    # Indeksy klas
    indices = np.arange(len(class_names))

    # Szerokość słupków
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    # Słupki dla klasyfikatora
    bar1 = ax.bar(indices - bar_width/2, precision_clf, bar_width, label='Precyzja (klasyfikator)')
    bar2 = ax.bar(indices + bar_width/2, recall_clf, bar_width, label='Czułość (klasyfikator)')

    # Słupki dla regresji
    bar3 = ax.bar(indices - bar_width/2, precision_reg, bar_width, label='Precyzja (regresja)', alpha=0.5)
    bar4 = ax.bar(indices + bar_width/2, recall_reg, bar_width, label='Czułość (regresja)', alpha=0.5)

    # Opisy na osi X
    ax.set_xticks(indices)
    ax.set_xticklabels(class_names)
    ax.set_xlabel('Klasy')
    ax.set_ylabel('Wartość')
    ax.set_title('Porównanie precyzji i czułości dla klasyfikatora i regresji')
    ax.legend()
    plt.show()

informations()
analyze()