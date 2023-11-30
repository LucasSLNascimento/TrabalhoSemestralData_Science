import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from pickle import dump
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


dados = pd.read_csv('C:/Users/lucas/OneDrive/Documentos/Estudos/Faculdade/Semestre_6/Data_Science/TrabalhoSemestralData_Science/breast-cancer.csv', sep=',')

dados.replace('?', pd.NaT, inplace=True)
dados.fillna(dados.mode().iloc[0], inplace=True)

dados_cat = dados.drop(columns=['deg-malig'])
dados_num = dados['deg-malig']

dados_num = dados_num.values.reshape(-1, 1)

dados_cat_normalizado = pd.get_dummies(data=dados_cat)

normalizador = preprocessing.MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_num)
dados_num_normalizado = modelo_normalizador.transform(dados_num)

dados_final = pd.DataFrame(data=dados_num_normalizado, columns=['deg-malig'])
dados_final = dados_final.join(dados_cat_normalizado, how='left')

# print(dados_final)

X = dados_final.drop(columns=['Class_recurrence-events', 'Class_no-recurrence-events'])
y = dados_final['Class_recurrence-events']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# print("Antes do balanceamento:")
# print(y.value_counts())

# print("\nDepois do balanceamento:")
# print(pd.Series(y_resampled).value_counts())







