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

print(dados_final)

X = dados_final.drop(columns=['Class_recurrence-events', 'Class_no-recurrence-events'])
y = dados_final['Class_recurrence-events']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


print("Antes do balanceamento:")
print(y.value_counts())

print("\nDepois do balanceamento:")
print(pd.Series(y_resampled).value_counts())

dado_classes = y_resampled
dado_atributos = X_resampled

tree = DecisionTreeClassifier()
atr_train, atr_test, class_train, class_test  = train_test_split(dado_atributos, dado_classes, test_size=0.3)
fertility_tree = tree.fit(atr_train, class_train)
Class_predict = fertility_tree.predict(atr_test)
print('Treinamento')
print(Class_predict)

acuracia = accuracy_score(class_test, Class_predict)
taxa_erro = 1 - acuracia

print(f'Acurácia: {acuracia:.2f}')
print(f'Taxa de Erro: {taxa_erro:.2f}')

cm = confusion_matrix(class_test, Class_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= fertility_tree.classes_)
disp.plot()
plt.show() 







