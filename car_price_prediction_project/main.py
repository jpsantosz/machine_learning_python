# Importando as dependências 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

# Carrega um arquivo CSV com informações de carros
car_dataset = pd.read_csv('C:/Users/user/Documents/machine_learning_python/car_price_prediction_project/car data.csv')

#Substitui valores categóricos em variáveis para que possam ser processados pelo modelo
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
car_dataset.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

"""
X: Contém todas as colunas exceto:
    Car_Name: Nome do carro (não relevante para previsão).
    Selling_Price: O preço que queremos prever.
y: A variável alvo, ou seja, o preço de venda.
"""
x = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
y = car_dataset['Selling_Price']

"""
train_test_split divide os dados em:
    x_train, y_train: Dados de treino (90%).
    x_test, y_test: Dados de teste (10%).
"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

ling_reg_model = LinearRegression()
ling_reg_model.fit(x_train, y_train)

# O modelo prediz preços com base nos dados de treino.
training_data_prediction = ling_reg_model.predict(x_train)

error_score = metrics.r2_score(y_train, training_data_prediction)

plt.scatter(y_train, training_data_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title("Actual prices vs Prediceted prices")
plt.show()