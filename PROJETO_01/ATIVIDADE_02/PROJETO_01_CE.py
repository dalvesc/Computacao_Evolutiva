# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 18:52:01 2020

@author: Daniel Alves
"""
import numpy as np


################################### VALORES DE X ###################################
import random


#lista com valores para X
x = []

#adicionando valores a X aleatoriamente
for i in range(100):
    x.append(random.randint(1, 100))


################################### EQUAÇÃO PARA ENCONTRAR VALOR DE Y ###################################
from EQUACAO_CARDIOIDE import equacao


#valor constante 'a' que é necessário na equação 
a = 2

#lista com valores para Y
y = []

for i in range(len(x)):
    y.append(equacao(x[i], a))

x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)


################################### REGRESSÃO DA REDE ###################################
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


#dividindo dados em teste e treinamento 
# Usou-se 25%(test_size = 0.25) como quantidade de atributos para teste e o restante para treinamento
previsores_treinamento, previsores_teste, previsao_treinamento, previsao_teste = train_test_split(x, y, test_size=0.25, random_state=0)

#atribuindo a função a uma variável para ser utilizada
regressor = MLPRegressor(random_state=0, max_iter=200, solver="lbfgs")

#treinando o modelo com os valores separados para treinamento
regressor.fit(previsores_treinamento, previsao_treinamento)

#realizando previsão usando o regressor perceptron multicamadas
resultados = regressor.predict(previsores_teste)

#precisão(acurácia) média nos dados de teste
regressor.score(previsores_teste, previsao_teste)

#exibir número de iteração que foi executada
print(regressor.n_iter_)

#exibir número de camadas
print(regressor.n_layers_)

#exibir função de ativação de saída
print(regressor.out_activation_)


################################### GRÁFICO PARA EXIBIÇÃO ###################################
import matplotlib.pyplot as plt


#armazenando valores obtidos na previsão
r = []

for i in resultados:
    r.append(int(i))

#inserindo no grafico
plt.title('Predições vs Valores reais')
plt.plot(r) #azul - previsão

plt.ylabel('Valores de y')
plt.legend(loc='best')
plt.plot(previsao_teste[:25]) #amarelo - reais