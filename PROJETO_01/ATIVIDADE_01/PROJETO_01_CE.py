# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:45:20 2020

@author: Daniel Alves
"""

import pandas as pd

################################### LEITURA DATASET ###################################
#leitura do arquivo .csv
base_cancer = pd.read_csv('./data/breast-cancer.csv')

#nomeando as colunas
base_cancer.rename( columns={'1000025' :'id'}, inplace=True ) 
base_cancer.rename( columns={'5' :'Clump_Thickness'}, inplace=True )  
base_cancer.rename( columns={'1' :'Cell_Size'}, inplace=True )    
base_cancer.rename( columns={'1.1' :'Cell_Shape'}, inplace=True )    
base_cancer.rename( columns={'1.2' :'Marginal_Adhesion'}, inplace=True )    
base_cancer.rename( columns={'2' :'Epithelial_Cell_Size'}, inplace=True )  
base_cancer.rename( columns={'1.3' :'Bare_Nuclei'}, inplace=True )    
base_cancer.rename( columns={'3' :'Bland_Chromatin'}, inplace=True )
base_cancer.rename( columns={'1.4' :'Normal_Nucleoli'}, inplace=True )
base_cancer.rename( columns={'1.5' :'Mitoses'}, inplace=True )
base_cancer.rename( columns={'2.1' :'Class'}, inplace=True )   


base_cancer.loc[1000025] = [1000025,5,1,1,1,2,'1',3,1,1,2] #adicionando a linha que foi considerada como atributos

#removendo a coluna id e a colocando como index da base
base_cancer.index = base_cancer['id']
base_cancer = base_cancer.drop(columns=['id'])

base_cancer = base_cancer.sort_index() #ordenando crescentemente 


################################### TRATAMENTO DE DADOS ###################################
import random

#verificação de valores 'NaN'
base_cancer.isnull().sum()

#verificação de valores de cada coluna, verificando valores ausentes
counts = base_cancer['Clump_Thickness'].value_counts().to_dict()
print('Clump_Thickness - ', counts)
counts = base_cancer['Cell_Size'].value_counts().to_dict()
print('Cell_Size - ', counts)
counts = base_cancer['Cell_Shape'].value_counts().to_dict()
print('Cell_Shape - ', counts)
counts = base_cancer['Marginal_Adhesion'].value_counts().to_dict()
print('Marginal_Adhesion - ', counts)
counts = base_cancer['Epithelial_Cell_Size'].value_counts().to_dict()
print('Epithelial_Cell_Size - ', counts)

counts = base_cancer['Bare_Nuclei'].value_counts().to_dict()
print('Bare_Nuclei - ', counts)

counts = base_cancer['Bland_Chromatin'].value_counts().to_dict()
print('Bland_Chromatin - ', counts)
counts = base_cancer['Normal_Nucleoli'].value_counts().to_dict()
print('Normal_Nucleoli - ', counts)
counts = base_cancer['Mitoses'].value_counts().to_dict()
print('Mitoses - ',counts)
counts = base_cancer['Class'].value_counts().to_dict()
print('Class - ', counts)

#verificando quais id possuem '?' em Bare_Nuclei
print(base_cancer.loc[base_cancer.Bare_Nuclei == '?', 'Bare_Nuclei'])


#tratando valores '?' em Bare_Nuclei
base_cancer.loc[61634, 'Bare_Nuclei'] = random.randint(1, 10)
base_cancer.loc[169356, 'Bare_Nuclei'] = random.randint(1, 10)
base_cancer.loc[432809, 'Bare_Nuclei'] = random.randint(1, 10)
base_cancer.loc[563649, 'Bare_Nuclei'] = random.randint(1, 10)
base_cancer.loc[606140, 'Bare_Nuclei'] = random.randint(1, 10)
base_cancer.loc[704168, 'Bare_Nuclei'] = random.randint(1, 10)
base_cancer.loc[733639, 'Bare_Nuclei'] = random.randint(1, 10)
base_cancer.loc[1057013, 'Bare_Nuclei'] = random.randint(1, 10)
base_cancer.loc[1057067, 'Bare_Nuclei'] = random.randint(1, 10)
base_cancer.loc[1096800, 'Bare_Nuclei'] = random.randint(1, 10)
base_cancer.loc[1183246, 'Bare_Nuclei'] = random.randint(1, 10)
base_cancer.loc[1184840, 'Bare_Nuclei'] = random.randint(1, 10)
base_cancer.loc[1193683, 'Bare_Nuclei'] = random.randint(1, 10)
base_cancer.loc[1197510, 'Bare_Nuclei'] = random.randint(1, 10)
base_cancer.loc[1238464, 'Bare_Nuclei'] = random.randint(1, 10)
base_cancer.loc[1241232, 'Bare_Nuclei'] = random.randint(1, 10)

#transformando os valores de 'Bare_Nuclei' em números(antes estavam como Strings)
base_cancer.Bare_Nuclei = pd.to_numeric(base_cancer.Bare_Nuclei)

#atributo que será previsto(o resultado se o paciente tem ou não cancer)
previsao = base_cancer.iloc[:,9].values

for i in range(len(previsao)):
    if previsao[i] >= 3:
        previsao[i] = 1
    else:
        previsao[i] = 0

#excluindo o que será previsto da base de dados
base_cancer = base_cancer.drop(columns='Class')

#valores para realizar a previsão
previsores = base_cancer.iloc[:,:].values


################################### CLASSIFICAÇÃO DOS DADOS ###################################

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


#dividindo dados em teste e treinamento 
# Usou-se 25%(test_size = 0.25) como quantidade de atributos para teste e o restante para treinamento
previsores_treinamento, previsores_teste, previsao_treinamento, previsao_teste = train_test_split(previsores, previsao, test_size=0.25, random_state=0)

#atribuindo a função a uma variável para ser utilizada
classificador = MLPClassifier(random_state=0, max_iter=300)

#treinando o modelo com os valores separados para treinamento
classificador.fit(previsores_treinamento, previsao_treinamento)

#realizando previsão usando o classificador perceptron multicamadas
classificador.predict_proba(previsores_teste)

#precisão média nos dados de teste
classificador.score(previsores_teste, previsao_teste)

#número de iterações que o classificador executou
print(classificador.n_iter_)

#número de camadas utilizadas
print(classificador.n_layers_)

#função de ativação de saída
print(classificador.out_activation_)
