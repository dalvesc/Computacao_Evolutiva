# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 18:52:01 2020

@author: Daniel Alves
"""


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
