# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:16:31 2020

@author: Daniel Alves
"""

import math as mt

#função para calcular o valor de y
def equacao (x, a):
    #resultado para primeira raiz
    p_raiz = int(mt.sqrt(a**3*(a+2*x)))
    
    #calculo do valor de y
    y = 2 * p_raiz + 2*a**2 + 2*a*x - x**2
    
    #caso o número calculado seja negativo
    y = mt.sqrt(abs(y))

    return int(y)

