import numpy as np
import math
"""
Problema de programação linear de exemplo
 min z = x1 - 2x2 + x3
 s.a.
 x1  + 2x2 - 2x3 <= 4
 2x1 + 0x2 - 2x3 <= 6
 2x1 - x2  + 2x3 <= 2
 x1, x2, x3 >= 0
"""
A = np.array([[1, 2, -2,], [2, 0, -2], [2, -1, 2]]) # Matriz dos coeficientes das restrições
b = np.array([4, 6, 2]) # valores de "b", após a igualdade
c = np.array([1, -2, 1]) # Vetor dos coeficientes da função objetivo, será tratado posteriormente casos de MAX

def simplex(A, b, c):

    m, n = A.shape #"m" número de restrições, "n" número de variáveis
    A = np.hstack([A, np.eye(m, dtype=int)])
    c = np.hstack([c, np.zeros(m)])

    var = np.arange(n + m) # var = [0, 1, ..., n + m - 1] -- portanto nao ficara na ordem correta... realizar a correcao posteiormente

    vb = var[n:] # vb = [n, n + 1, ..., n + m - 1]
    vnb = var[:n] # vnb = [0, 1, ..., n - 1]

    while True:
        #cálculo da solução básica factível
        sbf = np.dot(A[:, vb], b)

        # Calcular os custos relativos das variáveis não básicas
        cnb = c[vnb] - np.dot(c[vb], A[:, vnb])

        # Verificar se a solução atual é ótima
        # Se todos os custos relativos são maiores ou iguais a zero, a solução é ótima
        if  np.all(cnb >= 0):
            print('Solução ótima encontrada')
            break

        # Escolha da variável que entra na base
        # É aquela que tem o menor custo relativo negativo (mais negativo)
        k = np.argmin(cnb) #k é a posição da variável que entra na base no vetor vnb
        xk = cnb[k] # xk é o índice da variável que entra na baase

        #Teste de razão
        y = np.dot(A[:, vb], A[:, k])

        #print(k) # posicao que entra na base
        #print(xk) # valor que entra na base


        # Calcular os coeficientes da equação da reta que sai da solução atual em direção à melhora da função objetivo¬
        yA = np.dot(A[:, vb], b) / A[:, k]

        # Verificar se o problema tem solução limitada
        # Se todos os coeficientes da reta são menores ou iguais a zero, o problema é ilimitado
        if np.all(y <= 0):
            print("Problema ilimitado.")
            return None

        #Escolha da variável que sai da base
        y_pos = y[y > 0] # y_pos é o vetor dos coeficientes positivos da reta
        y_min = np.argmin(y_pos) # y_min é a posição da variável que sai da base no vetor y_pos (indice do valor mínimo)

        il = np.where(y == y_pos[y_min])[0][0] # il é a posição da variável que sai da base no vetor           
        xl = vb[il] # xl é o índice da variável que sai da base (vale a pena ressaltar [0...-n])

        # Atualizar a solução básica
        vb[il] = k # A variável que entra na base ocupa o lugar da que sai
        vnb[k] = xl # A variável que sai da base ocupa o lugar da que entra

        print(vb)
        print(vnb)




        break

simplex(A, b, c)