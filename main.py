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
A = np.array([[1, 2, -2], [2, 0, -2], [2, -1, 2]]) # Matriz dos coeficientes das restrições
b = np.array([4, 6, 2]) # valores de "b", após a igualdade
c = np.array([1, -2, 1]) # Vetor dos coeficientes da função objetivo, será tratado posteriormente casos de MAX

def simplex(A, b, c):

    m, n = A.shape #"m" número de restrições, "n" número de variáveis
    A = np.hstack([A, np.eye(m, dtype=int)]) #cria matriz identidade das variáveis de folga
    c = np.hstack([c, np.zeros(m)]) # "c" contém os custos

    var = np.arange(n + m) # var = [0, 1, ..., n + m - 1] -- portanto nao ficara na ordem correta... realizar a correcao posteiormente

    vb = var[n:] # vb = [n, n + 1, ..., n + m - 1] ('vb' variáveis da base)
    vnb = var[:n] # vnb = [0, 1, ..., n - 1] ('vnb' variáveis não base)

    while True:

        B = np.linalg.inv (A[:, vb]) #cálculo da inversa de B -  A[:, vb] contém a matriz dos elementos da base...

        #cálculo da solução básica factível
        sbf = np.dot(B, b)

        # Calcular os custos relativos das variáveis não básicas
        Pt = np.dot(c[vb], B)    
        cnb = c[vnb] - np.dot(Pt, A[:, vnb]) # "cnb" cusos da variáveis não base

        # Verificar se a solução atual é ótima
        # Se todos os custos relativos são maiores ou iguais a zero, a solução é ótima
        if np.all(cnb >= 0): #Se todos os custos de cnb forem maior que 0 o laço é encerrado.
            print('Solução ótima encontrada')
            break

        # Escolha da variável que entra na base
        # É aquela que tem o menor custo relativo negativo (mais negativo)
        k = np.argmin(cnb) #k é a posição da variável que entra na base no vetor vnb
        #xk = cnb[k] # xk é o índice da variável que entra na baase

        #Teste de razão
        y = np.dot(B, A[:, k])

        # Calcular os coeficientes da equação da reta que sai da solução atual em direção à melhora da função objetivo¬
        try:
            yA = np.dot(B, b) / y
        except ZeroDivisionError:
            pass

        # Verificar se o problema tem solução limitada
        # Se todos os coeficientes da reta são menores ou iguais a zero, o problema é ilimitado
        if np.all(yA <= 0):
            print("Problema ilimitado.")
            return None

        #Escolha da variável que sai da base
        y_pos = yA[yA > 0] # y_pos é o vetor dos coeficientes positivos da reta -- tratar valores infinitos caso der tempo
        y_min = np.argmin(y_pos) # y_min é a posição da variável que sai da base no vetor y_pos (indice do valor mínimo)


        il = np.where(yA == y_pos[y_min])[0][0] # il é a posição da variável que sai da base no vetor           
        xl = vb[il] # xl é o índice da variável que sai da base (vale a pena ressaltar [0...-n])

        # Atualizar a solução básica
        vb[il] = k# A variável que entra na base ocupa o lugar da que sai
        vnb[k] = xl# A variável que sai da base ocupa o lugar da que entra

    # Retornar a solução ótima
    # A solução ótima é dada pelos valores das variáveis básicas e pelo valor da função objetivo
    x = np.zeros(n + m) # x é o vetor da solução ótima
    print(x)
    x[vb] = sbf # As variáveis básicas recebem os valores dos termos independentes
    z = c[-1] + np.dot(c[vb], sbf) # z é o valor ótimo da função objetivo
    return x[:n], z # Retornar apenas as variáveis originais e o valor ótimo


simplex(A, b, c)

