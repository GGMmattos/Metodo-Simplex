import numpy as np
import math

np.seterr(divide='ignore', invalid='ignore')
tipo = ('min', 'max')

A = np.array([[1, 2, -2, '<='], [2, 0, -2, '<='], [2, -1, 2, '<=']])
b = np.array([4, 6, 2])
c = np.array([1, -2, 1])

problema = str(input('O problema é de MAX ou MIN? '))


def simplex(A, b, c, problema):
    m, n = A.shape  # "m" número de restrições, "n" número de variáveis
    matriz_aux = np.eye(m)
    c = np.hstack([c, np.zeros(m)])  # "c" contém os custos

    for i in range(0, m, 1):
        if '>=' in A[i]:
            matriz_aux[i] = np.dot(matriz_aux[i], -1)  # Matriz que contém os valor de folga - excesso

    A = np.delete(A, n-1, axis=1)  # Excluir a coluna que contém os sinais.
    A = np.hstack([A, matriz_aux])
    A = A.astype(float)
    n = n-1
    var = np.arange(n + m)  # var = [0, 1, ..., n + m - 1] -- portanto nao ficara na ordem correta... realizar a correcao posteiormente
    vb = var[n:]  # vb = [n, n + 1, ..., n + m - 1] ('vb' variáveis da base)
    vnb = var[:n]  # vnb = [0, 1, ..., n - 1] ('vnb' variáveis não base)

    while True:
        if problema.lower() in tipo:
            if problema.lower() == 'max':
                c = np.dot(c, -1)
                break
            elif problema.lower() == 'min':
                break
        else:
            problema = str(input('Opção incorreta...MAX ou MIN? '))

    while True:

        B = np.linalg.inv(A[:, vb])  # cálculo da inversa de B -  A[:, vb] contém a matriz dos elementos da base...

        # cálculo da solução básica factível
        sbf = np.dot(B, b)

        # Calcular os custos relativos das variáveis não básicas
        Pt = np.dot(c[vb], B)
        cnb = c[vnb] - np.dot(Pt, A[:, vnb])  # "cnb" cusos da variáveis não base

        # Verificar se a solução atual é ótima
        # Se todos os custos relativos são maiores ou iguais a zero, a solução é ótima
        if np.all(cnb >= 0):  # Se todos os custos de cnb forem maior que 0 o laço é encerrado.
            print('\nSolução ótima encontrada!!!\n')
            break

        # Escolha da variável que entra na base
        # É aquela que tem o menor custo relativo negativo (mais negativo)
        k = np.argmin(cnb)  # k é a posição da variável que entra na base no vetor vnb
        # xk = cnb[k] # xk é o índice da variável que entra na baase

        # Teste de razão
        y = np.dot(B, A[:, k])
        yA = np.dot(B, b) / y

        # Verificar se o problema tem solução limitada
        # Se todos os coeficientes da reta são menores ou iguais a zero, o problema é ilimitado
        if np.all(yA <= 0):
            print("Problema ilimitado.")
            return None

        # Escolha da variável que sai da base
        y_pos = yA[yA > 0]  # y_pos é o vetor dos coeficientes positivos da reta -- tratar valores infinitos caso der tempo
        y_min = np.argmin(y_pos)  # y_min é a posição da variável que sai da base no vetor y_pos (indice do valor mínimo)

        il = np.where(yA == y_pos[y_min])[0][0]  # il é a posição da variável que sai da base no vetor
        xl = vb[il]  # xl é o índice da variável que sai da base (vale a pena ressaltar [0...-n])

        # Atualizar a solução básica
        vb[il] = k  # A variável que entra na base ocupa o lugar da que sai
        vnb[k] = xl  # A variável que sai da base ocupa o lugar da que entra

    # Retornar a solução ótima
    # A solução ótima é dada pelos valores das variáveis básicas e pelo valor da função objetivo
    x = np.zeros(n + m)  # x é o vetor da solução ótima
    x[vb] = sbf  # As variáveis básicas recebem os valores dos termos independentes
    for i, valor in enumerate(x, start=1):
        print(f"X{i}= %.2f" % valor)

    z = c[-1] + np.dot(c[vb], sbf)  # z é o valor ótimo da função objetivo
    print("\nValor otimo: %.2f" % z)


simplex(A, b, c)