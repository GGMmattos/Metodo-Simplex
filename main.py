import os
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

tipo = ('min', 'max')

A = np.array([[0.5, 0.3], [0.1, 0.2], [0.4, 0.5]]) # A Contém as restrições do problema
b = np.array([3, 1, 3])
c = np.array([-3, -2]) # "c" contém os custos da função objetivo

def simplex(A, b, c, problema):
    m, n = A.shape  # "m" número de restricoes, "n" número de veriáveis na F.O

    A = np.hstack([A, np.eye(m, dtype=int)])  # cria matriz identidade
    c = np.hstack([c, np.zeros(m)])  # "c" contém os custos
    var = np.arange(n + m)  # var = [0, 1, ..., n + m - 1]
    vb = var[n:]  # vb = [n, n + 1, ..., n + m - 1] ('vb' variáveis da base)
    vnb = var[:n]  # vnb = [0, 1, ..., n - 1] ('vnb' variáveis não base) --

    while True: #Verifica se o problema é de MAX ou MIN, se MAX multiplica os custos da F.O por -1.
        if problema.lower() in tipo:
            if problema.lower() == 'max':
                c = np.dot(c, -1)
                break
            elif problema.lower() == 'min':
                break
        else:
            problema = str(input('Opção incorreta...MAX ou MIN? '))

    while True:
        #Verifica se a matriz é invertível.
        det = np.linalg.det(A[:, vb]) # Calculando o determinante

        if det != 0:
            pass
        else:
            print("A matriz não é invertível")
            break

        B = np.linalg.inv(A[:, vb])  # cálculo da inversa de B -  A[:, vb] contém a matriz dos elementos da base...

        # cálculo da solução básica factível
        sbf = np.dot(B, b)

        # Calculo dos custos relativos das variáveis não base.
        Pt = np.dot(c[vb], B)
        cnb = c[vnb] - np.dot(Pt, A[:, vnb])  # "cnb" custos das variáveis não base.

        # Verificar se a solução atual é ótima
        # Se todos os custos relativos são maiores ou iguais a zero, a solução é ótima
        if np.all(cnb >= 0):  # Se todos os custos de cnb forem maior que 0 o laço é encerrado.
            print('\nSolução ótima encontrada!!!\n')
            break

        # Escolha da variável que entra na base
        # É aquela que tem o menor custo relativo negativo (mais negativo)
        k = np.argmin(cnb)  # k é a posição da variável que entra na base no vetor vnb

        # Teste da razão
        y = np.dot(B, A[:, k])
        y = np.dot(B, b) / y

        # Verificar se o problema tem solução limitada
        if np.all(y <= 0):
            print("Problema ilimitado.")
            return None

        # Escolha da variável que sai da base
        y_pos = y[y > 0]  # y_pos é o vetor dos coeficientes positivos
        y_min = np.argmin(y_pos)  # y_min é a posição da variável que sai da base no vetor y_pos (indice do valor mínimo)

        il = np.where(y == y_pos[y_min])[0][0]  # il é a posição da variável que sai da base no vetor
        xl = vb[il]  # xl é o índice da variável que sai da base (vale a pena ressaltar [0...-n])

        # Atualiza a solução básica
        vb[il] = k  # A variável que entra na base ocupa o lugar da que sai
        vnb[k] = xl  # A variável que sai da base ocupa o lugar da que entra

    # Retornar a solução ótima
    # A solução ótima é dada pelos valores das variáveis básicas e pelo valor da função objetivo
    x = np.zeros(n + m)  # x é o vetor da solução ótima
    x[vb] = sbf  # As variáveis básicas recebem os valores dos termos independentes
    for i, valor in enumerate(x, start=1):
        print(f"X{i}= %.2f" % valor)

    z = np.dot(c[vb], sbf)  # z é o valor ótimo da função objetivo
    print("\nValor otimo: %.2f" % z)

while True:
    print("------------------------------------------")
    print("              Menu Principal              ")
    print("------------------------------------------")
    print("1 - \033[34mFazer Simplex\033[m")
    print("0 - \033[34mSair\033[m")
    print("------------------------------------------")
    
    escolha = input("\033[32mEscolha uma opção: \033[m")

    if escolha == "1":
        problema = str(input('O problema é de MAX ou MIN? '))
        simplex(A, b, c, problema)
        input("Aperte enter para continuar...")
        os.system('cls')
    elif escolha == "0":
        print("Saindo do programa. Adeus!")
        break
    else:
        print("\033[31mOpção inválida. Por favor, escolha 1 para fazer o Simplex ou 0 para sair.\033[m")
