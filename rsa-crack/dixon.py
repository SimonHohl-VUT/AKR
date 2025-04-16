from math import gcd
import numpy as np

def fun(n):
    """
    The main function implementing Dixon's factorization method.
    :param n: The number to factorize.
    :return: A tuple of the first two factors of n.
    """
    base = [2, 3, 5, 7]
    start = int(n ** 0.5)
    pairs = []

    for i in range(start, n):
        for j in range(len(base)):
            lhs = i**2 % n
            rhs = base[j]**2 % n
            
            if lhs == rhs:
                pairs.append([i, base[j]])

    new = []
    for i in range(len(pairs)):
        factor = gcd(pairs[i][0] - pairs[i][1], n)
        if factor != 1 and factor != n:
            new.append(factor)

    x = np.unique(np.array(new))
    if len(x) > 1:
        return (x[0], x[1])
    else:
        return (1, n)