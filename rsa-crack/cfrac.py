from collections import defaultdict
from math import isqrt, gcd, sqrt, log, exp, prod
import random

def is_square(n):
    """
    Check if n is a perfect square.
    :param n: The number to check.
    :return: True if n is a perfect square, False otherwise.
    """
    return isqrt(n) ** 2 == n

def factor(n):
    """
    Factorize n into its prime components using trial division.
    :param n: The number to factorize.
    :return: A dictionary with prime factors as keys and their exponents as values.
    """
    Ans = defaultdict(lambda: 0)
    d = 2
    while d * d <= n:
        if n % d == 0:
            Ans[d] += 1
            n //= d
        else:
            d += 1
    if n > 1:
        Ans[n] += 1
    return Ans

def is_square_free(n):
    """
    Check if n is square-free (i.e., no prime factor appears more than once).
    :param n: The number to check.
    :return: True if n is square-free, False otherwise.
    """
    for e in factor(n).values():
        if e >= 2:
            return False
    return True

def next_multiplier(n, k):
    """
    Find the next multiplier k for the continued fraction factorization method.
    :param n: The number to factorize.
    :param k: The current multiplier.
    :return: The next suitable multiplier.
    """
    k += 2
    while (not is_square_free(k) or gcd(k, n) != 1):
        k += 1
    return k

def gaussian_elimination(A, n):
    """
    Perform Gaussian elimination on a binary matrix A.
    :param A: The binary matrix to perform elimination on.
    :param n: The number of columns in the matrix.
    :return: The identity matrix after elimination.
    """
    m = len(A)
    I = [1 << k for k in range(m + 1)]
    nrow = 0

    for col in range(1, min(m, n) + 1):
        npivot = 0

        for row in range(nrow + 1, m + 1):
            if ((A[row - 1] >> (col-1)) & 1) == 1:
                npivot = row
                nrow += 1
                break

        if npivot == 0:
            continue

        if npivot != nrow:
            A[npivot - 1], A[nrow - 1] = A[nrow - 1], A[npivot - 1]
            I[npivot - 1], A[nrow - 1] = I[nrow - 1], I[npivot - 1]

        for row in range(nrow+1, m + 1):
            if ((A[row - 1] >> (col-1)) & 1) == 1:
                A[row - 1] = A[row - 1] ^ A[nrow - 1]
                I[row - 1] = I[row - 1] ^ I[nrow - 1]

    return I

def is_prime(n, k=10):
    """
    Check if n is a prime using the Miller-Rabin primality test.
    :param n: The number to check.
    :param k: The number of iterations for the test.
    :return: True if n is likely prime, False otherwise.
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False

    s = 0
    d = n-1
    while d % 2 == 0:
        s += 1
        d //= 2

    for i in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)

        if x != 1:
            for r in range(s):
                if x == n - 1:
                    break
                x = (x ** 2) % n
            if x != n - 1:
                return False

    return True

def check_factor(n, g, factors):
    """
    Check if g is a factor of n and update the factor list.
    :param n: The number to factorize.
    :param g: The potential factor.
    :param factors: The list of factors.
    :return: The remaining part of n after dividing out g.
    """
    while n % g == 0:
        n //= g
        factors.append(g)
        if is_prime(n):
            factors.append(n)
            return 1
    return n

def is_smooth_over_prod(n, k):
    """
    Check if n is smooth over the product of the factor base.
    :param n: The number to check.
    :param k: The product of the factor base.
    :return: True if n is smooth, False otherwise.
    """
    g = gcd(n, k)
    while g > 1:
        n //= g
        while n % g == 0:
            n //= g
        if n == 1:
            return True
        g = gcd(n, g)
    return n == 1

def jacobi(a, n):
    """
    Compute the Jacobi symbol (a/n).
    :param a: The numerator.
    :param n: The denominator.
    :return: The Jacobi symbol value.
    """
    a %= n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a >>= 1
            if ((n % 8) in [3, 5]):
                result *= -1
        a, n = n, a
        if (a % 4 == n % 4 == 3):
            result *= -1
        a %= n
    return result if n == 1 else 0

def cffm(n, multiplier=1):
    """
    The main function implementing the Continued Fraction Factorization Method (CFFM).
    :param n: The number to factorize.
    :param multiplier: The multiplier to use for the factorization.
    :return: A list of prime factors of n.
    """
    if n <= 1:
        return []
    if is_prime(n):
        return [n]

    if n % 2 == 0:
        v = 0
        while n % 2 == 0:
            v += 1
            n >>= 1
        arr1 = [2] * v
        arr2 = cffm(n)
        return sorted(arr1 + arr2)

    if is_square(n):
        f = cffm(isqrt(n))
        return sorted(f + f)

    N = n * multiplier
    x = isqrt(N)
    y = x
    z = 1
    w = x + x
    r = w

    B = round(exp(sqrt(log(n) * log(log(n))) / 2))
    factor_base = []

    for p in range(B):
        if is_prime(p) and jacobi(N, p) >= 0:
            factor_base.append(p)

    factor_prod = prod(factor_base)
    factor_index = {}

    for k in range(1, len(factor_base) + 1):
        factor_index[factor_base[k - 1]] = k - 1

    def exponent_signature(factors):
        """
        Compute the exponent signature of a number based on its prime factors.
        :param factors: A dictionary of prime factors and their exponents.
        :return: A binary signature representing the parity of exponents.
        """
        sig = 0
        for p, e in factors.items():
            if e % 2 == 1:
                sig |= (1 << factor_index[p])
        return sig

    L = len(factor_base) + 1
    Q = []
    A = []

    (f1, f2) = (1, x)

    while len(A) < L:
        y = (r * z - y)
        z = (N - y * y) // z
        r = (x + y) // z

        (f1, f2) = (f2, (r * f2 + f1) % n)

        if is_square(z):
            g = gcd(f1 - isqrt(z), n)
            if g > 1 and g < n:
                arr1 = cffm(g)
                arr2 = cffm(n // g)
                return sorted(arr1 + arr2)

        if z > 1 and is_smooth_over_prod(z, factor_prod):
            A.append(exponent_signature(factor(z)))
            Q.append([f1, z])

        if z == 1:
            return cffm(n, next_multiplier(n, multiplier))

    while len(A) < L:
        A.append(0)

    I = gaussian_elimination(A, len(A))

    LR = 0
    for k in range(len(A) - 1, 0, -1):
        if A[k] != 0:
            LR = k + 1
            break

    remainder = n
    factors = []

    def cfrac_find_factors(solution, remainder):
        """
        Find factors of n using the solution from Gaussian elimination.
        :param solution: The solution vector from Gaussian elimination.
        :param remainder: The remaining part of n to factorize.
        :return: A tuple (flag, remainder) where flag is True if a factor was found.
        """
        X = 1
        Y = 1

        for i in range(len(Q)):
            if ((solution >> i) & 1) == 1:
                X *= Q[i][0]
                Y *= Q[i][1]

                g = gcd(X - isqrt(Y), remainder)

                if (g > 1 and g < remainder):
                    remainder = check_factor(remainder, g, factors)
                    if remainder == 1:
                        return True, remainder

        return False, remainder

    for k in range(LR - 1, len(I)):
        flag, remainder = cfrac_find_factors(I[k], remainder)
        if flag:
            break

    final_factors = []

    for f in factors:
        if is_prime(f):
            final_factors.append(f)
        else:
            final_factors += cffm(f)

    if remainder != 1:
        if remainder != n:
            final_factors += cffm(remainder)
        else:
            final_factors.append(remainder)

    if remainder == n:
        return cffm(n, next_multiplier(n, multiplier))

    return sorted(final_factors)

def fun(n):
    """
    A wrapper function for the CFFM method.
    :param n: The number to factorize.
    :return: A tuple of the first two factors of n.
    """
    f = cffm(n)
    if len(f) > 1:
        return (f[0], f[1])
    else:
        return (1, n)