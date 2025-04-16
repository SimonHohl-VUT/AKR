import sys
import random
import numpy as np

def gauss(M):
    """
    Perform Gaussian elimination on a binary matrix M.
    :param M: The binary matrix to perform elimination on.
    :return: A tuple (marks, M) where marks indicate which rows are marked.
    """
    
    marks = [False] * len(M)
    for j in range(len(M[0])):
        for i in range(len(M)):
            if M[i][j] == 1:
                marks[i] = True
                for k in range(j):
                    if M[i][k] == 1:
                        for row in range(len(M)):
                            M[row][k] = (M[row][k] + M[row][j]) % 2
                for k in range(j+1, len(M[0])):
                    if M[i][k] == 1:
                        for row in range(len(M)):
                            M[row][k] = (M[row][k] + M[row][j]) % 2
                break
    return (marks, M)

def get_dep_cols(row):
    """
    Get the indices of dependent columns in a row.
    :param row: The row to analyze.
    :return: A list of column indices where the row has a value of 1.
    """
    
    ret = []
    for i in range(len(row)):
        if row[i] == 1:
            ret.append(i)
    return ret

def row_add(new_row, current, M):
    """
    Add two rows in GF(2).
    :param new_row: The index of the row to add.
    :param current: The current row to add to.
    :param M: The matrix containing the rows.
    :return: The result of adding the rows.
    """
    
    ret = current
    for i in range(len(M[new_row])):
        ret[i] ^= M[new_row][i]
    return ret

def is_dependent(cols, row):
    """
    Check if a row is dependent on a set of columns.
    :param cols: The set of columns to check.
    :param row: The row to analyze.
    :return: True if the row is dependent, False otherwise.
    """
    
    for i in cols:
        if row[i] == 1:
            return True
    return False

def find_linear_deps(row, M):
    """
    Find linear dependencies involving a specific row in the matrix.
    :param row: The row to analyze.
    :param M: The matrix containing the rows.
    :return: A list of linear dependencies.
    """
    
    ret = []
    dep_cols = get_dep_cols(M[row])
    current_rows = [row]
    current_sum = M[row][:]
    for i in range(len(M)):
        if i == row:
            continue
        if is_dependent(dep_cols, M[i]):
            current_rows.append(i)
            current_sum = row_add(i, current_sum, M)
            if sum(current_sum) == 0:
                ret.append(current_rows[:])
    return ret

def testdep(dep, smooth_vals, N):
    """
    Test a dependency to find factors of N.
    :param dep: The dependency to test.
    :param smooth_vals: The smooth values used in the sieve.
    :param N: The number to factorize.
    :return: A factor of N if found, otherwise 1.
    """
    
    x = y = 1
    for row in dep:
        x *= smooth_vals[row][0]
        y *= smooth_vals[row][1]
    return xgcd(x - isqrt(y), N)[0]

def phi(p):
    """
    Euler's totient function for a prime p.
    :param p: The prime number.
    :return: p-1, since phi(p) = p-1 for prime p.
    """
    
    return p-1

def legendre(a, p):
    """
    Compute the Legendre symbol (a/p).
    :param a: The numerator.
    :param p: The prime denominator.
    :return: The Legendre symbol value.
    """
    
    if a % p == 0:
        return 0
    return pow(a, (p - 1) // 2, p)

def miller(n, trials=5):
    """
    Miller-Rabin primality test.
    :param n: The number to test.
    :param trials: The number of iterations for the test.
    :return: True if n is likely prime, False otherwise.
    """
    
    s = 0
    d = n - 1
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    while True:
        quotient, remainder = divmod(d, 2)
        if remainder == 1:
            break
        s += 1
        d = quotient
    for _ in range(trials):
        a = random.randrange(2, n)
        composite = True
        if pow(a, d, n) == 1:
            continue
        for i in range(s):
            if pow(a, 2**i * d, n) == n-1:
                composite = False
                break
        if composite:
            return False
    return True

def isqrt(n):
    """
    Compute the integer square root of n using Newton's method.
    :param n: The number to compute the square root of.
    :return: The integer square root of n.
    """
    
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

def xgcd(a, b):
    """
    Extended Euclidean algorithm.
    :param a: The first number.
    :param b: The second number.
    :return: A tuple (gcd, x, y) where gcd is the greatest common divisor of a and b,
             and x, y are the coefficients for Bézout's identity.
    """
    
    prevx, x = 1, 0
    prevy, y = 0, 1
    while b:
        q, r = divmod(a, b)
        x, prevx = prevx - q*x, x
        y, prevy = prevy - q*y, y
        a, b = b, r
    return a, prevx, prevy

def tonelli(n, p):
    """
    Tonelli-Shanks algorithm for finding square roots modulo a prime.
    :param n: The number to find the square root of.
    :param p: The prime modulus.
    :return: A square root of n modulo p.
    """
    
    q = p - 1
    s = 0
    z = 2
    i = 1
    while q % 2 == 0:
        q //= 2
        s += 1
    if s == 1:
        return pow(n, (p + 1) // 4, p)
    while z < p and p - 1 != legendre(z, p):
        z += 1
    c = pow(z, q, p)
    r = pow(n, (q + 1) // 2, p)
    t = pow(n, q, p)
    m = s
    t2 = 0
    while (t - 1) % p != 0:
        t2 = (t * t) % p
        i = 1
        while i < m:
            if (t2 - 1) % p == 0:
                break
            t2 = (t2 * t2) % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        r = (r * b) % p
        c = (b * b) % p
        t = (t * c) % p
        m = i
    return r

def create_base(n, B):
    """
    Create a factor base for the quadratic sieve.
    :param n: The number to factorize.
    :param B: The bound for the factor base.
    :return: A list of primes in the factor base.
    """
    
    base = []
    i = 2
    while len(base) < B:
        if legendre(n, i) == 1:
            base.append(i)
        i += 1
        while not miller(i):
            i += 1
    return base

def poly(x, a, b, n):
    """
    Define the sieving polynomial (Ax + B)^2 - N.
    :param x: The variable.
    :param a: The coefficient A.
    :param b: The coefficient B.
    :param n: The number to factorize.
    :return: The value of the polynomial at x.
    """
    
    return ((a * x + b) ** 2) - n

def solve(a, b, n, base):
    """
    Solve the polynomial equation for sieving.
    :param a: The coefficient A.
    :param b: The coefficient B.
    :param n: The number to factorize.
    :param base: The factor base.
    :return: A list of starting values for sieving.
    """
    
    start_vals = []
    for p in base:
        ainv = 1
        if a != 1:
            g, ainv, _ = xgcd(a, p)
            assert g == 1
        r1 = tonelli(n, p)
        r2 = (-1 * r1) % p
        start1 = (ainv * (r1 - b)) % p
        start2 = (ainv * (r2 - b)) % p
        start_vals.append([start1, start2])
    return start_vals

def trial(n, base):
    """
    Perform trial division to produce an exponent vector for n with respect to the factor base.
    :param n: The number to factorize.
    :param base: The factor base.
    :return: An exponent vector in GF(2).
    """
    
    ret = [0] * len(base)
    if n > 0:
        for i in range(len(base)):
            while n % base[i] == 0:
                n //= base[i]
                ret[i] = (ret[i] + 1) % 2
    return ret

def fun(N):
    """
    The main function implementing the Quadratic Sieve algorithm.
    :param N: The number to factorize.
    :return: A tuple of the first two factors of N.
    """
    
    a = 1
    b = isqrt(N) + 1
    bound = 50
    base = create_base(N, bound)
    needed = phi(base[-1]) + 1

    sieve_start = 0
    sieve_stop = 0
    sieve_interval = 100000

    M = []
    smooth_vals = []
    start_vals = solve(a, b, N, base)
    seen = set([])

    while len(smooth_vals) < needed:
        sieve_start = sieve_stop
        sieve_stop += sieve_interval
        interval = [poly(x, a, b, N) for x in range(sieve_start, sieve_stop)]

        for p in range(len(base)):
            t = start_vals[p][0]

            while start_vals[p][0] < sieve_start + sieve_interval:
                while interval[start_vals[p][0] - sieve_start] % base[p] == 0:
                    interval[start_vals[p][0] - sieve_start] /= base[p]
                start_vals[p][0] += base[p]

            if start_vals[p][1] != t:
                while start_vals[p][1] < sieve_start + sieve_interval:
                    while interval[start_vals[p][1] - sieve_start] % base[p] == 0:
                        interval[start_vals[p][1] - sieve_start] /= base[p]
                    start_vals[p][1] += base[p]

        for i in range(sieve_interval):
            if interval[i] == 1:
                x = sieve_start + i
                y = poly(x, a, b, N)
                exp = trial(y, base)
                if not tuple(exp) in seen:
                    smooth_vals.append(((a * x) + b, y))
                    M.append(exp)
                    seen |= set([tuple(exp)])

    marks, M = gauss(M)

    f = []
    for i in range(len(marks)):
        if not marks[i]:
            deps = find_linear_deps(i, M)
            for dep in deps:
                gcd = testdep(dep, smooth_vals, N)
                if gcd != 1 and gcd != N:
                    f.append(gcd)

    f = np.unique(np.array(f))
    if len(f) > 1:
        return (f[0], f[1])
    else:
        return (1, N)