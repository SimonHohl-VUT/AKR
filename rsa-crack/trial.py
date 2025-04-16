def fun(n):
    """
    The main function implementing Trial Division for factorization.
    :param n: The number to factorize.
    :return: A tuple of the first two factors of n.
    """
    nsqrt = int(n ** 0.5)

    for f in range(2, nsqrt + 1):
        if n % f == 0:
            return (f, int(n / f))

    return (1, n)