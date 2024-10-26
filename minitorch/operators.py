"""Collection of the core mathematical operators used throughout the code base."""

import math


def mul(x: float, y: float) -> float:
    return x * y


def id(x):
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -1.0 * x


def lt(x: float, y: float) -> float:
    # return float(x < y)
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    # return float(x == y)
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    return x if x >= y else y


def is_close(x: float, y: float) -> float:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x))


def relu(x: float) -> float:
    return (x > 0) * x


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    return 1 / x


def log_back(x: float, y) -> float:
    return y / x


def inv_back(x: float, y) -> float:
    # return neg(y) / x ** 2
    return -y / x ** 2


def relu_back(x: float, y) -> float:
    return (x > 0) * y


def map(f):
    return lambda x: [f(y) for y in x]


def zipWith(f):
    return lambda x, y: [f(a, b) for a, b in zip(x, y)]


def reduce(f, initializer):
    def fun(f, x, initializer):
        it = iter(x)
        val = initializer
        for i in it:
            val = f(val, i)
        return val

    return lambda x: fun(f, x, initializer)


def negList(x):
    return map(neg)(x)


def addLists(x, y):
    return zipWith(add)(x, y)


def sum(x):
    return reduce(add, 0)(x)


def prod(x):
    return reduce(mul, 1)(x)
