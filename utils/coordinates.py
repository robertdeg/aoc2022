class vector(tuple):
    def __new__(self, *xs):
        return super().__new__(vector, xs)

    def __add__(self, other):
        return vector(*(x + y for x, y in zip(self, other)))

    def __sub__(self, other):
        return vector(*(x - y for x, y in zip(self, other)))

    def __abs__(self):
        return vector(*map(abs, self))

    def __gt__(self, other):
        return vector(*map(lambda x: x > other, self))

    def clip(self, a, b):
        return vector(*map(lambda x: max(a, min(b, x)), self))
