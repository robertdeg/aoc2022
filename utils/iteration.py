
def overwrite(xs: list, func, init):
    """
    Recomputes elements a list by accumulating a function, stopping at the point
    where the function raises a StopIteration exception

    :param xs: list
    :param func:
    :param init:
    :return: [r[0] = func(init, xs[0]), r[1] = func(r[0], xs[1]), ...]
    """
    for i, x in enumerate(xs):
        try:
            init = func(init, x)
            xs[i] = init
        except StopIteration:
            break
