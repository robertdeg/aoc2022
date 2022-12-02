import operator as ops
from functools import reduce

def day1(filename: str):
    cals = sorted(map(lambda s: sum(map(int, s.split('\n'))), open(filename).read().split('\n\n')), reverse=True)
    print(f"day 1 - part 1: {cals[0]}, part 2: {sum(cals[:3])}")

def day2(filename: str):
    add_tups = lambda xs, ys: map(ops.add, xs, ys)
    points = dict(A = dict(X = 4, Y = 8, Z = 3), B = dict(X = 1, Y = 5, Z = 9), C = dict(X = 7, Y = 2, Z = 6))
    plays = dict(A = dict(X = 'Z', Y = 'X', Z = 'Y'),
                 B = dict(X = 'X', Y = 'Y', Z = 'Z'),
                 C = dict(X = 'Y', Y = 'Z', Z = 'X'))
    chars = (l.split() for l in open(filename).read().split('\n'))
    scores = ((points[a][b], points[a][plays[a][b]]) for a, b in chars)
    part1, part2 = reduce(add_tups, scores)
    print(f"day 2 - part 1: {part1}, part 2: {part2}")

if __name__ == '__main__':
    day1("input/day1.txt")
    day2("input/day2.txt")


