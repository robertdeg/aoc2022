import collections
from itertools import zip_longest, starmap
import operator as ops
from functools import reduce, partial
from collections import Counter
import numpy as np
import re

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

def day3(filename: str):
    split = lambda s : (s[:len(s) // 2] ,s[len(s) // 2:])
    shared = lambda xs : next(iter(reduce(set.intersection, map(set, xs))))
    priority = lambda c : ord(c) - ord('a') + 1 if 'a' <= c <= 'z' else ord(c) - ord('A') + 27
    lines = [l.strip() for l in open(filename).read().split()]
    part1 = sum(priority(shared(split(xs))) for xs in lines)
    part2 = sum(priority(shared(lines[i:i + 3])) for i in range(0, len(lines), 3))
    print(f"day 3 - part 1: {part1}, part 2: {part2}")

def day4(filename: str):
    lines = (l.strip() for l in open(filename).read().split())
    ranges = (re.match(r"(\d+)-(\d+),(\d+)-(\d+)", line).groups() for line in lines)
    ranges = [(int(a), int(b), int(c), int(d)) for a, b, c, d in ranges]
    part1 = sum(a <= c <= d <= b or c <= a <= b <= d for a, b, c, d in ranges)
    part2 = sum(not(b < c or d < a) for a, b, c, d in ranges)
    print(f"day 4 - part 1: {part1}, part 2: {part2}")

def dropwhile(predicate, iterable):
    # dropwhile(lambda x: x<5, [1,4,6,4,1]) --> 6 4 1
    iterable = iter(iterable)
    for x in iterable:
        if not predicate(x):
            yield x
            break
    for x in iterable:
        yield x
def day5(filename: str):
    def move(source: list, dest: list, count: int, reverse=True) -> None:
        stack = source[-count:]
        dest.extend(reversed(stack) if reverse else stack)
        del source[-count:]

    stack_data, moves_list = map(str.splitlines, open(filename).read().split('\n\n'))
    blocks = [[line[i+1:i+2] for i in range(0, len(line), 4)] for line in stack_data[:-1]]
    stacks = [list(''.join(reversed(stack)).strip()) for stack in zip_longest(*blocks, fillvalue='')]
    moves = (re.match(r'move (\d+) from (\d+) to (\d+)', move).groups() for move in moves_list)
    moves = [(int(n), int(src), int(dst)) for n, src, dst in moves]

    collections.deque(starmap(lambda n, src, dst : move(stacks[src-1], stacks[dst-1], n), moves), maxlen=0)
    part1 = ''.join([stack[-1] for stack in stacks])

    stacks = [list(''.join(reversed(stack)).strip()) for stack in zip_longest(*blocks, fillvalue='')]
    collections.deque(starmap(lambda n, src, dst : move(stacks[src-1], stacks[dst-1], n, reverse=False), moves), maxlen=0)
    part2 = ''.join([stack[-1] for stack in stacks])

    print(f"day 5 - part 1: {part1}, part 2: {part2}")

if __name__ == '__main__':
    day1("input/day1.txt")
    day2("input/day2.txt")
    day3("input/day3.txt")
    day4("input/day4.txt")
    day5("input/day5.txt")


