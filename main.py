import math
from utils.coordinates import vector
from dataclasses import dataclass
from utils.iteration import overwrite
import collections
import operator
from collections import defaultdict
from itertools import zip_longest, starmap, count, chain, islice, takewhile, accumulate, tee, dropwhile, repeat, pairwise
import operator as ops
from functools import reduce, partial, cmp_to_key
from collections import Counter
import numpy as np
import re


def day1(filename: str):
    cals = sorted(map(lambda s: sum(map(int, s.split('\n'))), open(filename).read().split('\n\n')), reverse=True)
    return cals[0], sum(cals[:3])


def day2(filename: str):
    add_tups = lambda xs, ys: map(ops.add, xs, ys)
    points = dict(A=dict(X=4, Y=8, Z=3), B=dict(X=1, Y=5, Z=9), C=dict(X=7, Y=2, Z=6))
    plays = dict(A=dict(X='Z', Y='X', Z='Y'),
                 B=dict(X='X', Y='Y', Z='Z'),
                 C=dict(X='Y', Y='Z', Z='X'))
    chars = (l.split() for l in open(filename).read().split('\n'))
    scores = ((points[a][b], points[a][plays[a][b]]) for a, b in chars)
    part1, part2 = reduce(add_tups, scores)
    return part1, part2


def day3(filename: str):
    split = lambda s: (s[:len(s) // 2], s[len(s) // 2:])
    shared = lambda xs: next(iter(reduce(set.intersection, map(set, xs))))
    priority = lambda c: ord(c) - ord('a') + 1 if 'a' <= c <= 'z' else ord(c) - ord('A') + 27
    lines = [l.strip() for l in open(filename).read().split()]
    part1 = sum(priority(shared(split(xs))) for xs in lines)
    part2 = sum(priority(shared(lines[i:i + 3])) for i in range(0, len(lines), 3))
    return part1, part2


def day4(filename: str):
    lines = (l.strip() for l in open(filename).read().split())
    ranges = (re.match(r"(\d+)-(\d+),(\d+)-(\d+)", line).groups() for line in lines)
    ranges = [(int(a), int(b), int(c), int(d)) for a, b, c, d in ranges]
    part1 = sum(a <= c <= d <= b or c <= a <= b <= d for a, b, c, d in ranges)
    part2 = sum(not (b < c or d < a) for a, b, c, d in ranges)
    return part1, part2


def day5(filename: str):
    def move(source: list, dest: list, count: int, reverse=True) -> None:
        stack = source[-count:]
        dest.extend(reversed(stack) if reverse else stack)
        del source[-count:]

    stack_data, moves_list = map(str.splitlines, open(filename).read().split('\n\n'))
    blocks = [[line[i + 1:i + 2] for i in range(0, len(line), 4)] for line in stack_data[:-1]]
    stacks = [list(''.join(reversed(stack)).strip()) for stack in zip_longest(*blocks, fillvalue='')]
    moves = (re.match(r'move (\d+) from (\d+) to (\d+)', move).groups() for move in moves_list)
    moves = [(int(n), int(src), int(dst)) for n, src, dst in moves]

    collections.deque(starmap(lambda n, src, dst: move(stacks[src - 1], stacks[dst - 1], n), moves), maxlen=0)
    part1 = ''.join([stack[-1] for stack in stacks])

    stacks = [list(''.join(reversed(stack)).strip()) for stack in zip_longest(*blocks, fillvalue='')]
    collections.deque(starmap(lambda n, src, dst: move(stacks[src - 1], stacks[dst - 1], n, reverse=False), moves),
                      maxlen=0)
    part2 = ''.join([stack[-1] for stack in stacks])

    return part1, part2


def day6(filename: str):
    data = open(filename).read().strip()
    part1 = next(idx for idx, xs in enumerate(zip(*(data[i:] for i in range(4)))) if len(set(xs)) == 4)
    part2 = next(idx for idx, xs in enumerate(zip(*(data[i:] for i in range(14)))) if len(set(xs)) == 14)
    return part1, part2


def day7(filename: str):
    def parse(commands) -> dict:
        result = dict()
        next(commands)  # skip '$ ls'
        t1, t2 = tee(commands)
        ls = map(str.split, takewhile(lambda s: not s.startswith('$'), t1))
        size = sum(int(size) for size, _ in ls if size.isnumeric())
        commands = dropwhile(lambda s: not s.startswith('$'), t2)
        cd = re.match(r'\$ cd (.*)', next(commands, '$ cd ..')).group(1)
        while cd != '..':
            result[cd] = parse(commands)
            size += result[cd]['size']
            cd = re.match(r'\$ cd (.*)', next(commands, '$ cd ..')).group(1)

        result['size'] = size
        return result

    def totalsize(name: str, dir: dict):
        yield from chain.from_iterable(totalsize(v, desc) for v, desc in dir.items() if type(desc) is dict)
        yield name, dir['size']

    data = map(str.strip, open(filename).readlines())
    next(data)
    fs = parse(data)
    minsize = fs['size'] - 40000000
    xs = list(totalsize('/', fs))
    part1 = sum(sz for name, sz in totalsize('/', fs) if sz < 100000)
    part2 = min(sz for name, sz in totalsize('/', fs) if sz >= minsize)
    return part1, part2


def day8(filename: str):
    data = [s.strip() for s in open(filename).readlines()]
    N = len(data)
    trees = dict(((i, j), int(height)) for i in range(N) for j, height in enumerate(data[i]))
    left = [[(i, j) for j in range(N)] for i in range(N)]
    right = [[(i, j - 1) for j in range(N, 0, -1)] for i in range(N)]
    top = [[(i, j) for i in range(N)] for j in range(N)]
    bottom = [[(i - 1, j) for i in range(N, 0, -1)] for j in range(N)]

    def visible(state: (set, int), pos: (int, int)) -> (set, int):
        xs, minheight = state
        if trees[pos] > minheight:
            xs.add(pos)
        return (xs, max(minheight, trees[pos]))

    part1 = len(reduce(set.union, (reduce(visible, xs, (set(), -1))[0] for xs in left + right + top + bottom)))

    def count_upto(pred, it):
        result = 0
        for val in it:
            result += 1
            if not pred(val):
                break
        return result

    scores = dict()

    def viewrange(row: int, col: int) -> int:
        treeheight = trees[(row, col)]
        viewable = lambda h: h < treeheight
        ls = count_upto(viewable, (trees[(r, c)] for r, c in left[row] if c > col))
        rs = count_upto(viewable, (trees[(r, c)] for r, c in right[row] if c < col))
        ts = count_upto(viewable, (trees[(r, c)] for r, c in top[col] if r > row))
        bs = count_upto(viewable, (trees[(r, c)] for r, c in bottom[col] if r < row))
        return ls * rs * ts * bs

    part2 = max(viewrange(row, col) for row, col in trees)
    return part1, part2


def day9(filename: str):
    def move(head, tail):
        if any(abs(head - tail) > 1):
            return tail + (head - tail).clip(-1, 1)
        else:
            raise StopIteration

    deltas = dict(U=vector(1, 0), D=vector(-1, 0), L=vector(0, -1), R=vector(0, 1))
    data = (s.strip().split(' ') for s in open(filename).readlines())
    steps = ((deltas[x], int(y)) for x, y in data)

    heads = accumulate(chain.from_iterable(repeat(step, n) for step, n in steps), operator.add, initial=vector(0, 0))
    tail = [vector(0, 0)] * 9
    part1, part2 = set(), set()
    for head in heads:
        overwrite(tail, move, head)
        part1.add(tail[0])
        part2.add(tail[-1])

    return len(part1), len(part2)


def day10(filename: str):
    data = map(re.Match.groups, re.finditer(r'noop|addx (-?\d+)', open(filename).read()))
    deltas = chain.from_iterable(starmap(lambda x: [0] if x is None else [0, int(x)], data))
    xs = list(accumulate(deltas, operator.add, initial=1))
    part1 = sum(x * i if (i - 20) % 40 == 0 else 0 for x, i in zip(xs, count(1)))

    sprites = ((x - 1, x, x + 1) for x in xs)
    part2 = ''.join('@' if t % 40 in sprite else '.' for t, sprite in zip(count(0), sprites))

    return part1, '\n' + '\n'.join(part2[i:i + 40] for i in range(0, 240, 40))


def day11(filename: str):
    class monkey:
        def __init__(self, op, modulus, iftrue, iffalse):
            self.op = lambda old : eval(op)
            self.target = lambda level: int(iftrue) if level % int(modulus) == 0 else int(iffalse)


    pattern = re.compile(r'\: ([^\n]*)[^=]*= ([^\n]*)\n[^\d]*(\d+)[^\d]*(\d+)[^\d]*(\d+)')
    matches = pattern.finditer(open(filename).read())
    items, ops, divs, to_true, to_false = zip(*(match.groups() for match in matches))

    monkeys = list(starmap(monkey, zip(ops, divs, to_true, to_false)))
    items = list(chain.from_iterable([(idx, int(level)) for level in s.split(', ')] for idx, s in enumerate(items)))
    lcm = reduce(math.lcm, map(int, divs))

    def throw(round: int, monkey_idx: int, worry_level: int, div: int = 3):
        while True:
            yield round, monkey_idx, worry_level
            worry_level = monkeys[monkey_idx].op(worry_level) // div % lcm
            next_monkey_idx = monkeys[monkey_idx].target(worry_level)
            if next_monkey_idx < monkey_idx:
                round += 1
            monkey_idx = next_monkey_idx

    part1 = Counter(i for j, lvl in items for _, i, _ in takewhile(lambda t: t[0] < 20, throw(0, j, lvl)))
    part2 = Counter(i for j, lvl in items for _, i, _ in takewhile(lambda t: t[0] < 10000, throw(0, j, lvl, 1)))

    business = lambda counts : reduce(operator.mul, sorted(counts.values(), reverse=True)[:2])

    return business(part1), business(part2)


def day12(filename: str):
    data = {(row, col): char for row, line in enumerate(open(filename).readlines()) for col, char in enumerate(line)}
    endpos, start = map(operator.itemgetter(1), sorted((val, pos) for pos, val in data.items() if val in 'SE'))
    data.update({start: 'a', endpos: 'z'})
    data.update({pos: ord(ch) - ord('a') for pos, ch in data.items()})
    distances = {pos: 0 if pos == endpos else 10000 for pos in data}

    def edges(r, c):
        yield from (p for p in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))
                    if p in data and -1 <= data[p] - data[(r, c)])

    planned = set(data.keys())
    while planned:
        mindist, pos = min((distances[pos], pos) for pos in planned)
        planned.remove(pos)
        distances.update({nbor: min(distances[nbor], distances[pos] + 1) for nbor in edges(*pos)})

    return distances[start], min(distances[pos] for pos in data if data[pos] == 0)


def day13(filename: str):
    data = [[eval(line) for line in s.split()] for s in open(filename).read().split('\n\n')]

    def compare(xs, ys) -> int:
        if (type(xs) == int) + (type(ys) == int) == 1:
            return compare([xs], ys) if type(xs) == int else compare(xs, [ys])
        elif type(xs) == type(ys) == int:
            return xs - ys
        else:
            return reduce(lambda s, pair: s or compare(*pair), zip(xs, ys), 0) or len(xs) - len(ys)

    xs = sorted(chain.from_iterable(data + [[[2], [6]]]), key=cmp_to_key(compare))

    part1 = sum(idx + 1 for idx, pair in enumerate(data) if compare(*pair) < 0)
    part2 = reduce(operator.mul, (i + 1 for i, ls in enumerate(xs) if ls == [2] or ls == [6]))
    return part1, part2

def day14(filename: str):
    data = [[eval(line) for line in s.split(' -> ')] for s in open(filename).readlines()]
    cave = dict()
    for line in data:
        for (ax, ay), (bx, by) in pairwise(line):
            cave.update({(x, ay) : '#' for x in range(min(ax, bx), max(ax, bx) + 1)})
            cave.update({(ax, y) : '#' for y in range(min(ay, by), max(ay, by) + 1)})

    def fall(x, y, blocked: bool = False):
        if not blocked and not any(ay > y for ax, ay in cave if ax == x):
            return False
        elif (x, y + 1) not in cave:
            return fall(x, y + 1, blocked)
        elif (x - 1, y + 1) not in cave:
            return fall(x - 1, y + 1, blocked)
        elif (x + 1, y + 1) not in cave:
            return fall(x + 1, y + 1, blocked)
        else:
            cave[(x, y)] = 'o'
            return True
    part1 = 0
    while fall(500, 0):
        part1 += 1

    maxy = max(y for x, y in cave) + 2
    cave.update({(x, maxy) : '#' for x in range(500 - maxy - 1, 500 + maxy + 2)})

    part2 = part1
    while (500, 0) not in cave and fall(500, 0, True):
        part2 += 1

    return part1, part2

if __name__ == '__main__':
    solvers = [(key, value) for key, value in globals().items() if key.startswith("day") and callable(value)]
    solvers = sorted(((int(key.split('day')[-1]), value) for key, value in solvers), reverse=True)

    for idx, solver in solvers:
        p1, p2 = solver(f"input/day{idx}.txt")
        print(f"day {idx} - part 1: {p1}, part 2: {p2}")
