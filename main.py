import bisect
import math
import time

import numpy as np
import sympy
from sympy import symbols
from utils.coordinates import vector
from utils.iteration import overwrite
import collections
import operator
from itertools import zip_longest, starmap, count, chain, islice, takewhile, accumulate, tee, dropwhile, repeat, \
    pairwise, combinations, permutations, cycle, product
import operator as ops
from functools import reduce, partial, cmp_to_key
from collections import Counter
import queue as Q
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
            self.op = lambda old: eval(op)
            self.target = lambda level: int(iftrue) if level % int(modulus) == 0 else int(iffalse)

    pattern = re.compile(r'\: ([^\n]*)[^=]*= ([^\n]*)\n[^\d]*(\d+)[^\d]*(\d+)[^\d]*(\d+)')
    matches = pattern.finditer(open(filename).read())
    items, ops, divs, to_true, to_false = zip(*(match.groups() for match in matches))

    monkeys = list(starmap(monkey, zip(ops, divs, to_true, to_false)))
    items = list(chain.from_iterable([(idx, int(level)) for level in s.split(', ')] for idx, s in enumerate(items)))
    lcm = reduce(math.lcm, map(int, divs))

    def throw(monkey_idx: int, worry_level: int, rounds: int = 20, div: int = 3):
        round = 0
        visited = dict()
        counts = Counter()
        while round < rounds:
            counts = counts + Counter([monkey_idx])
            if (monkey_idx, worry_level) in visited:
                prev_round, prev_counts = visited[(monkey_idx, worry_level)]
                skip = (rounds - 1 - round) // (round - prev_round)
                counts = Counter({key: counts[key] + skip * (counts[key] - prev_counts[key]) for key in counts})
                round += skip * (round - prev_round)
                visited.clear()
            visited[(monkey_idx, worry_level)] = round, counts

            worry_level = monkeys[monkey_idx].op(worry_level) // div % lcm
            next_monkey_idx = monkeys[monkey_idx].target(worry_level)
            if next_monkey_idx < monkey_idx:
                round += 1
            monkey_idx = next_monkey_idx
        return counts

    part1 = reduce(operator.add, (throw(j, lvl, 20, 3) for j, lvl in items))
    part2 = reduce(operator.add, (throw(j, lvl, 10000, 1) for j, lvl in items))
    business = lambda counts: reduce(operator.mul, sorted(counts.values(), reverse=True)[:2])

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


def day14(filename: str):  # TODO: use DFS to solve part 2
    data = [[eval(line) for line in s.split(' -> ')] for s in open(filename).readlines()]
    cave = dict()
    for line in data:
        for (ax, ay), (bx, by) in pairwise(line):
            cave.update({(x, ay): '#' for x in range(min(ax, bx), max(ax, bx) + 1)})
            cave.update({(ax, y): '#' for y in range(min(ay, by), max(ay, by) + 1)})

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
    cave.update({(x, maxy): '#' for x in range(500 - maxy - 1, 500 + maxy + 2)})

    part2 = part1
    while (500, 0) not in cave and fall(500, 0, True):
        part2 += 1

    return part1, part2


def day15(filename: str):
    class rectangle:  # TODO: move to some other file
        def __init__(self, sx, sy, xb, yb):
            self.topleft = sx, sy
            self.topright = xb, sy
            self.bottomleft = sx, yb
            self.bottomright = xb, yb
            self.mid = (xb + sx) // 2, (yb + sy) // 2

        def external(self, outer):
            up = rectangle(*outer.topleft, *map(min, outer.bottomright, self.topright))
            down = rectangle(*map(max, outer.topleft, self.bottomleft), *outer.bottomright)
            left = rectangle(*map(max, outer.topleft, up.bottomleft), *map(min, outer.bottomright, down.bottomleft))
            right = rectangle(*map(max, outer.topleft, up.topright), *map(min, outer.bottomright, down.topright))
            return up, left, down, right

        def __bool__(self):
            return self.topleft[0] < self.bottomright[0] and self.topleft[1] < self.bottomright[1]

        def __contains__(self, item):
            x, y = item
            return self.topleft[0] <= x < self.bottomright[0] and self.topleft[1] <= y < self.bottomright[1]

    def find(space: rectangle, diamonds):
        if space:
            if not diamonds:
                yield space
            else:
                up, left, down, right = diamonds[0].external(space)
                yield from chain.from_iterable(find(r, diamonds[1:]) for r in diamonds[0].external(space))

    warp = lambda x, y: (x - y, x + y)
    unwarp = lambda x, y: ((x + y) // 2, (y - x) // 2)
    data = [eval(f"(({m.group(1)},{m.group(2)}),({m.group(3)},{m.group(4)}))")
            for m in re.finditer(r'x=(\d+), y=(\d+)\:[^x]*x=(-?\d+), y=(-?\d+)', open(filename).read())]
    sensors = [warp(*sensor) for sensor, beacon in data]
    beacons = {b for _, b in data}
    halfsizes = [abs(sx - bx) + abs(sy - by) for (sx, sy), (bx, by) in data]

    rs = [rectangle(sx - r, sy - r, sx + r + 1, sy + r + 1) for (sx, sy), r in zip(sensors, halfsizes)]
    part1 = 0  # len(reduce(set.union, (d.void(2000000) for d in ds))) - sum(1 for x, y in beacons if y == 2000000)
    emptyrects = (unwarp(*r.mid) for r in find(rectangle(-4000000, -4000000, 8000000, 8000000), rs))
    x, y = next(filter(lambda p: p in rectangle(0, 0, 4000000, 4000000), emptyrects))
    part2 = x * 4000000 + y

    return part1, part2


def day16(filename: str):
    data = re.finditer(r'Valve ([^\s]+) has flow rate=(\d+); [\sa-z]+ ([^\n]+)', open(filename).read())
    valves = {v: (int(rate), ws.split(', ')) for v, rate, ws in map(re.Match.groups, data)}
    rates = {v: rate for v, (rate, _) in valves.items() if rate > 0}
    distances = {(v, v): 0 for v in valves}

    changes = 1
    while changes:
        changes = 0
        for v, w in permutations(valves, 2):
            distance = min(1 + distances.get((s, w), 1000) for s in valves[v][1])
            changes += distance < distances.get((v, w), 1000)
            distances[(v, w)] = distance

    distances = {(v, w): distance + 1 for (v, w), distance in distances.items() if v != w}

    def out_edges(node: ((str, str), (int, int)), unopened: set):
        (v, w), (t1, t2) = node
        yield from ((rates[u] * (t1 - distances[v, u]), (u, w), (t1 - distances[v, u], t2)) for u in unopened - {v} if
                    distances[v, u] < t1 if u < w)
        yield from ((rates[u] * (t2 - distances[w, u]), (v, u), (t1, t2 - distances[w, u])) for u in unopened - {w} if
                    distances[w, u] < t2 if v < u)

    def max_pressure(top_order: list[((str, str), (int, int))]) -> int:
        pressures = dict()
        remaining_valves = dict()
        for (v1, v2), (t1, t2) in top_order:
            pressure_v = pressures.get(((v1, v2), (t1, t2)), 0)
            remaining_valves_v = remaining_valves.get(((v1, v2), (t1, t2)), rates.keys() - {v1, v2})
            for pressure, (w1, w2), (tt1, tt2) in out_edges(((v1, v2), (t1, t2)), remaining_valves_v):
                if pressure_v + pressure > pressures.get(((v1, v2), (tt1, tt2)), 0):
                    pressures[((v1, v2), (tt1, tt2))] = pressure_v + pressure
                    remaining_valves[((v1, v2), (tt1, tt2))] = remaining_valves_v - {w1, w2}
        return max(pressures.values())

    def top_order(start: str, time_left: int, succ_fun, visited: set[(str, int)] = set()):
        visited.add((start, time_left))
        result = []
        for _, next_node, t in succ_fun((start, time_left), rates.keys()):
            if (next_node, t) not in visited:
                result = top_order(next_node, t, succ_fun, visited) + result
        return [(start, time_left)] + result

    part1 = 0
    part1 = max_pressure(top_order(('AA', 'AA'), (0, 30), out_edges))
    part2 = max_pressure(top_order(('AA', 'AA'), (26, 26), out_edges))
    return part1, part2


def day17(filename: str):
    data = open(filename).read().strip()
    rocks = [
        [{0, 1, 2, 3}],
        [{1}, {0, 1, 2}, {1}],
        [{0, 1, 2}, {2}, {2}],
        [{0}, {0}, {0}, {0}],
        [{0, 1}, {0, 1}]
    ]

    def push_right(rock: list[set[int]]) -> list[set[int]]:
        result = [{i + 1 for i in layer} for layer in rock]
        return rock if max(max(layer) for layer in result) > 6 else result

    def push_left(rock: list[set[int]]) -> list[set[int]]:
        result = [{i - 1 for i in layer} for layer in rock]
        return rock if min(min(layer) for layer in result) < 0 else result

    def collide(xss: list[set[int]], yss: list[set[int]]) -> bool:
        return any(map(set.intersection, xss, yss))

    pile = [{0, 1, 2, 3, 4, 5, 6}]
    jets, moves, dropped = cycle(data), 0, 0
    visited = dict()
    rock, height = push_right(push_right(rocks[0])), 4
    for c in jets:
        rock = push_left(rock) if c == '<' else push_right(rock)
        moves += 1
        if collide(rock, pile[height:height + len(rock)]):
            rock = push_left(rock) if c == '>' else push_right(rock)
        if collide(rock, pile[height - 1:height + len(rock) - 1]):
            pile[height:height + len(rock)] = list(map(set.union, rock, pile[height:height + len(rock)] + 4 * [set()]))
            dropped += 1
            state = (moves % len(data), dropped % len(rocks))
            if dropped == 2022:
                part1 = len(pile)
            if state in visited:
                j, h = visited[state]
                if (1000000000000 - dropped) % (dropped - j) == 0:
                    part2 = len(pile) + ((1000000000000 - dropped) // (dropped - j)) * (len(pile) - h)
                    if dropped > 2022:
                        break
            visited[state] = dropped, len(pile)
            rock, height = push_right(push_right(rocks[dropped % len(rocks)])), len(pile) + 3
        else:
            height = height - 1
    return part1 - 1, part2 - 1


def day18(filename: str):
    data = {eval(line) for line in open(filename).readlines()}
    lo, hi = min(min(pt) for pt in data), max(max(pt) for pt in data)

    def nbors(pt):
        deltas = (xs for xs in product([-1, 0, 1], repeat=3) if sum(map(abs, xs)) == 1)
        yield from (tuple(map(operator.add, pt, d)) for d in deltas)

    def fill3d(pos: (int, int, int), visited: set, nborpred) -> int:
        count = 0
        queue = [(pos, nbors(pos))]
        while queue:
            p, it = queue[-1]
            if p not in visited:
                count += sum(s in data for s in nbors(p))
            visited.add(p)
            for s in filter(lambda p: (p not in visited) and nborpred(p), it):
                queue.append((s, nbors(s)))
                break
            else:
                queue.pop()
        return count

    def water_nbors(pt):
        return pt not in data and lo - 1 <= min(pt) and max(pt) <= hi + 1

    visited = set()
    adjacent = 0
    for point in data:
        if point not in visited:
            adjacent += fill3d(point, visited, lambda pt: pt in data)

    part1 = len(visited) * 6 - adjacent
    part2 = fill3d((lo - 1, lo - 1, lo - 1), set(), water_nbors)
    return part1, part2


def day19(filename: str):
    Counts = collections.namedtuple("Counts", "ore, clay, obsidian, geode")
    add_counts = lambda *xss: Counts(*(sum(xs) for xs in zip(*xss)))
    mul_counts = lambda a, xs: Counts(*map(lambda x : a * x, xs))

    xs = add_counts(Counts(1, 0, 0, 0), Counts(2, 1, 0, 0), Counts(3, 2, 1, 0))
    arr = np.array([[0, 0, 0, 0]])
    matches = (re.match(r'\D+' + 7 * r'(\d+)\D+', line).groups() for line in open(filename).readlines())

    blueprints = [Counts(
        Counts(int(ore), 0, 0, 0),
        Counts(int(clay), 0, 0, 0),
        Counts(int(o1), int(o2), 0, 0),
        Counts(int(g1), 0, int(g2), 0))
        for _, ore, clay, o1, o2, g1, g2 in matches]


    def estimate_geodes(blueprint: Counts[Counts], time: int, active: Counts, resources: Counts) -> int:
        pending = [0] * 4
        available = [list(resources)] * 4
        active = list(active)

        result = 0
        for t in range(0, time):
            for i, p in enumerate(pending):
                active[i] += pending[i]
            for i, bp, counts in zip(count(0), blueprint, available):
                pending[i] = min(1, min(c // v for v, c in zip(bp, counts) if v > 0))
            for bp, counts, p in zip(blueprint, available, pending):

            available = Counts(*(Counts(*(n - p * costs + a for n, costs, a in zip(counts, bp, active))) for bp, counts, p in zip(blueprint, available, pending)))
            result += active[3]
        return -(resources.geode + result)

    def next_states(blueprint: dict, max_bots: Counter[str], time: int, active: Counter[str], resources: Counter[str]):
        production = (Counts(1, 0, 0, 0), Counts(0, 1, 0, 0), Counts(0, 0, 1, 0), Counts(0, 0, 0, 1))
        constructible = (all(avail > 0 for req, avail in zip(bp, active) if req > 0) for bp in blueprint)
        sensible = (c and avail < upp for avail, upp, c in zip(active, max_bots, constructible))
        delays = (max(0, 1 + max((req - avail - 1) // inc if inc > 0 else 0 for req, avail, inc in zip(bp, resources, active))) for bp in blueprint)
        feasible = list((time - t - 1, add_counts(active, p), add_counts(resources, mul_counts(-1, bp), mul_counts(t + 1, active))) for t, s, p, bp in zip(delays, sensible, production, blueprint) if s and t < time)
        infeasible = list((0, active, add_counts(resources, mul_counts(time, active))) for t, s in zip(delays, sensible) if s and t >= time)
        yield from infeasible
        yield from feasible
        for bot in blueprint:
            continue
            if active[bot] >= max_bots[bot]:
                continue
            try:
                t = max(0, 1 + max((amount - resources[res] - 1) // active[res] for res, amount in blueprint[bot].items()))
                if t < time:
                    yield time - t - 1, active + Counts({bot: 1}), Counter({
                        res: resources[res] + active[res] * (t + 1) - blueprint[bot][res] for res in blueprint})
                else:
                    yield 0, active, Counter({res: resources[res] + active[res] * time for res in blueprint})
            except ZeroDivisionError:
                pass

    def find_max_geodes(blueprint: Counts[Counts], time: int):
        max_bots = Counts(*(max(res) for res in zip(*blueprint)))._replace(geode = time)
        queue = Q.PriorityQueue()
        start = (time, Counts(1, 0, 0, 0), Counts(0, 0, 0, 0))
        queue.put_nowait( (estimate_geodes(blueprint, *start), start) )
        result = 0
        states = 0
        while not queue.empty():
            states += 1
            est, state = queue.get_nowait()
            if -est < result:
                break
            for s in next_states(blueprint, max_bots, *state):
                t, active, resources = s
                result = max(result, resources.geode)
                if t > 0:
                    queue.put_nowait( (estimate_geodes(blueprint, *s), s) )

        return result

    xs = [find_max_geodes(bp, 24) for bp in blueprints]
    ys = [find_max_geodes(bp, 32) for bp in blueprints]
    part1 = sum((i + 1) * find_max_geodes(bp, 24) for i, bp in enumerate(blueprints))
    part2 = reduce(operator.mul, (find_max_geodes(bp, 32) for bp in blueprints[:3]))
    return part1, part2


def day20(filename: str):
    # list -> index ->
    data = [(idx, int(line.strip())) for idx, line in enumerate(open(filename).readlines())]
    encrypted = [i for i, _ in enumerate(data)]
    n = len(data)

    def move(i):
        idx, val = data[i % n]
        new_idx = (idx + val) % (n - 1)
        data[i % n] = new_idx, val

        encrypted[(idx + val) % (n - 1)] = i
        for j, (k, v) in enumerate(data):
            if (j != i) and idx < k <= new_idx:
                data[j] = (k - 1) % n, v
            elif (j != i) and new_idx <= k <= idx:
                data[j] = (k + 1) % n, v

    for i in range(len(data)):
        move(i)

    # build permutation
    permutation = {i: j for i, (j, _) in enumerate(data)}

    poss = {idx: val for idx, val in data}
    idx = next(i for i, v in data if v == 0)
    part1 = sum(poss[(idx + k * 1000) % n] for k in range(1, 4))

    return part1, 0


def day21(filename: str):
    matches = [re.match(r'(\w+)\: (?:(-?\d+)|(?:(\w+) (.) (\w+)))', line.strip()).groups() for line in
               open(filename).readlines()]
    values = {name: int(value) if value is not None else (left, op, right) for name, value, left, op, right in matches}

    def evaluate(name: str):
        rhs = values[name]
        if type(rhs) is not tuple:
            return rhs
        else:
            left, op, right = rhs
            result = eval(f'evaluate(left) {op} evaluate(right)',
                          dict(left=left, right=right, op=op, evaluate=evaluate))
            # values[name] = result
            return result

    part1 = evaluate('root')

    x = symbols('x')
    # values = {name: int(value) if value is not None else (left, op, right) for name, value, left, op, right in matches}
    values['humn'] = x
    left, _, right = values['root']
    _, solution = sympy.solve_linear(evaluate(left), evaluate(right), [x])
    return int(part1), int(solution)


def day22(filename: str):
    board, path = open(filename).read().split('\n\n')
    tiles = {(row + 1, col + 1): c for row, line in enumerate(board.split('\n')) for col, c in enumerate(line) if
             c in '.#'}
    walls = {pos for pos, ch in tiles.items() if ch == '#'}

    def left(row, col):
        col_new = max(c for r, c in tiles if r == row) if (row, col - 1) not in tiles else col - 1
        return (row, col) if (row, col_new) in walls else (row, col_new)

    def right(row, col):
        col_new = min(c for r, c in tiles if r == row) if (row, col + 1) not in tiles else col + 1
        return (row, col) if (row, col_new) in walls else (row, col_new)

    def down(row, col):
        row_new = min(r for r, c in tiles if c == col) if (row + 1, col) not in tiles else row + 1
        return (row, col) if (row_new, col) in walls else (row_new, col)

    def up(row, col):
        row_new = max(r for r, c in tiles if c == col) if (row - 1, col) not in tiles else row - 1
        return (row, col) if (row_new, col) in walls else (row_new, col)

    position = 1, min(col for row, col in tiles if row == 1)
    direction = dict(right=dict(R="down", L="up", score=0),
                     down=dict(R="left", L="right", score=1),
                     left=dict(R="up", L="down", score=2),
                     up=dict(R="right", L="left", score=3))

    facing = "up"
    test = list(map(re.Match.groups, re.finditer(r'(R|L)(\d+)', 'R' + path)))
    for turn, steps in test:
        facing = direction[facing][turn]
        for _ in range(int(steps)):
            position = up(*position) if facing == "up" else down(*position) if facing == "down" else left(
                *position) if facing == "left" else right(*position)

    row, col = position
    part1 = 1000 * row + 4 * col + direction[facing]["score"]
    return part1, 0


def day23(filename: str):
    data = open(filename).readlines()
    elves = {(row, col) for row, line in enumerate(data) for col, c in enumerate(line.strip()) if c == '#'}

    def nbors(positions, row, col):
        return {(row + r, col + c) for r, c in product([-1, 0, 1], repeat=2) if
                abs(r) + abs(c) != 0 and (row + r, col + c) in positions}

    def move(positions, row, col, index):
        ns = nbors(positions, row, col)
        if not ns:
            return False, (row, col)
        proposed = [
            lambda: (row - 1, col) if not ns & {(row - 1, col), (row - 1, col + 1), (row - 1, col - 1)} else None,
            lambda: (row + 1, col) if not ns & {(row + 1, col), (row + 1, col + 1), (row + 1, col - 1)} else None,
            lambda: (row, col - 1) if not ns & {(row, col - 1), (row - 1, col - 1), (row + 1, col - 1)} else None,
            lambda: (row, col + 1) if not ns & {(row, col + 1), (row - 1, col + 1), (row + 1, col + 1)} else None]
        for i in range(4):
            target = proposed[(index + i) % 4]()
            if target is not None:
                return True, target
        return False, (row, col)

    def round(index, positions: set[(int, int)]) -> set[(int, int)]:
        changes = {pos: move(positions, *pos, index) for pos in positions}
        if any(changed for changed, _ in changes.values()):
            clashes = Counter(target for _, target in changes.values())
            return {pos if clashes[target] > 1 else target for pos, (_, target) in changes.items()}
        else:
            return None

    i = 0
    positions = round(i, elves)
    while positions:
        i += 1
        if i == 10:
            maxr, minr = max(r for r, c in positions), min(r for r, c in positions)
            maxc, minc = max(c for r, c in positions), min(c for r, c in positions)
            part1 = (maxr - minr + 1) * (maxc - minc + 1) - len(positions)
        positions = round(i, positions)

    part2 = i + 1

    return part1, part2


def day24(filename: str):
    lines = open(filename).readlines()
    data = {(row - 1, col - 1): c for row, line in enumerate(lines) for col, c in enumerate(line.strip())}
    width, height = max(col for row, col in data), max(row for row, col in data)
    lcm = math.lcm(width, height)
    all_nrs = set(range(2 * lcm))

    def merge_ranges(ranges: list[int]) -> list[(int, int)]:
        it = iter(ranges)
        a = next(it)
        b = a + 1
        result = list()
        for c in it:
            if b < c:
                result.append((a, b))
                a, b = c, c + 1
            else:
                b = c + 1
        result.append((a, b))
        return result

    def blizzard_times(row: int, col: int) -> list[(int, int)]:
        lefts = set(((c - col) % width, width) for (r, c), ch in data.items() if r == row and ch == '<')
        rights = set(((col - c) % width, width) for (r, c), ch in data.items() if r == row and ch in '>')
        ups = set(((r - row) % height, height) for (r, c), ch in data.items() if c == col and ch in '^')
        downs = set(((row - r) % height, height) for (r, c), ch in data.items() if c == col and ch in 'v')
        blocked = set(b for a, m in lefts | rights | ups | downs for b in range(a, 2 * lcm, m))
        return merge_ranges(sorted(all_nrs - blocked))

    blizzards = {pos: blizzard_times(*pos) for pos, ch in data.items() if ch != '#'}

    def free_interval(row: int, col: int, time: int):
        intervals = blizzards[(row, col)]
        # find interval
        lo, hi = 0, len(intervals)
        while lo < hi:
            mid = (lo + hi) // 2
            a, b = intervals[mid]
            if a <= time < b:
                return a, b
            elif time < a:
                hi = mid
            else:
                lo = mid + 1
        return intervals[lo]

    def neighbours(row: int, col: int, time: int):
        _, until = free_interval(row, col, time)
        adjacent = filter(lambda pos: data.get(pos, '#') != '#',
                          ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)))
        for r, c in adjacent:
            soonest, _ = free_interval(r, c, time + 1)
            if soonest <= until:
                yield max(soonest - time, 1), (r, c)

    def find_path(time: int, start: (int, int), finish: (int, int)) -> int:
        visited = set()
        queue = Q.PriorityQueue()
        queue.put_nowait((time, start))
        while queue:
            time, (row, col) = queue.get_nowait()
            if (time, (row, col)) in visited:
                continue
            visited.add((time, (row, col)))
            if (row, col) == finish:
                return time
            for dt, pos in neighbours(row, col, time % lcm):
                queue.put_nowait((time + dt, pos))

    part1 = find_path(0, (-1, 0), (height, width - 1))
    back = find_path(part1, (height, width - 1), (-1, 0))
    part2 = find_path(back, (-1, 0), (height, width - 1))
    return part1, part2


def day25(filename: str):
    data = open(filename).read().split('\n')
    dec2snafu = {2: '2', 1: '1', 0: '0', -1: '-', -2: '='}
    snafu2dec = {'2': 2, '1': 1, '0': 0, '-': -1, '=': -2}

    def to_snafu(nr: int) -> str:
        digits = list()
        while nr > 0:
            digits = [(nr + 2) % 5 - 2] + digits
            nr = (nr + 2) // 5
        return ''.join(dec2snafu[digit] for digit in digits)

    def to_dec(snafu: str) -> int:
        return reduce(lambda nr, c: nr * 5 + snafu2dec[c], snafu, 0)

    return to_snafu(sum(to_dec(line) for line in data)), None


if __name__ == '__main__':
    solvers = [(key, value) for key, value in globals().items() if key.startswith("day") and callable(value)]
    solvers = sorted(((int(key.split('day')[-1]), value) for key, value in solvers), reverse=True)

    for idx, solver in reversed(solvers):
        if idx != 19:
            continue
        ns1 = time.process_time_ns()
        p1, p2 = solver(f"input/day{idx}.txt")
        ns2 = time.process_time_ns()
        print(f"day {idx} - part 1: {p1}, part 2: {p2}. time: {(ns2 - ns1) * 1e-9} seconds")
        # break
