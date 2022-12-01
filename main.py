if __name__ == '__main__':
    cals = list(map(lambda s: sum(map(int, s.split('\n'))), open("input/day1.txt").read().split('\n\n')))
    print(f"day 1 - part 1: {max(cals)}")
    print(f"day 1 - part 2: {sum(sorted(cals, reverse=True)[:3])}")

