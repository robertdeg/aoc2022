if __name__ == '__main__':
    cals = sorted(map(lambda s: sum(map(int, s.split('\n'))), open("input/day1.txt").read().split('\n\n')), reverse=True)[:3]
    print(f"day 1 - part 1: {cals[0]}, part 2: {sum(cals)}")

