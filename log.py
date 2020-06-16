from tools import file_path

result = []
for line in open(file_path("result.txt")):
    if 'Average' in line:
        result.append(line.split()[-1])
assert len(result) % 6 == 0
for i in range(int(len(result) / 6)):
    print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}".format(
        result[i * 6],
        result[i * 6 + 1],
        result[i * 6 + 2],
        result[i * 6 + 3],
        result[i * 6 + 4],
        result[i * 6 + 5]
    ))
