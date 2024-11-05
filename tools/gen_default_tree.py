import json

edges = [
    [0, 1], [0, 2], [0, 3], [0, 4],
    [1, 5], [1, 6], [1, 7],
    [2, 8], [2, 9],
    [3, 10], [3, 11],
    [4, 12],
    [5, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [16, 17],
    [17, 18],
    [18, 19]
]

N = max(sum(edges, [])) + 1

print("N = {}".format(N))

childs = {}

for i in range(len(edges)):
    u, v = tuple(edges[i])
    if u not in childs:
        childs[u] = []
    childs[u].append(v)

for u in range(N):
    if u not in childs:
        childs[u] = []

print(json.dumps(childs, indent=4))
