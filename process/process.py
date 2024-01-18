access_name = None
s = [None]
s2 = [None]


def access(name):
    global s
    global s2
    global access_name
    access_name = name
    with open(name) as file:
        s0 = file.readlines()[1:-2]
        s = [i.strip().split('//')[-1][1:] for i in s0]
        s2 = [i.split()[0:4] for i in s0]


def delim(name):
    if access_name != name:
        access(name)
    all_space = []
    useful_space = []
    for i in s2:
        all_space += [int(i[2][3])]
        useful_space += [int(i[3][3])]
    return all_space, useful_space