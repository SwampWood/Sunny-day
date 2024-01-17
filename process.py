with open('Srok8c.ddl') as file:
    s0 = file.readlines()[1:-1]
    s = [i.strip().split('//')[-1][1:] for i in s0]
    s2 = [i.split()[0:4] for i in s0]
    print(*s, sep='\n')
    sp = []
    for i in s2:
        if i[0] in ['KEY(I)', 'MIT']:
            sp += [i[1]]
        else:
            if i[1] == 'Q':
                sp += [i[0]]
            elif i[1] == 'D':
                sp += [i[0][:-1] + 'ЗНАК)']
            elif i[1] == 'Q1':
                sp += [i[0][:-1] + '1)']
    print(sp)
    sl = {}
    for i in range(len(sp)):
        sl[sp[i]] = s[i]
    print(str(sl).replace('\'', '\"').replace(', ', ',\n'))