import numpy as np

# a = 0
# for i in range(100):
#     num = np.random.randint(100)
#     if num < a:
#         a = num
#     print(num)
#
# print(a,"============22")


num = []
for i in range(100):
    a = np.random.randint(100)
    num.append(a)
    print(a)
    if len(num) == 2:
        print(num)
        b = min(num)
        num[0] = b
        num.pop()

print(num[0],"444")

def selectionSort(nb):
    for i in range(len(nb)):
        for j in range(i,len(nb)):
            if nb[i] > nb[j]:
                nb[j],nb[i]=nb[i],nb[j]
    return nb

c = []
for i in range(100):
    a = np.random.randint(100)
    c.append(a)

print(selectionSort(c))