def square(n):
    return n ** 2

lst =[]
for x in range(1, 11):
    lst.append(x)
print(lst)

lst = [x for x in range(1,11)]
lst1 = [x*x for x in range(1,11)]
lst2 = [square(x) for x in range(1,11)]
print('lst = ', lst)
print('lst1 = ', lst1)
print('lst2 = ', lst2)


