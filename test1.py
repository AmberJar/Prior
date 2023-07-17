from functools import reduce


my_list = [2, 3, 5]

result = reduce(lambda x, y: x * y, my_list)
print(result) # 👉️ 30