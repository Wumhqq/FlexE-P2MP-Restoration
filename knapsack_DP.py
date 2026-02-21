#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : code 
# @File    : knapsack_DP.py
# @Author  : Wumh
# @Time    : 2024/7/16 21:12
import numpy as np
def knapsack_DP(capacity, list_items):
    """
    测试数据：
    n = 6  物品的数量，
    c = 10 书包能承受的重量，
    w = [2, 2, 3, 1, 5, 2] 每个物品的重量，
    v = [2, 3, 1, 5, 4, 3] 每个物品的价值
    """
    # 使用 Python 列表来动态存储元素
    item_weight_list = []
    item_value_list = []
    items = np.array(list_items, dtype=object)
    row, vol = items.shape
    # 提取每个项目的重量和值
    if vol >= 6:
        for i in items:
            item_weight_list.append(i[3])
            item_value_list.append(i[3])
    else:
        for i in items:
            item_weight_list.append(i[1])
            item_value_list.append(i[1])
    # 将 Python 列表转换为 NumPy 数组
    item_weight = np.array(item_weight_list, dtype=int)
    item_value = np.array(item_value_list, dtype=int)

    # 置零，表示初始状态
    value = np.zeros((row + 1, capacity + 1), dtype=int)
    bag = np.empty((row + 1, capacity + 1), dtype=object)

    for j in range(capacity + 1):
        bag[0, j] = []
    for i in range(row + 1):
        bag[i, 0] = []

    for i in range(1, row + 1):
        for j in range(1, capacity + 1):
            value[i, j] = value[i - 1, j]
            bag[i, j] = [] # 因为是列表
            # 背包总容量够放当前物体，遍历前一个状态考虑是否置换
            if j >= item_weight[i - 1] and value[i, j] < value[i - 1, j - item_weight[i - 1]] + item_value[i - 1]:
                value[i, j] = value[i - 1, j - item_weight[i - 1]] + item_value[i - 1]
                bag[i, j].extend(bag[i - 1, j - item_weight[i - 1]])
                bag[i, j].append(items[i - 1][0])
            else:
                bag[i, j].extend(bag[i - 1, j])

    # for x in value:
    #     print(x)
    # print(bag)
    final_items = bag[row][capacity]
    final_value = value[row][capacity]
    return value, bag, final_value, final_items

if __name__ == '__main__':
    capacity = 16
    items = np.array([[0, 0, 1, 3],
                     [1, 2, 4, 1],
                     [2, 5, 2, 7],
                     [3, 2, 5, 1],
                     [4, 0, 3, 5],
                     [5, 3, 1, 2]])
    value, bag, final_value, final_items= knapsack_DP(capacity, items)
    print(bag[4, 4][1])
