#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : code 
# @File    : topology.py
# @Author  : Wumh
# @Time    : 2024/7/16 21:34

import numpy as np

def topology(type):
    if type == 1:
        topo_num = 6
        topo_matrix = np.array([[0, 1, 0, 0, 0, 1],
                                [1, 0, 1, 0, 0, 1],
                                [0, 1, 0, 1, 1, 0],
                                [0, 0, 1, 0, 1, 0],
                                [0, 0, 1, 1, 0, 1],
                                [1, 1, 0, 0, 1, 0]])
        topo_dis = np.array([[float('inf'), 250, float('inf'), float('inf'), float('inf'), 250],
                             [250, float('inf'), 250, float('inf'), float('inf'), 250],
                             [float('inf'), 250, float('inf'), 250, 250, float('inf')],
                             [float('inf'), float('inf'), 250, float('inf'), 250, float('inf')],
                             [float('inf'), float('inf'), 250, 250, float('inf'), 250],
                             [250, 250, float('inf'), float('inf'), 250, float('inf')]])
        link_num = 16
        link_index = np.array([[0, 1, 0, 0, 0, 2],
                               [9, 0, 4, 0, 0, 3],
                               [0, 12, 0, 7, 6, 0],
                               [0, 0, 15, 0, 8, 0],
                               [0, 0, 14, 16, 0, 5],
                               [10, 11, 0, 0, 13, 0]])
    elif type == 2:
        topo_num = 24
        topo_matrix = np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]])
        topo_dis = np.array([[float('inf'), 200, float('inf'), float('inf'), float('inf'), 250, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
                             [200, float('inf'), 220, float('inf'), float('inf'), 190, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
                             [float('inf'), 220, float('inf'), 50, 220, float('inf'), 200, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
                             [float('inf'), float('inf'), 50, float('inf'), 160, float('inf'), 170, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
                             [float('inf'), float('inf'), 220, 160, float('inf'), float('inf'), float('inf'), 240, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
                             [200, 190, float('inf'), float('inf'), float('inf'), float('inf'), 200, float('inf'), 240, float('inf'), 380, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
                             [float('inf'), float('inf'), 200, 170, float('inf'), 200, float('inf'), 230, 200, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
                             [float('inf'), float('inf'), float('inf'), float('inf'), 240, float('inf'), 230, float('inf'), float('inf'), 190, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
                             [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 240, 200, float('inf'), float('inf'), 200, 280, 200, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
                             [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 190, 200, float('inf'), float('inf'), float('inf'), 190, 170, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
                             [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 380, float('inf'), float('inf'), 280, float('inf'), float('inf'), 180, float('inf'), float('inf'), 260, float('inf'), float('inf'), float('inf'), 520, float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
                             [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 200, float('inf'), 180, float('inf'), 180, float('inf'), float('inf'), 200, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
                             [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 190, float('inf'), 180, float('inf'), 130, float('inf'), float('inf'), 220, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
                             [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 170, float('inf'), float('inf'), 130, float('inf'), float('inf'), float('inf'), float('inf'), 240, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
                             [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 260, float('inf'), float('inf'), float('inf'), float('inf'), 120, float('inf'), float('inf'), float('inf'), 260, float('inf'), float('inf'), float('inf'), float('inf')],
                             [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 200, float('inf'), float('inf'), 120, float('inf'), 200, float('inf'), float('inf'), float('inf'), 200, 160, float('inf'), float('inf')],
                             [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 220, float('inf'), float('inf'), 200, float('inf'), 160, float('inf'), float('inf'), float('inf'), 170, 200, float('inf')],
                             [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 240, float('inf'), float('inf'), 160, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 180],
                             [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 520, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 240, float('inf'), float('inf'), float('inf'), float('inf')],
                             [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 260, float('inf'), float('inf'), float('inf'), 240, float('inf'), 140, float('inf'), float('inf'), float('inf')],
                             [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 200, float('inf'), float('inf'), float('inf'), 140, float('inf'), 60, float('inf'), float('inf')],
                             [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 160, 170, float('inf'), float('inf'), float('inf'), 60, float('inf'), 120, float('inf')],
                             [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 200, float('inf'), float('inf'), float('inf'), float('inf'), 120, float('inf'), 180],
                             [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 180, float('inf'), float('inf'), float('inf'), float('inf'), 180, float('inf')]])
        link_num = 86
        link_index = np.array([[ 0,  1,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                               [44,  0,  2,  0,  0,  5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0, 45,  0,  7,  3,  0,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0, 50,  0,  8,  0,  9,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0, 46, 51,  0,  0,  0, 10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                [47, 48,  0,  0,  0,  0, 11,  0, 14,  0, 13,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0, 49, 52,  0, 54,  0, 12, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0, 53,  0, 55,  0,  0, 16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0, 57, 58,  0,  0, 18, 17, 22,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0, 59, 61,  0,  0,  0, 23, 24,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0, 56,  0,  0, 60,  0,  0, 21,  0,  0, 20,  0,  0,  0, 19,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0, 65,  0, 64,  0, 25,  0,  0, 27,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0, 66,  0, 68,  0, 26,  0,  0, 28,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0, 67,  0,  0, 69,  0,  0,  0,  0, 29,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 63,  0,  0,  0,  0, 31,  0,  0,  0, 30,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 70,  0,  0, 74,  0, 32,  0,  0,  0, 36, 37,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 71,  0,  0, 75,  0, 33,  0,  0,  0, 38, 39,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 72,  0,  0, 76,  0,  0,  0,  0,  0,  0, 40],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 62,  0,  0,  0,  0,  0,  0,  0,  0, 34,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 73,  0,  0,  0, 77,  0, 35,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 79,  0,  0,  0, 78,  0, 41,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 80, 81,  0,  0,  0, 84,  0, 42,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 82,  0,  0,  0,  0, 85,  0, 43],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 83,  0,  0,  0,  0, 86,  0]
])
    return topo_num, topo_matrix, topo_dis, link_num, link_index

if __name__ == "__main__":
    topo_num, topo_matrix, topo_dis, link_num, link_index = topology(1)
    print(topo_num, topo_matrix, topo_dis, link_num, link_index)