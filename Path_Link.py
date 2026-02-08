#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : code 
# @File    : Path_Link.py
# @Author  : Wumh
# @Time    : 2025/6/11 0:25
import numpy as np


def get_links_from_path(path, link_index):
    """
    根据路径和邻接矩阵获取路径中使用的link编号。

    参数:
        path (list[int]): 节点路径（从1开始编号，例如 [1, 2, 3, 4]）
        link_index (np.ndarray): 邻接矩阵，link_index[i][j] 表示从节点i到j的link编号（i, j从0开始）

    返回:
        list[int]: 按顺序返回路径中使用的link编号
    """
    # 将节点编号从1-based转换为0-based
    path_zero_based = [p - 1 for p in path]

    # 提取路径中的所有link编号
    used_links = []
    for i in range(len(path_zero_based) - 1):
        u = path_zero_based[i]
        v = path_zero_based[i + 1]
        link = link_index[u][v]
        if link == 0:
            raise ValueError(f"无效路径：节点 {u + 1} 到 {v + 1} 之间没有链接")
        used_links.append(link)

    return used_links
