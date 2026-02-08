#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : ILP.py 
# @File    : Yens_k_paths.py
# @Author  : Wumh
# @Time    : 2025/12/7 17:02
import heapq

# 定义无穷大，用于表示节点之间没有边
INF = float('inf')


def get_path_cost(matrix, path):
    """根据邻接矩阵计算路径总开销"""
    cost = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        weight = matrix[u][v]
        if weight == INF:
            return INF
        cost += weight
    return cost


def dijkstra_matrix(matrix, start, end, blocked_edges=None, blocked_nodes=None):
    """
    基于邻接矩阵的 Dijkstra 实现
    matrix: N x N 的二维列表或数组
    """
    if blocked_edges is None: blocked_edges = set()
    if blocked_nodes is None: blocked_nodes = set()

    n = len(matrix)

    # 距离表，初始化为无穷大
    distances = {node: INF for node in range(n)}
    distances[start] = 0

    # 优先队列 (cost, current_node, path)
    pq = [(0, start, [start])]

    while pq:
        d, u, path = heapq.heappop(pq)

        if u == end:
            return path, d

        # 性能优化：如果当前路径开销已经大于已知最短路，跳过
        if d > distances[u]:
            continue

        # 遍历所有可能的邻居 v (0 到 n-1)
        for v in range(n):
            weight = matrix[u][v]

            # 条件：
            # 1. 存在边 (weight != INF)
            # 2. 节点 v 没有被屏蔽
            # 3. 边 (u, v) 没有被屏蔽
            # 4. 不是自环 (weight > 0，视具体需求而定，通常最短路不走自环)
            if weight != INF and weight >= 0:
                if v not in blocked_nodes and (u, v) not in blocked_edges:
                    new_dist = d + weight
                    if new_dist < distances[v]:
                        distances[v] = new_dist
                        heapq.heappush(pq, (new_dist, v, path + [v]))

    return None, INF


def yen_k_shortest_paths_matrix(matrix, start, end, K):
    """
    基于邻接矩阵的 Yen's 算法主逻辑
    """
    n = len(matrix)

    # 结果列表 A
    A = []
    # 候选路径堆 B
    B = []

    # 1. 第一条最短路径
    path, cost = dijkstra_matrix(matrix, start, end)

    if not path:
        return []

    A.append({'path': path, 'cost': cost})

    # 2. 寻找第 k 条路径
    for k in range(1, K):
        prev_path = A[-1]['path']

        # 偏离点循环
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[:i + 1]

            blocked_edges = set()
            blocked_nodes = set()

            # (1) 屏蔽已经在 A 中的边
            for path_data in A:
                p = path_data['path']
                if len(p) > i and p[:i + 1] == root_path:
                    blocked_edges.add((p[i], p[i + 1]))

            # (2) 屏蔽 root_path 中的节点 (保持无环)
            for node in root_path[:-1]:
                blocked_nodes.add(node)

            # (3) 计算偏离路径
            spur_path, spur_cost = dijkstra_matrix(matrix, spur_node, end, blocked_edges, blocked_nodes)

            if spur_path:
                total_path = root_path[:-1] + spur_path
                total_cost = get_path_cost(matrix, total_path)

                potential_item = (total_cost, tuple(total_path))  # tuple才能hash放入set/比较
                if potential_item not in B:
                    heapq.heappush(B, potential_item)

        if not B:
            break

        cost, path_tuple = heapq.heappop(B)
        path = list(path_tuple)
        A.append({'path': path, 'cost': cost})

    return A


# ==========================================
# 测试示例 (邻接矩阵)
# ==========================================
if __name__ == "__main__":
    # 假设有 5 个节点：0, 1, 2, 3, 4
    # INF 代表无连接，0 代表自己到自己

    # 构建一个简单的图矩阵
    # 0 -> 1 (w=3), 0 -> 2 (w=2)
    # 1 -> 3 (w=4)
    # 2 -> 1 (w=1), 2 -> 3 (w=2), 2 -> 4 (w=3)
    # 3 -> 4 (w=2)
    # 1 -> 4 (w=5) - 这是一条增加的直接路径

    adj_matrix = [[float('inf'), 250, float('inf'), float('inf'), float('inf'), 250],
                             [250, float('inf'), 250, float('inf'), float('inf'), 250],
                             [float('inf'), 250, float('inf'), 250, 250, float('inf')],
                             [float('inf'), float('inf'), 250, float('inf'), 250, float('inf')],
                             [float('inf'), float('inf'), 250, 250, float('inf'), 250],
                             [250, 250, float('inf'), float('inf'), 250, float('inf')]]

    start_node = 0
    end_node = 3
    K = 3

    results = yen_k_shortest_paths_matrix(adj_matrix, start_node, end_node, K)
    print(results)
