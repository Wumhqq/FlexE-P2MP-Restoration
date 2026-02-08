#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : ILP.py 
# @File    : Heuristic_al.py
# @Author  : Wumh
# @Time    : 2025/12/6 20:34

import numpy as np
import SC_FS_list
import topology as tp
from k_shortest_path import k_shortest_path

def Heuristic_algorithm(affected_flow, new_flow_acc, new_node_flow, new_node_P2MP, break_node, Tbox_num, Tbox_P2MP,
                    new_P2MP_SC_1, new_link_FS, node_flow_DP, node_P2MP_DP, flow_acc_DP, link_FS_DP, P2MP_SC_DP, flow_path_DP):

    flow_num = len(affected_flow)
    M = 10000
    P2MP_num = Tbox_num * Tbox_P2MP
    topo_num, topo_matrix, topo_dis, link_num, link_index = tp.topology(1)
    type_P2MP = [[1, 1, 1], [2, 4, 2], [3, 16, 6]]

    MAX_REACH = {
        "DP-16QAM": 500,  # km
        "DP-QPSK": M  # km
    }

    # 2. 算法权重参数 (根据之前的讨论设置)
    ALPHA_DIST = 1.0  # 距离权重系数
    HOP_PENALTY = 1000.0  # 跳数惩罚 (起步价)
    RECONFIG_PENALTY = 10000.0  # 策略2的重构惩罚 (核武器级惩罚)
    K_PATHS = 5  # Yen's 算法搜索的路径数量
    INF = float('inf')

    """
        输入: 
          - num_nodes: 节点总数
          - get_ksp_func: 一个函数，输入(src, dst)，返回你描述的 K条路径列表
        输出:
          - virtual_adj_matrix: 虚拟层的邻接矩阵 (存权重)
          - physical_mapping: 字典, key=(u,v), value=物理路径节点列表
        """
    # ... (前文参数定义保持不变) ...

    num_nodes = topo_num

    # ==========================================
    # 1. 初始化虚拟矩阵 (使用 Numpy)
    # ==========================================
    # 使用 np.inf (无穷大) 填充矩阵，表示默认都不连通
    virtual_adj_matrix = np.full((num_nodes, num_nodes), np.inf)

    # 对角线置为 0 (自己到自己距离为 0)
    np.fill_diagonal(virtual_adj_matrix, 0)

    # 初始化物理路径映射表 (这个因为长度不固定，还是得用字典，没法做成简单矩阵)
    physical_mapping = {}

    # ==========================================
    # 2. 构建虚拟拓扑 (填充矩阵)
    # ==========================================
    print(f"正在构建虚拟拓扑 (Nodes: {num_nodes})...")

    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src == dst: continue  # 对角线已处理，跳过

            # 调用 K-SP 寻找物理路径
            # 注意：src+1, dst+1 是因为你的 k_shortest_path 可能用的是 1-based 索引
            k_paths_output = k_shortest_path(topo_dis, src + 1, dst + 1, 1)

            best_valid_path = None
            best_valid_dist = INF

            for path_entry in k_paths_output:
                path_nodes = path_entry[0]  # e.g., [1, 4, 2]
                phy_cost = path_entry[1]  # Physical Distance
                best_valid_path = path_nodes
                best_valid_dist = phy_cost
                break  # 找到最短的一条即可

            # 如果找到了合法路径，填入矩阵
            if best_valid_path is not None:
                # 计算权重: 物理距离 + 跳数惩罚
                weight = best_valid_dist + HOP_PENALTY

                # 【矩阵赋值】直接修改 numpy 数组对应位置
                virtual_adj_matrix[src][dst] = weight

                # 记录映射关系 (Tuple 作为 key)
                physical_mapping[(src, dst)] = best_valid_path


    print("a")


