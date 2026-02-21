#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
# 初始化资源表：把输入的拆解流 disass_flows_info 做规范化排序、分组，
# 并构建节点-端口的 P2MP 资源矩阵、每端口的 SC 使用记录、链路频谱向量等。
def initialize_resource_tables(flows_info, flows_num, topo_num, link_num, Tbox_num, Tbox_P2MP):
    # P2MP 类型定义：type_id, 背包可容纳的 SC 数量, 对应占用的连续 FS 块大小
    type_P2MP = [[1, 1, 1], [2, 4, 2], [3, 16, 6]]
    # 解释：
    # - 每个元素为 [type_id, 可用SC数量, 所需连续FS块大小]
    # - type_id=1 => 1个SC、占用1个FS；type_id=2 => 4个SC、占用2个FS；type_id=3 => 16个SC、占用6个FS

    # 每个节点的 P2MP 端口总数，以及用于按索引均分型号的两个分界点
    _p2mp_total = Tbox_num * Tbox_P2MP
    _t1 = _p2mp_total // 3
    _t2 = 2 * _p2mp_total // 3
    # 解释：
    # - _p2mp_total：每节点的端口总数（Tbox_num * Tbox_P2MP）
    # - _t1/_t2：把端口索引按 1/3 与 2/3 切分，便于均分三种型号

    # 链路频谱使用向量：每条链路 358 个 FS 单元，后续分配成功才会累加占用
    link_FS = np.zeros((link_num, 358), dtype=int)
    # 解释：
    # - 形状为 (link_num, 358)，每列代表一个FS位置；数值为占用计数（0=空闲，>0 表示被占用次数）

    # 按源/目的节点排序，并重写 flow_id，确保后续索引连续且可控
    temp = np.array(flows_info, dtype=object)
    sorted_indices = np.lexsort((temp[:, 2], temp[:, 1]))
    flows_info_sorted = temp[sorted_indices]
    for i, f in enumerate(flows_info_sorted):
        f[0] = i
    # flows_info_sorted 每条子流的 7 列含义：
    # [0] 子流ID（重编号为 0..N-1）
    # [1] 源节点（0-based）
    # [2] 目的节点（0-based）
    # [3] 带宽（Gbps）
    # [4] SC_cap（单个SC的可承载带宽，可能为小数，如12.5）
    # [5] SC_num（满足带宽需求的SC个数，按5Gbps整数TS粒度向上取整）
    # [6] 原始flow序号（用于回溯）
        
    # 按源节点分组，scr_flows[n] 收集源为 n 的所有子流（七列基础信息）
    scr_flows = np.empty(topo_num, dtype=object)
    for n in range(topo_num):
        scr_flows[n] = []
        for flow in flows_info_sorted:
            if int(flow[1]) == n:
                scr_flows[n].append(flow[0:7])
    # 解释：
    # - scr_flows 形状为 (topo_num,) 的 object 数组，每个元素是列表
    # - 列表内每项为某源节点 n 的子流，包含上述 0..6 七列基础信息

    # 节点-端口矩阵与各类记录的初始化
    node_P2MP = np.zeros((topo_num, _p2mp_total, 6), dtype=int)
    node_flow = np.empty((topo_num, _p2mp_total, 1), dtype=object)
    flow_acc = np.zeros((flows_num, 15), dtype=object)
    P2MP_SC = np.empty((topo_num, _p2mp_total, 16, 9), dtype=object)
    P2MP_FS = np.empty((topo_num, _p2mp_total, 6, 9), dtype=object)
    flow_path = np.empty((flows_num, 4), dtype=object)
    # 解释：
    # - node_P2MP[u][p][*] 六列属性：
    #   [0] 所属Tbox序号 = floor(p / Tbox_P2MP)
    #   [1] 端口序号 p（0-based）
    #   [2] 使用标记 used_flag：0未使用；1为源节点hub；2为目的节点leaf
    #   [3] 端口型号 type_id（按索引均分：前1/3为3，中间1/3为2，最后1/3为1）
    #   [4] 剩余容量（Gbps），初始化为400，分配后按带宽扣减
    #   [5] 中心绝对FS起点 hub_fs0，初始化-1（未选），分配后为整数索引
    # - node_flow[u][p][0] 为该端口承载的子流列表（每项为子流的7列基础信息）
    # - P2MP_SC[u][p][s][c] 为该端口SC记录：s=0..15（16个SC），c=0..8（九个槽位，均为列表）
    #   c=0 caps：该SC的容量记录（float SC_cap）
    #   c=1 flows：使用记录 [子流ID, 使用带宽Gbps, 原flow序号]
    #   c=2 phys_paths：对应一次追加的物理路径节点序列（1-based）
    #   c=3 remaining_cap：剩余容量列表（后处理填入 cap - sum(used)）
    #   c=4 dest_labels：目的节点标签
    #   c=5 src_labels：源节点标签
    #   c=6 hub_ports：源端hub端口编号
    #   c=7 dst_labels：目的节点标签（与 c=4 一致，用于 FS 固定 leaf）
    #   c=8 leaf_ports：目的端leaf端口编号
    #   说明：发送端/发送P2MP/接收端/接收P2MP 可由索引确定
    #   发送端=当前 u，发送P2MP=当前 p；接收端=dest_labels，接收P2MP在 flow_acc[fid][8]
    # - P2MP_FS[u][p][fs_rel][c] 为相对FS记录：fs_rel=0..5，c=0..8（九个槽位，均为列表）
    #   c=0 caps：占位（当前未使用）
    #   c=1 flows：使用记录 [子流ID, 使用带宽Gbps, 原flow序号]
    #   c=2 phys_paths：对应一次追加的物理路径节点序列（1-based）
    #   c=3 remaining_cap：占位（当前未使用）
    #   c=4 dest_labels：目的节点标签
    #   c=5 src_labels：源节点标签
    #   c=6 hub_ports：源端hub端口编号
    #   c=7 dst_labels：目的节点标签
    #   c=8 leaf_ports：目的端leaf端口编号
    # - flow_path[fid][*] 为子流的路径/链路记录：
    #   [0] flow_id_list：记录自身子流ID（通常 append fid）
    #   [1] used_links：0-based 链路编号序列（涉及的所有链路，可能包含重复）
    #   [2] phys_nodes：1-based 物理路径节点序列
    #   [3] orig_flow_labels：原始flow序号标签（f[6]），用于关联原始流
    for u in range(topo_num):
        for p in range(_p2mp_total):
            # 每端口的基础属性：所属 Tbox、端口序号、剩余容量(400Gbps)、中心频率未选(-1)
            node_P2MP[u][p][0] = math.floor(p / Tbox_P2MP)
            node_P2MP[u][p][1] = p
            node_P2MP[u][p][4] = 400
            node_P2MP[u][p][5] = -1
            for s in range(16):
                for c in range(9):
                    P2MP_SC[u][p][s][c] = []
            for fs in range(6):
                for c in range(9):
                    P2MP_FS[u][p][fs][c] = []

            # 端口型号按索引均分：前 1/3 为 type=3，中间 1/3 为 type=2，最后为 type=1
            if p < _t1:
                node_P2MP[u][p][3] = 3
            elif p < _t2:
                node_P2MP[u][p][3] = 2
            else:
                node_P2MP[u][p][3] = 1
            node_flow[u][p][0] = []

    for i in range(flows_num):
        # 复制七列基础信息到 flow_acc；初始化路径/链路记录为空
        flow_acc[i][0:7] = flows_info_sorted[i][0:7]
        flow_path[i][0] = []
        flow_path[i][1] = []
        flow_path[i][2] = []
        flow_path[i][3] = []
        # flow_acc[i] 的 15 列含义：
        # [0] 子流ID（重编号）
        # [1] 源节点（0-based）
        # [2] 目的节点（0-based）
        # [3] 带宽（Gbps）
        # [4] SC_cap（float）
        # [5] SC_num（int）
        # [6] 原始flow序号
        # [7] hub端口编号（源节点端口索引，分配后填）
        # [8] leaf端口编号（目的节点端口索引，分配后填）
        # [9] sc_start（起始SC索引，分配后填）
        # [10] sc_end（结束SC索引，含端，分配后填）
        # [11] fs_s_abs（绝对FS起点，含端，分配后填）
        # [12] fs_e_abs（绝对FS终点，含端，分配后填）
        # [13] leaf_sc_start（目的端SC起点，分配后填）
        # [14] leaf_sc_end（目的端SC终点，分配后填）
        flow_acc[i][7:15] = [-1] * 8
        
    return (type_P2MP, _p2mp_total, _t1, _t2, link_FS, flows_info_sorted, scr_flows,
            node_P2MP, node_flow, flow_acc, P2MP_SC, P2MP_FS, flow_path)
