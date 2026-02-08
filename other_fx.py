#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : ILP.py 
# @File    : other_fx.py
# @Author  : Wumh
# @Time    : 2025/12/30 22:27
import numpy as np
import math
from Path_Link import get_links_from_path
from modu_format_Al import modu_format_Al
from SC_FS import sc_fs

def extract_old_flow_info(f_id, src, dst, flow_acc_DP, P2MP_SC_DP):
    """从历史记录中提取流的原始配置"""
    info = {
        'src_small_id': -1, 'des_small_id': -1,
        'hub_idx': -1, 'leaf_idx': -1,
        'hub_sc_range': None, 'leaf_sc_range': None,
        'hub_cap': 0, 'leaf_cap': 0
    }

    # 使用 numpy 布尔索引筛选
    relevant_rows = flow_acc_DP[flow_acc_DP[:, 6] == f_id]
    # row[4] = 25/12
    for row in relevant_rows:
        if row[1] == src:
            info['src_small_id'] = row[0]
            info['hub_idx'] = row[7]
            info['hub_sc_range'] = (row[9], row[10])
            info['hub_cap'] = row[4]

        if row[2] == dst:
            info['des_small_id'] = row[0]
            info['leaf_idx'] = row[8]
            info['leaf_cap'] = row[4]

            leaf_idx = row[8]
            if leaf_idx != -1:
                used_scs = []
                sub_flow_id = row[0]
                for s in range(16):
                    try:
                        flow_list = P2MP_SC_DP[dst][leaf_idx][s][1]
                        for entry in flow_list:
                            if entry[0] == sub_flow_id:
                                used_scs.append(s)
                                break
                    except IndexError:
                        pass
                if used_scs:
                    info['leaf_sc_range'] = (min(used_scs), max(used_scs))
                else:
                    info['leaf_sc_range'] = None
    return info


def is_path_compatible(phy_dist, hardware_modu):
    """
    判断物理距离是否兼容特定的硬件调制格式。
    hardware_modu: 1 (16QAM/短距), 2 (QPSK/长距)
    """
    # 如果硬件是 16QAM (Type 1)，物理路径必须 <= 500km
    if hardware_modu == 1:
        return phy_dist <= 500

    # 如果硬件是 QPSK (Type 2)，它理论上可以跑任意距离（包括短距）
    # 这里假设 Type 2 没有下限，只有上限（假设无穷大或根据你的ILP M值）
    if hardware_modu == 2:
        return True

    return False


def build_virtual_topology(src, dst, topo_num, phy_pool, break_node, new_node_P2MP, band,
                           force_strategy1, orig_src_modu, orig_dst_modu,
                           RECONFIG_PENALTY, HOP_PENALTY):
    """构建虚拟拓扑矩阵"""
    virtual_adj = np.full((topo_num, topo_num), np.inf)
    np.fill_diagonal(virtual_adj, 0)
    best_phy_map = {}

    node_capable = np.zeros(topo_num, dtype=bool)
    for n in range(topo_num):
        if n == break_node:
            node_capable[n] = False
            continue
        # 简单容量检查
        max_cap = 0
        for p in range(len(new_node_P2MP[n])):
            if new_node_P2MP[n][p][4] > max_cap:
                max_cap = new_node_P2MP[n][p][4]
        node_capable[n] = (max_cap >= band)

    for (u, v), candidates in phy_pool.items():
        if not node_capable[u] or not node_capable[v]:
            continue

        min_cost = np.inf
        best_candidate = None

        for path_data in candidates:
            # path_data: [path_nodes(0-based), distance]
            phy_dist = path_data[1]
            # 这条路"原生"支持的最佳格式 (用于 Strategy 2 参考)
            native_modu = 1 if phy_dist <= 500 else 2
            cost = phy_dist + HOP_PENALTY

            # --- 兼容性检查 (核心修改) ---
            is_compatible_src = True
            is_compatible_dst = True

            # 只有当 u 是源节点时，才受源端旧硬件限制
            if u == src:
                is_compatible_src = is_path_compatible(phy_dist, orig_src_modu)
            # 只有当 v 是宿节点时，才受宿端旧硬件限制
            if v == dst:
                is_compatible_dst = is_path_compatible(phy_dist, orig_dst_modu)
            # 如果两端都兼容，就不需要重构
            needs_reconfig = not (is_compatible_src and is_compatible_dst)

            if force_strategy1:
                # 强约束模式 (Tier 1): 必须兼容旧硬件
                if needs_reconfig:
                    continue
            else:
                # 宽松模式 (Tier 2): 不兼容则加罚分
                if needs_reconfig:
                    cost += RECONFIG_PENALTY

            if cost < min_cost:
                min_cost = cost
                # 记录路径信息
                # 注意：这里我们暂时记录 native_modu，
                # 但在 get_phy_path_info 里会根据 strict_s1 决定实际用哪个 modu 计算容量
                best_candidate = {
                    'path': path_data[0],
                    'dist': phy_dist,
                    'modu': native_modu,
                    'cost': cost
                }

        if min_cost < np.inf:
            virtual_adj[u][v] = min_cost
            best_phy_map[(u, v)] = best_candidate

    return virtual_adj, best_phy_map

def _copy_sc_records_by_flow_id(old_cell, new_cell, match_flow_id):
    """
    将 old_cell 中属于 match_flow_id 的记录，按“并行列对齐”的方式复制到 new_cell。
    old_cell/new_cell 结构：长度为 5 的 list，其中 [0],[1],[2],[4] 是并行记录列，[3] 通常不用。
    """
    # 保险：保证 new_cell 有 5 列
    while len(new_cell) < 5:
        new_cell.append([])

    # 逐条扫描 old_cell[1]（flow 条目列），找到 entry[0] == match_flow_id 的记录索引 idx
    for idx, entry in enumerate(old_cell[1]):
        if entry[0] != match_flow_id:
            continue

        # 并行复制：0/1/2/4 四列必须保持同一 idx 对齐
        # 备注：old_cell[3] 在你工程里一般没用（初始化为空），这里不动它
        new_cell[0].append(old_cell[0][idx])
        new_cell[1].append(old_cell[1][idx])
        new_cell[2].append(old_cell[2][idx])
        new_cell[4].append(old_cell[4][idx])

def _remove_sc_records_by_flow_id(cell, match_flow_id):
    """
    从 cell 中删除所有 entry[0] == match_flow_id 的记录，并保证 0/1/2/4 四列对齐。
    """
    while len(cell) < 5:
        cell.append([])

    keep_idx = [i for i, e in enumerate(cell[1]) if e[0] != match_flow_id]
    cell[0] = [cell[0][i] for i in keep_idx]
    cell[1] = [cell[1][i] for i in keep_idx]
    cell[2] = [cell[2][i] for i in keep_idx]
    cell[4] = [cell[4][i] for i in keep_idx]
    # cell[3] 不动（通常为空）

def manage_s1_reservation_by_copy(flow_list, metadata_map, P2MP_SC_DP, new_P2MP_SC_1, new_node_P2MP,
                                  action='reserve'):
    """
    Strategy 1 资源预占位（Pre-reservation）：
    - reserve：把故障前端点（源 Hub / 宿 Leaf）的 SC 占用记录从旧表搬运到新表，并扣减 new_node_P2MP 的剩余容量
    - rollback：若 S1 后续失败，释放预占的端点容量，并从新表删除搬运过来的 SC 记录

    关键修正点（相对你 PDF 的最小改动）：
    1) 匹配 P2MP_SC 条目时，用 sub-flow id（src_small_id/des_small_id），而不是原始流 id（f_id）
    2) copy/rollback 时同时维护 [0],[1],[2],[4] 四个并行列表，防止错位
    """
    for flow in flow_list:
        f_id = flow[0]
        src = flow[1]
        dst = flow[2]
        band = flow[3]

        info = metadata_map[f_id]

        # sub-flow id：用于匹配 P2MP_SC[*][*][*][1] 里 entry[0]
        # 如果 metadata 里没给（极端情况），退化用 f_id（但通常你的 P2MP_SC 里不会存 f_id）
        src_small_id = info.get('src_small_id', f_id)
        des_small_id = info.get('des_small_id', f_id)

        # ----------------------------
        # 1) 源端 (Hub) 预占位
        # ----------------------------
        u = src
        h_idx = info.get('hub_idx', -1)
        hub_range = info.get('hub_sc_range', None)

        if h_idx != -1 and hub_range:
            s_start, s_end = hub_range

            if action == 'reserve':
                # 1) 扣除该 Hub P2MP 的剩余容量（保持你原逻辑不变）
                new_node_P2MP[u][h_idx][4] -= band

                # 2) 逐 SC，把旧表中属于“源端 sub-flow”的记录搬运到新表
                for s in range(s_start, s_end + 1):
                    old_cell = P2MP_SC_DP[u][h_idx][s]
                    new_cell = new_P2MP_SC_1[u][h_idx][s]
                    _copy_sc_records_by_flow_id(old_cell, new_cell, src_small_id)


            else:  # rollback
                new_node_P2MP[u][h_idx][4] += band
                for s in range(s_start, s_end + 1):
                    cell = new_P2MP_SC_1[u][h_idx][s]
                    _remove_sc_records_by_flow_id(cell, src_small_id)

        # ----------------------------
        # 2) 宿端 (Leaf) 预占位
        # ----------------------------
        v = dst
        l_idx = info.get('leaf_idx', -1)
        leaf_range = info.get('leaf_sc_range', None)

        if l_idx != -1 and leaf_range:
            s_start, s_end = leaf_range

            if action == 'reserve':
                new_node_P2MP[v][l_idx][4] -= band
                for s in range(s_start, s_end + 1):
                    old_cell = P2MP_SC_DP[v][l_idx][s]
                    new_cell = new_P2MP_SC_1[v][l_idx][s]
                    _copy_sc_records_by_flow_id(old_cell, new_cell, des_small_id)

            else:  # rollback
                new_node_P2MP[v][l_idx][4] += band
                for s in range(s_start, s_end + 1):
                    cell = new_P2MP_SC_1[v][l_idx][s]
                    _remove_sc_records_by_flow_id(cell, des_small_id)



def build_fs_meta_np_from_p2mp_sc(
        new_P2MP_SC_1,
        new_node_P2MP,
        sc_fs,
        fs_total: int,
        link_index,  # topology() 的 link_index: 0=无边, 正数=link_id(1..link_num)
        link_num: int  # topology() 的 link_num
):
    """
    从 new_P2MP_SC_1 + new_node_P2MP + sc_fs 直接生成：
      - new_P2MP_FS_1: np.ndarray(object), (N, P, fs_total, 5)
      - link_FS_meta:  np.ndarray(object), (link_num, fs_total, 5)

    cell 结构（对齐 SC）:
      [0]=[]        (cap_list, FS 层不用)
      [1]=list      (used_list, 直接 append SC 的条目)
      [2]=list      (path_list, 直接 append path)
      [3]=None      (空出)
      [4]=list      (dst_list, 直接 append dst)
    """
    num_nodes = len(new_P2MP_SC_1)
    num_p = len(new_P2MP_SC_1[0]) if num_nodes > 0 else 0

    # -------- 初始化 object 数组（注意：必须逐格放新 list，避免共享引用）--------
    new_P2MP_FS_1 = np.empty((num_nodes, num_p, 6, 5), dtype=object)
    for u in range(num_nodes):
        for p in range(num_p):
            for fs in range(6):
                new_P2MP_FS_1[u, p, fs, 0] = []
                new_P2MP_FS_1[u, p, fs, 1] = []
                new_P2MP_FS_1[u, p, fs, 2] = []
                new_P2MP_FS_1[u, p, fs, 3] = None
                new_P2MP_FS_1[u, p, fs, 4] = []

    link_FS_meta = np.empty((link_num, fs_total, 5), dtype=object)
    for l in range(link_num):
        for fs in range(fs_total):
            link_FS_meta[l, fs, 0] = []
            link_FS_meta[l, fs, 1] = []
            link_FS_meta[l, fs, 2] = []
            link_FS_meta[l, fs, 3] = None
            link_FS_meta[l, fs, 4] = []

    # -------- 主循环：SC -> FS，并投影到 link 层 --------
    for u in range(num_nodes):
        for p in range(num_p):
            p_type = int(new_node_P2MP[u][p][3])
            base_fs = int(new_node_P2MP[u][p][5])
            if base_fs < 0:
                continue  # P2MP 未激活

            sc_table = new_P2MP_SC_1[u][p]
            for sc_idx in range(len(sc_table)):
                cell = sc_table[sc_idx]
                if not isinstance(cell, list) or len(cell) < 5:
                    continue

                used_list = cell[1]
                if not isinstance(used_list, list) or len(used_list) == 0:
                    continue

                # paths：轻量处理（仅为能遍历 link）
                # - 若 cell[2] 是 [1,5,7] 这种：视为单条 path
                # - 若 cell[2] 是 [[...],[...]]：视为多条 path
                paths_raw = cell[2]
                if isinstance(paths_raw, list) and len(paths_raw) > 0 and isinstance(paths_raw[0], int):
                    paths = [paths_raw]
                elif isinstance(paths_raw, list):
                    paths = paths_raw
                else:
                    paths = []

                # dsts：轻量处理
                dst_raw = cell[4]
                if isinstance(dst_raw, int):
                    dsts = [dst_raw]
                elif isinstance(dst_raw, list):
                    dsts = dst_raw
                else:
                    dsts = []

                # SC -> 绝对 FS 区间
                if p_type == 1 and sc_idx <= 0 or p_type == 2 and sc_idx <= 3 or p_type == 3 and sc_idx <= 15:
                    fs_s = base_fs + int(sc_fs(p_type, sc_idx, 1))
                    fs_e = base_fs + int(sc_fs(p_type, sc_idx, 2))
                if fs_s < 0 or fs_e >= fs_total or fs_s > fs_e:
                    continue

                # 预先把每条 path 映射成 link 序列（0-based link id）
                # 注意：path 节点为 1-based，要先转 0-based 再查 link_index
                links_per_path = []
                for pp in paths:
                    if not isinstance(pp, list) or len(pp) < 2:
                        continue
                    nodes0 = [int(x) - 1 for x in pp]
                    links0 = []
                    ok = True
                    for a0, b0 in zip(nodes0[:-1], nodes0[1:]):
                        lid = int(link_index[a0][b0])  # 1..link_num
                        if lid <= 0:
                            ok = False
                            break
                        links0.append(lid - 1)  # 0..link_num-1
                    if ok:
                        links_per_path.append(links0)
                        break

                # 把该 SC 的 used/path/dst 投影到每个 FS
                for fs in range(fs_s, fs_e + 1):
                    # ---- endpoint FS ----
                    f_used = new_P2MP_FS_1[u, p, fs - base_fs, 1]
                    f_path = new_P2MP_FS_1[u, p, fs - base_fs, 2]
                    f_dst = new_P2MP_FS_1[u, p, fs - base_fs, 4]

                    # used：直接加（不合并、不去重）
                    for ent in used_list:
                        f_used.append(ent)

                    # path：直接加（不去重）
                    for pp in paths:
                        f_path.append(pp)

                    # dst：直接加（不去重）
                    for d in dsts:
                        f_dst.append(d)

                    # ---- link FS ----
                    for links0 in links_per_path:
                        for l in links0:
                            if l < 0 or l >= link_num:
                                continue
                            l_used = link_FS_meta[l, fs, 1]
                            l_path = link_FS_meta[l, fs, 2]
                            l_dst = link_FS_meta[l, fs, 4]

                            for ent in used_list:
                                l_used.append(ent)
                            for pp in paths:
                                l_path.append(pp)
                            for d in dsts:
                                l_dst.append(d)

    return new_P2MP_FS_1, link_FS_meta




