#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : ILP.py 
# @File    : other_fx_fix.py
# @Author  : Wumh
# @Time    : 2026/1/4 13:54
# /mnt/data/other_fx.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from SC_FS import sc_fs


def _recompute_cell_cap_left(cell: np.ndarray) -> None:
    used_list = cell[1]
    used_sum = sum(float(e[1]) for e in used_list) if used_list else 0.0
    total = float(cell[0][0]) if cell[0] else 0.0
    cell[3] = [total - used_sum]


def _copy_sc_records_by_subflow_id(old_cell: np.ndarray, new_cell: np.ndarray, subflow_id: int) -> None:
    """
    将 old_cell 中 entry[0]==subflow_id 的记录复制到 new_cell，保持 0/1/2/4 四列对齐。
    """
    # ensure list columns

    old_used = old_cell[1]
    for idx, entry in enumerate(old_used):
        if int(entry[0]) != subflow_id:
            continue

        # 对齐复制
        cap_val = old_cell[0][idx] if idx < len(old_cell[0]) else (old_cell[0][0] if old_cell[0] else 0.0)
        path_val = old_cell[2][idx] if idx < len(old_cell[2]) else []
        dst_val = old_cell[4][idx] if idx < len(old_cell[4]) else -1

        new_cell[0].append(cap_val)
        new_cell[1].append(entry)
        new_cell[2].append(path_val)
        new_cell[4].append(dst_val)

    _recompute_cell_cap_left(new_cell)


def _remove_sc_records_by_subflow_id(cell: np.ndarray, subflow_id: int) -> None:
    """
    删除 cell 中所有 entry[0]==subflow_id 的记录，并保持 0/1/2/4 列对齐。
    """

    kept_0 = []
    kept_1 = []
    kept_2 = []
    kept_4 = []

    for i, entry in enumerate(cell[1]):
        if int(entry[0]) == subflow_id:
            continue
        if cell[0]:
            kept_0.append(cell[0][i])
        if cell[1]:
            kept_1.append(cell[1][i])
        if cell[2]:
            kept_2.append(cell[2][i])
        if cell[4]:
            kept_4.append(cell[4][i])

    cell[0] = kept_0
    cell[1] = kept_1
    cell[2] = kept_2
    cell[4] = kept_4

    # 如果空了，可以把 cap_left 设为 [0]（后续 heuristic 会按需初始化）
    if len(cell[1]) == 0:
        cell[3] = [0.0]
    else:
        _recompute_cell_cap_left(cell)


def extract_old_flow_info(flow_id: int, src: int, dst: int,
                          flow_acc_DP: np.ndarray, P2MP_SC_DP: np.ndarray) -> Dict[str, Any]:
    """
    从 DP 解中抽取端点信息：
      - src_small_id / des_small_id：对应 DP 的 subflow id（同一 orig flow 可能多 hop）
      - hub_idx / leaf_idx：端点 P2MP index
      - hub_sc_range：来自 flow_acc_DP 的 [9,10]
      - leaf_sc_range：来自 flow_acc_DP 的 [13,14]
      - hub_cap / leaf_cap：用于推断旧调制(<=500:25, >500:12.5)
    """
    info = {
        "src_small_id": -1,
        "des_small_id": -1,
        "hub_idx": -1,
        "leaf_idx": -1,
        "hub_sc_range": None,
        "leaf_sc_range": None,
        "hub_cap": 0.0,
        "leaf_cap": 0.0,
        "hub_sc_usage": None,
        "leaf_sc_usage": None,
    }

    # 取 DP 中属于该 orig flow 的所有 subflow
    rows = flow_acc_DP[flow_acc_DP[:, 6] == flow_id]
    for row in rows:
        if int(row[1]) == src:
            info["src_small_id"] = int(row[0])
            info["hub_idx"] = int(row[7])
            if int(row[9]) >= 0 and int(row[10]) >= 0:
                info["hub_sc_range"] = (int(row[9]), int(row[10]))
            info["hub_cap"] = float(row[4])

        if int(row[2]) == dst:
            info["des_small_id"] = int(row[0])
            info["leaf_idx"] = int(row[8])
            info["leaf_cap"] = float(row[4])

            if int(row[13]) >= 0 and int(row[14]) >= 0:
                info["leaf_sc_range"] = (int(row[13]), int(row[14]))

    # 统计该 flow 在源端(hub)侧每个 SC 的实际使用量
    if info["src_small_id"] != -1 and info["hub_idx"] != -1 and info["hub_sc_range"] is not None:
        a = int(src)
        hub_p = int(info["hub_idx"])
        sub_id = int(info["src_small_id"])
        hub_usage = {}
        # 遍历所有 SC，从 DP 的 P2MP_SC 里累加该 subflow 的用量
        for sc in range(P2MP_SC_DP.shape[2]):
            try:
                used_list = P2MP_SC_DP[a][hub_p][int(sc)][1]
            except Exception:
                continue
            if not used_list:
                continue
            for entry in used_list:
                if int(entry[0]) == sub_id:
                    hub_usage[int(sc)] = hub_usage.get(int(sc), 0.0) + float(entry[1])
        if hub_usage:
            info["hub_sc_usage"] = hub_usage

    # 统计该 flow 在目的端(leaf)侧每个 SC 的实际使用量
    if info["des_small_id"] != -1 and info["leaf_idx"] != -1 and info["leaf_sc_range"] is not None:
        b = int(dst)
        leaf_p = int(info["leaf_idx"])
        sub_id = int(info["des_small_id"])
        leaf_usage = {}
        # 遍历所有 SC，从 DP 的 P2MP_SC 里累加该 subflow 的用量
        for sc in range(P2MP_SC_DP.shape[2]):
            try:
                used_list = P2MP_SC_DP[b][leaf_p][int(sc)][1]
            except Exception:
                continue
            if not used_list:
                continue
            for entry in used_list:
                if int(entry[0]) == sub_id:
                    leaf_usage[int(sc)] = leaf_usage.get(int(sc), 0.0) + float(entry[1])
        if leaf_usage:
            info["leaf_sc_usage"] = leaf_usage

    return info


def build_virtual_topology(src: int, dst: int, topo_num: int, phy_pool: Dict[Tuple[int, int], List[Dict[str, Any]]],
                           break_node: int, new_node_P2MP: np.ndarray, band: float,
                           force_strategy1: bool,
                           orig_src_modu: int, orig_dst_modu: int,
                           RECONFIG_PENALTY: float, HOP_PENALTY: float) -> Tuple[np.ndarray, Dict[Tuple[int, int], Dict[str, Any]]]:
    """
    从每条虚拟边 (u, v) 的候选物理路径里选最低代价的一条 ，并写入虚拟拓扑：
    - 遍历 phy_pool 里每条虚拟边的候选路径列表
    - 跳过端点能力不足的虚拟边
    - 对每条候选计算代价 cost = 物理距离 + 跳数惩罚
    - 判断是否需要重构（端点调制不兼容）
        - 策略1：不兼容就丢弃该候选
        - 策略2：允许但加重构惩罚
    - 选代价最小的候选作为该虚拟边的物理映射
    - 最后返回更新后的 virtual_adj 和 best_phy_map
    """
    virtual_adj = np.full((topo_num, topo_num), np.inf)
    np.fill_diagonal(virtual_adj, 0.0)
    best_phy_map: Dict[Tuple[int, int], Dict[str, Any]] = {}

    # 简单节点能力检查：端口最大容量是否满足带宽
    node_capable = np.zeros(topo_num, dtype=bool)
    for n in range(topo_num):
        if n == break_node:
            node_capable[n] = False
            continue
        max_cap = 0.0
        for p in range(len(new_node_P2MP[n])):
            max_cap = max(max_cap, float(new_node_P2MP[n][p][4]))
        node_capable[n] = (max_cap >= band)

    # 端点硬件制式兼容性判定（仅对 src/dst 约束）
    def is_path_compatible(dist: float, hw: int) -> bool:
        if hw == 1:  # 16QAM short
            return dist <= 500
        if hw == 2:  # QPSK long
            return True
        return False

    # 遍历虚拟边，选取最小代价的物理候选
    for (u, v), candidates in phy_pool.items():
        if not node_capable[u] or not node_capable[v]:
            continue

        best_cost = np.inf
        best_candidate: Optional[Dict[str, Any]] = None

        for cand in candidates:
            phy_dist = float(cand["dist"])
            native_modu = 1 if phy_dist <= 500 else 2

            cost = phy_dist + HOP_PENALTY

            # 只对端点做兼容限制，中间节点不受限；任一端不兼容即视为重构
            reconfig_src = (u == src) and (not is_path_compatible(phy_dist, orig_src_modu))
            reconfig_dst = (v == dst) and (not is_path_compatible(phy_dist, orig_dst_modu))
            needs_reconfig = reconfig_src or reconfig_dst

            # 这里的 continue 是为了在强制策略1时直接丢弃“需要重构的候选路径”。
            # 因为 needs_reconfig 表示端点调制与原配置不兼容，而策略1要求完全复用原配置，所以一旦不兼容就跳过该候选，继续看下一个候选。
            if force_strategy1 and needs_reconfig:
                continue
            if (not force_strategy1) and needs_reconfig:
                # 策略2允许重构，但增加惩罚
                cost += RECONFIG_PENALTY

            if cost < best_cost:
                best_cost = cost
                best_candidate = {"path": cand["path"], "dist": phy_dist, "modu": native_modu, "cost": cost}

        if best_candidate is not None:
            virtual_adj[u][v] = best_cost
            best_phy_map[(u, v)] = best_candidate

    return virtual_adj, best_phy_map


def manage_s1_reservation_by_copy(flow_list: List[Any], metadata_map: Dict[int, Dict[str, Any]],
                                  flow_acc_DP: np.ndarray, node_P2MP_DP: np.ndarray, P2MP_SC_DP: np.ndarray,
                                  new_node_P2MP: np.ndarray, new_P2MP_SC_1: np.ndarray,
                                  action: str = "reserve") -> None:
    """
    S1 预占位/回滚：
      - reserve: 从 DP 解中把该 flow 的端点 SC 占用记录复制到 new_P2MP_SC_1，并扣减 new_node_P2MP 剩余容量
      - rollback: 删除这些复制记录，并恢复 new_node_P2MP 剩余容量

    注意：这里以 DP 的 subflow_id 来匹配 P2MP_SC[*][*][*][1] 的 entry[0]
    """
    if action not in ("reserve", "rollback"):
        raise ValueError(f"Unknown action={action}")

    for flow in flow_list:
        # flow: [orig_flow_id, src, dst, band, ...]
        f_id = int(flow[0])
        src = int(flow[1])
        dst = int(flow[2])
        band = float(flow[3])

        # 优先使用 metadata_map 里保存的端点子流与端口信息
        info = metadata_map.get(f_id, {})
        src_small_id = int(info.get("src_small_id", -1)) if info else -1
        des_small_id = int(info.get("des_small_id", -1)) if info else -1
        hub_p = int(info.get("hub_idx", -1)) if info else -1
        leaf_p = int(info.get("leaf_idx", -1)) if info else -1
        hub_range = info.get("hub_sc_range", None) if info else None
        leaf_range = info.get("leaf_sc_range", None) if info else None

        if hub_p == -1 or hub_range is None or src_small_id == -1 or leaf_p == -1 or leaf_range is None or des_small_id == -1:
            raise ValueError(f"metadata incomplete for flow {f_id}")

        # 只对源端 hub 预占/回滚
        if hub_p != -1 and hub_range is not None and src_small_id != -1:
            sc0, sc1 = int(hub_range[0]), int(hub_range[1])
            if sc0 >= 0 and sc1 >= 0:
                if action == "reserve":
                    # 预占带宽：扣除端口剩余带宽
                    new_node_P2MP[src][hub_p][4] -= band
                    # 预占端口状态：标记为 Hub
                    if new_node_P2MP[src][hub_p][5] == -1:
                        new_node_P2MP[src][hub_p][5] = node_P2MP_DP[src][hub_p][5]
                    if new_node_P2MP[src][hub_p][2] == 0:
                        new_node_P2MP[src][hub_p][2] = 1

                    # 预占时隙/SC：复制该流在对应时隙范围内的占用记录
                    for s in range(sc0, sc1 + 1):
                        _copy_sc_records_by_subflow_id(P2MP_SC_DP[src][hub_p][s], new_P2MP_SC_1[src][hub_p][s], src_small_id)
                else:
                    # 回滚带宽：加回端口剩余带宽
                    new_node_P2MP[src][hub_p][4] += band
                    # 回滚时隙/SC：删除该流在对应时隙范围内的占用记录
                    for s in range(sc0, sc1 + 1):
                        _remove_sc_records_by_subflow_id(new_P2MP_SC_1[src][hub_p][s], src_small_id)

        # 只对目的端 leaf 预占/回滚
        if leaf_p != -1 and leaf_range is not None and des_small_id != -1:
            l0, l1 = int(leaf_range[0]), int(leaf_range[1])
            if l0 >= 0 and l1 >= 0:
                if action == "reserve":
                    # 预占带宽：扣除端口剩余带宽
                    new_node_P2MP[dst][leaf_p][4] -= band
                    # 预占端口状态：标记为 Leaf
                    if new_node_P2MP[dst][leaf_p][5] == -1:
                        new_node_P2MP[dst][leaf_p][5] = node_P2MP_DP[dst][leaf_p][5]
                    if new_node_P2MP[dst][leaf_p][2] == 0:
                        new_node_P2MP[dst][leaf_p][2] = 2

                    # 预占时隙/SC：复制该流在对应时隙范围内的占用记录
                    for s in range(l0, l1 + 1):
                        _copy_sc_records_by_subflow_id(P2MP_SC_DP[dst][leaf_p][s], new_P2MP_SC_1[dst][leaf_p][s], des_small_id)
                else:
                    # 回滚带宽：加回端口剩余带宽
                    new_node_P2MP[dst][leaf_p][4] += band
                    # 回滚时隙/SC：删除该流在对应时隙范围内的占用记录
                    for s in range(l0, l1 + 1):
                        _remove_sc_records_by_subflow_id(new_P2MP_SC_1[dst][leaf_p][s], des_small_id)


def build_fs_meta_np_from_p2mp_sc(P2MP_SC: np.ndarray, node_P2MP: np.ndarray,
                                 fs_total: int, link_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于 P2MP_SC 重建：
      - P2MP_FS[u,p,fs_rel,1/2/4]：每个 P2MP 的 6 个相对 FS 上有哪些 flow/路径/dst
      - link_FS_meta[l,fs_abs,1/2]：每条链路每个绝对 FS 上有哪些 flow/路径
    """
    topo_num = P2MP_SC.shape[0]
    p2mp_num = P2MP_SC.shape[1]

    P2MP_FS = np.empty((topo_num, p2mp_num, 6, 9), dtype=object)
    for u in range(topo_num):
        for p in range(p2mp_num):
            for fs in range(6):
                for c in range(9):
                    P2MP_FS[u, p, fs, c] = []
                P2MP_FS[u, p, fs, 3] = None

    link_num = int(np.max(link_index))
    link_FS_meta = np.empty((link_num, fs_total, 5), dtype=object)
    for l in range(link_num):
        for fs in range(fs_total):
            link_FS_meta[l, fs, 0] = 0
            link_FS_meta[l, fs, 1] = []
            link_FS_meta[l, fs, 2] = []
            link_FS_meta[l, fs, 3] = None
            link_FS_meta[l, fs, 4] = None

    sc_max = {1: 0, 2: 3, 3: 15}

    for u in range(topo_num):
        for p in range(p2mp_num):
            base_fs = int(node_P2MP[u][p][5])
            if base_fs < 0:
                continue
            p_type = int(node_P2MP[u][p][3])
            if p_type not in (1, 2, 3):
                continue

            for sc in range(16):
                if sc > sc_max[p_type]:
                    continue
                used_list = P2MP_SC[u, p, sc, 1]
                if not used_list:
                    continue

                paths_list = P2MP_SC[u, p, sc, 2]
                dst_list = P2MP_SC[u, p, sc, 4]
                src_node_list = P2MP_SC[u, p, sc, 5]
                src_p_list = P2MP_SC[u, p, sc, 6]
                dst_node_list = P2MP_SC[u, p, sc, 7]
                dst_p_list = P2MP_SC[u, p, sc, 8]

                fs_s_abs = base_fs + sc_fs(p_type, sc, 1)
                fs_e_abs = base_fs + sc_fs(p_type, sc, 2)
                if fs_s_abs < 0 or fs_e_abs >= fs_total:
                    continue

                for idx, entry in enumerate(used_list):
                    path_1b = paths_list[idx] if idx < len(paths_list) else []
                    dst = dst_list[idx] if idx < len(dst_list) else -1
                    src_node = src_node_list[idx] if idx < len(src_node_list) else -1
                    src_p = src_p_list[idx] if idx < len(src_p_list) else -1
                    dst_node = dst_node_list[idx] if idx < len(dst_node_list) else -1
                    dst_p = dst_p_list[idx] if idx < len(dst_p_list) else -1

                    for fs_abs in range(fs_s_abs, fs_e_abs + 1):
                        fs_rel = fs_abs - base_fs
                        if 0 <= fs_rel < 6:
                            P2MP_FS[u, p, fs_rel, 1].append(entry)
                            P2MP_FS[u, p, fs_rel, 2].append(path_1b)
                            P2MP_FS[u, p, fs_rel, 4].append(dst)
                            P2MP_FS[u, p, fs_rel, 5].append(int(src_node))
                            P2MP_FS[u, p, fs_rel, 6].append(int(src_p))
                            P2MP_FS[u, p, fs_rel, 7].append(int(dst_node))
                            P2MP_FS[u, p, fs_rel, 8].append(int(dst_p))

                    if len(path_1b) >= 2:
                        for x1, y1 in zip(path_1b[:-1], path_1b[1:]):
                            x0, y0 = int(x1) - 1, int(y1) - 1
                            if x0 < 0 or y0 < 0:
                                continue
                            l_id = int(link_index[x0][y0])
                            if l_id <= 0:
                                continue
                            l0 = l_id - 1
                            for fs_abs in range(fs_s_abs, fs_e_abs + 1):
                                link_FS_meta[l0, fs_abs, 1].append(entry)
                                link_FS_meta[l0, fs_abs, 2].append(path_1b)

    return P2MP_FS, link_FS_meta
