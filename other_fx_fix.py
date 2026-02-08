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


def _as_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _cell_total_cap(cell: np.ndarray) -> float:
    """
    推断单个 SC 的“总容量”：
      1) 若 cell[0] 非空，则 max(cell[0]) 视为 total_cap
      2) 否则 total_cap = cap_left + sum(used)
      3) 都没有则 0
    """
    cap_list = _ensure_list(cell[0])
    if cap_list:
        return max(_as_float(v, 0.0) for v in cap_list)

    used_list = _ensure_list(cell[1])
    used_sum = sum(_as_float(e[1], 0.0) for e in used_list) if used_list else 0.0

    cap_left_list = _ensure_list(cell[3])
    cap_left = _as_float(cap_left_list[0], 0.0) if cap_left_list else 0.0
    return cap_left + used_sum


def _recompute_cell_cap_left(cell: np.ndarray) -> None:
    used_list = _ensure_list(cell[1])
    used_sum = sum(_as_float(e[1], 0.0) for e in used_list) if used_list else 0.0
    total = _cell_total_cap(cell)
    cell[3] = [total - used_sum]


def _copy_sc_records_by_subflow_id(old_cell: np.ndarray, new_cell: np.ndarray, subflow_id: int) -> None:
    """
    将 old_cell 中 entry[0]==subflow_id 的记录复制到 new_cell，保持 0/1/2/4 四列对齐。
    """
    # ensure list columns
    for k in (0, 1, 2, 4):
        if not isinstance(new_cell[k], list):
            new_cell[k] = []
    if not isinstance(new_cell[3], list):
        new_cell[3] = []

    old_used = _ensure_list(old_cell[1])
    for idx, entry in enumerate(old_used):
        if _as_int(entry[0]) != subflow_id:
            continue

        # 防止重复复制
        if any(_as_int(e[0]) == subflow_id for e in _ensure_list(new_cell[1])):
            # 但一个 cell 可能有多条同 subflow 的记录？通常不会，这里选择跳过
            continue

        # 对齐复制
        cap_val = _ensure_list(old_cell[0])[idx] if idx < len(_ensure_list(old_cell[0])) else _cell_total_cap(old_cell)
        path_val = _ensure_list(old_cell[2])[idx] if idx < len(_ensure_list(old_cell[2])) else []
        dst_val = _ensure_list(old_cell[4])[idx] if idx < len(_ensure_list(old_cell[4])) else -1

        new_cell[0].append(cap_val)
        new_cell[1].append(entry)
        new_cell[2].append(path_val)
        new_cell[4].append(dst_val)

    _recompute_cell_cap_left(new_cell)


def _remove_sc_records_by_subflow_id(cell: np.ndarray, subflow_id: int) -> None:
    """
    删除 cell 中所有 entry[0]==subflow_id 的记录，并保持 0/1/2/4 列对齐。
    """
    for k in (0, 1, 2, 4):
        if not isinstance(cell[k], list):
            cell[k] = []
    if not isinstance(cell[3], list):
        cell[3] = []

    keep_idx = [i for i, e in enumerate(cell[1]) if _as_int(e[0]) != subflow_id]

    cell[0] = [cell[0][i] for i in keep_idx] if cell[0] else []
    cell[1] = [cell[1][i] for i in keep_idx] if cell[1] else []
    cell[2] = [cell[2][i] for i in keep_idx] if cell[2] else []
    cell[4] = [cell[4][i] for i in keep_idx] if cell[4] else []

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
      - leaf_sc_range：通过扫描 P2MP_SC_DP[dst][leaf_idx][:][1] 找到该 subflow 的 sc 范围
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
    }

    # 取 DP 中属于该 orig flow 的所有 subflow
    rows = flow_acc_DP[flow_acc_DP[:, 6] == flow_id]
    for row in rows:
        if _as_int(row[1]) == src:
            info["src_small_id"] = _as_int(row[0])
            info["hub_idx"] = _as_int(row[7])
            if _as_int(row[9]) >= 0 and _as_int(row[10]) >= 0:
                info["hub_sc_range"] = (_as_int(row[9]), _as_int(row[10]))
            info["hub_cap"] = _as_float(row[4], 0.0)

        if _as_int(row[2]) == dst:
            info["des_small_id"] = _as_int(row[0])
            info["leaf_idx"] = _as_int(row[8])
            info["leaf_cap"] = _as_float(row[4], 0.0)

            leaf_idx = info["leaf_idx"]
            sub_id = info["des_small_id"]
            if leaf_idx != -1 and sub_id != -1:
                used_scs = []
                for s in range(16):
                    used_list = _ensure_list(P2MP_SC_DP[dst][leaf_idx][s][1])
                    if any(_as_int(e[0]) == sub_id for e in used_list):
                        used_scs.append(s)
                if used_scs:
                    info["leaf_sc_range"] = (min(used_scs), max(used_scs))

    return info


def build_virtual_topology(src: int, dst: int, topo_num: int, phy_pool: Dict[Tuple[int, int], List[Dict[str, Any]]],
                           break_node: int, new_node_P2MP: np.ndarray, band: float,
                           force_strategy1: bool,
                           orig_src_modu: int, orig_dst_modu: int,
                           RECONFIG_PENALTY: float, HOP_PENALTY: float) -> Tuple[np.ndarray, Dict[Tuple[int, int], Dict[str, Any]]]:
    """
    构建虚拟拓扑矩阵 + 每条虚拟边对应的最佳物理候选(含 path/dist/modu/cost)
    其中 force_strategy1=True 时会过滤掉不兼容旧硬件的物理候选
    """
    virtual_adj = np.full((topo_num, topo_num), np.inf)
    np.fill_diagonal(virtual_adj, 0.0)
    best_phy_map: Dict[Tuple[int, int], Dict[str, Any]] = {}

    # 简单节点能力检查
    node_capable = np.zeros(topo_num, dtype=bool)
    for n in range(topo_num):
        if n == break_node:
            node_capable[n] = False
            continue
        max_cap = 0.0
        for p in range(len(new_node_P2MP[n])):
            max_cap = max(max_cap, float(new_node_P2MP[n][p][4]))
        node_capable[n] = (max_cap >= band)

    def is_path_compatible(dist: float, hw: int) -> bool:
        if hw == 1:  # 16QAM short
            return dist <= 500
        if hw == 2:  # QPSK long
            return True
        return False

    for (u, v), candidates in phy_pool.items():
        if not node_capable[u] or not node_capable[v]:
            continue

        best_cost = np.inf
        best_candidate: Optional[Dict[str, Any]] = None

        for cand in candidates:
            phy_dist = float(cand["dist"])
            native_modu = 1 if phy_dist <= 500 else 2

            cost = phy_dist + HOP_PENALTY

            # only constrain endpoints
            ok_src = True if u != src else is_path_compatible(phy_dist, orig_src_modu)
            ok_dst = True if v != dst else is_path_compatible(phy_dist, orig_dst_modu)
            needs_reconfig = not (ok_src and ok_dst)

            if force_strategy1 and needs_reconfig:
                continue
            if (not force_strategy1) and needs_reconfig:
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
        f_id = int(flow[0])

        # 该 orig flow 可能有多 hop -> 遍历 DP 中属于它的所有 subflow
        rows = flow_acc_DP[flow_acc_DP[:, 6] == f_id]
        for row in rows:
            sub_id = int(row[0])
            a = int(row[1])
            b = int(row[2])
            band = float(row[3])

            hub_p = int(row[7])
            leaf_p = int(row[8])
            sc0 = int(row[9])
            sc1 = int(row[10])

            # --- hub side (a, hub_p, [sc0,sc1]) ---
            if hub_p != -1 and sc0 >= 0 and sc1 >= 0:
                if action == "reserve":
                    new_node_P2MP[a][hub_p][4] -= band
                    if new_node_P2MP[a][hub_p][5] == -1:
                        new_node_P2MP[a][hub_p][5] = node_P2MP_DP[a][hub_p][5]
                    if new_node_P2MP[a][hub_p][2] == 0:
                        new_node_P2MP[a][hub_p][2] = 1

                    for s in range(sc0, sc1 + 1):
                        _copy_sc_records_by_subflow_id(P2MP_SC_DP[a][hub_p][s], new_P2MP_SC_1[a][hub_p][s], sub_id)
                else:
                    new_node_P2MP[a][hub_p][4] += band
                    for s in range(sc0, sc1 + 1):
                        _remove_sc_records_by_subflow_id(new_P2MP_SC_1[a][hub_p][s], sub_id)

            # --- leaf side (b, leaf_p, leaf_sc_range inferred) ---
            if leaf_p != -1:
                used_scs = []
                for s in range(16):
                    used_list = _ensure_list(P2MP_SC_DP[b][leaf_p][s][1])
                    if any(int(e[0]) == sub_id for e in used_list):
                        used_scs.append(s)

                if used_scs:
                    l0, l1 = min(used_scs), max(used_scs)
                    if action == "reserve":
                        new_node_P2MP[b][leaf_p][4] -= band
                        if new_node_P2MP[b][leaf_p][5] == -1:
                            new_node_P2MP[b][leaf_p][5] = node_P2MP_DP[b][leaf_p][5]
                        if new_node_P2MP[b][leaf_p][2] == 0:
                            new_node_P2MP[b][leaf_p][2] = 2

                        for s in range(l0, l1 + 1):
                            _copy_sc_records_by_subflow_id(P2MP_SC_DP[b][leaf_p][s], new_P2MP_SC_1[b][leaf_p][s], sub_id)
                    else:
                        new_node_P2MP[b][leaf_p][4] += band
                        for s in range(l0, l1 + 1):
                            _remove_sc_records_by_subflow_id(new_P2MP_SC_1[b][leaf_p][s], sub_id)


def build_fs_meta_np_from_p2mp_sc(P2MP_SC: np.ndarray, node_P2MP: np.ndarray,
                                 fs_total: int, link_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于 P2MP_SC 重建：
      - P2MP_FS[u,p,fs_rel,1/2/4]：每个 P2MP 的 6 个相对 FS 上有哪些 flow/路径/dst
      - link_FS_meta[l,fs_abs,1/2]：每条链路每个绝对 FS 上有哪些 flow/路径
    """
    topo_num = P2MP_SC.shape[0]
    p2mp_num = P2MP_SC.shape[1]

    P2MP_FS = np.empty((topo_num, p2mp_num, 6, 5), dtype=object)
    for u in range(topo_num):
        for p in range(p2mp_num):
            for fs in range(6):
                P2MP_FS[u, p, fs, 0] = []
                P2MP_FS[u, p, fs, 1] = []
                P2MP_FS[u, p, fs, 2] = []
                P2MP_FS[u, p, fs, 3] = None
                P2MP_FS[u, p, fs, 4] = []

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
            base_fs = _as_int(node_P2MP[u][p][5])
            if base_fs < 0:
                continue
            p_type = _as_int(node_P2MP[u][p][3])
            if p_type not in (1, 2, 3):
                continue

            for sc in range(16):
                if sc > sc_max[p_type]:
                    continue
                used_list = _ensure_list(P2MP_SC[u, p, sc, 1])
                if not used_list:
                    continue

                paths_list = _ensure_list(P2MP_SC[u, p, sc, 2])
                dst_list = _ensure_list(P2MP_SC[u, p, sc, 4])

                fs_s_abs = base_fs + sc_fs(p_type, sc, 1)
                fs_e_abs = base_fs + sc_fs(p_type, sc, 2)
                if fs_s_abs < 0 or fs_e_abs >= fs_total:
                    continue

                for idx, entry in enumerate(used_list):
                    path_1b = paths_list[idx] if idx < len(paths_list) else []
                    dst = dst_list[idx] if idx < len(dst_list) else -1

                    for fs_abs in range(fs_s_abs, fs_e_abs + 1):
                        fs_rel = fs_abs - base_fs
                        if 0 <= fs_rel < 6:
                            P2MP_FS[u, p, fs_rel, 1].append(entry)
                            P2MP_FS[u, p, fs_rel, 2].append(path_1b)
                            P2MP_FS[u, p, fs_rel, 4].append(dst)

                    if isinstance(path_1b, list) and len(path_1b) >= 2:
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
