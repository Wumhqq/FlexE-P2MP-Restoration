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
        if int(entry[0]) == subflow_id:
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
    # 留下cell中所有 entry[0]!=subflow_id 的记录
    kept_0 = []
    kept_1 = []
    kept_2 = []
    kept_4 = []

    for i, entry in enumerate(cell[1]):
        if int(entry[0]) != subflow_id:
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


def _copy_fs_records_by_subflow_id(old_cell: np.ndarray, new_cell: np.ndarray, subflow_id: int) -> None:
    """把 old_cell 中属于指定 subflow 的 FS 记录复制到 new_cell。"""
    old_used = old_cell[1]
    for idx, entry in enumerate(old_used):
        if int(entry[0]) != subflow_id:
            continue
        path_val = old_cell[2][idx] if idx < len(old_cell[2]) else []
        dst_val = old_cell[4][idx] if idx < len(old_cell[4]) else -1
        src_node_val = old_cell[5][idx] if idx < len(old_cell[5]) else -1
        src_p_val = old_cell[6][idx] if idx < len(old_cell[6]) else -1
        dst_node_val = old_cell[7][idx] if idx < len(old_cell[7]) else -1
        dst_p_val = old_cell[8][idx] if idx < len(old_cell[8]) else -1
        new_cell[1].append(entry)
        new_cell[2].append(path_val)
        new_cell[4].append(dst_val)
        new_cell[5].append(src_node_val)
        new_cell[6].append(src_p_val)
        new_cell[7].append(dst_node_val)
        new_cell[8].append(dst_p_val)


def _remove_fs_records_by_subflow_id(cell: np.ndarray, subflow_id: int) -> None:
    """删除 cell 中属于指定 subflow 的 FS 记录，并保持各列对齐。"""
    kept_1 = []
    kept_2 = []
    kept_4 = []
    kept_5 = []
    kept_6 = []
    kept_7 = []
    kept_8 = []

    for i, entry in enumerate(cell[1]):
        if int(entry[0]) == subflow_id:
            continue
        if cell[1]:
            kept_1.append(cell[1][i])
        if cell[2]:
            kept_2.append(cell[2][i])
        if cell[4]:
            kept_4.append(cell[4][i])
        if cell[5]:
            kept_5.append(cell[5][i])
        if cell[6]:
            kept_6.append(cell[6][i])
        if cell[7]:
            kept_7.append(cell[7][i])
        if cell[8]:
            kept_8.append(cell[8][i])

    cell[1] = kept_1
    cell[2] = kept_2
    cell[4] = kept_4
    cell[5] = kept_5
    cell[6] = kept_6
    cell[7] = kept_7
    cell[8] = kept_8


def _port_has_sc_records(P2MP_SC: np.ndarray, node: int, p: int) -> bool:
    """判断一个端口是否还有任意 SC 占用记录。"""
    for sc in range(P2MP_SC.shape[2]):
        try:
            if P2MP_SC[node][p][sc][1]:
                return True
        except Exception:
            continue
    return False


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

    # 端点硬件制式是否需要重构的判定（仅对 src/dst 约束）
    # 返回 True 表示需要重构，False 表示可直接复用原配置
    def needs_path_reconfig(dist: float, hw: int) -> bool:
        if hw == 1:  # 16QAM short
            return dist > 500
        if hw == 2:  # QPSK long
            return False
        return True

    # 遍历每一条虚拟边 (u, v)，从其所有物理候选路径中选出“当前允许条件下”总代价最小的一条。
    # 这里做的是逐边独立选择，而不是全局联合优化，因此本段的职责只包括：
    # 1) 判断这条虚拟边两端节点是否还有足够带宽能力；
    # 2) 遍历该虚拟边对应的所有物理候选；
    # 3) 根据距离、跳数惩罚、是否需要重构来计算候选代价；
    # 4) 记录代价最低的候选，写入 virtual_adj 和 best_phy_map。
    for (u, v), candidates in phy_pool.items():
        # 若虚拟边任一端点的节点能力不足（例如端口剩余容量无法承载 band），
        # 则整条虚拟边直接不可用，不再尝试其任何物理候选。
        if not node_capable[u] or not node_capable[v]:
            continue

        # best_cost / best_candidate 用于在当前虚拟边的候选列表中维护“最优解”。
        # 只要找到代价更低的候选，就覆盖之前的记录。
        best_cost = np.inf
        best_candidate: Optional[Dict[str, Any]] = None

        for cand in candidates:
            # 候选物理路径的距离，后续既用于计算 cost，也用于推断该路径天然适合的调制格式。
            phy_dist = float(cand["dist"])

            # 根据距离得到这条物理路径“天然匹配”的调制方式：
            # - 距离 <= 500：认为可用短距调制（modu=1）
            # - 距离 > 500：认为需要长距调制（modu=2）
            # 这个 native_modu 最终会随 best_candidate 一起记录，供后续映射/资源分配阶段使用。
            native_modu = 1 if phy_dist <= 500 else 2

            # 基础代价由“物理距离 + 固定跳数惩罚”构成。
            # 这里的 HOP_PENALTY 虽然是常数，但保留它的意义在于：
            # - 统一不同边的选路打分模型；
            # - 为后续若扩展为和 hop 相关的代价模型保留接口。
            cost = phy_dist + HOP_PENALTY

            # 只检查“本条虚拟边是否触碰到原业务端点 src/dst”时的兼容性问题。
            # 中间转发节点不要求继承原始调制配置，因此不参与重构判定。
            #
            # reconfig_src:
            #   仅当当前虚拟边的起点就是原始源点 src 时，才检查该物理距离
            #   是否与源端原有调制 orig_src_modu 兼容。
            #
            # reconfig_dst:
            #   仅当当前虚拟边的终点就是原始宿点 dst 时，才检查该物理距离
            #   是否与宿端原有调制 orig_dst_modu 兼容。
            #
            # needs_reconfig:
            #   只要 src 端或 dst 端任一侧不兼容，就认为该候选路径需要重构。
            #   换言之，这个布尔值表达的是“该候选是否还能无缝复用原端点配置”。
            reconfig_src = (u == src) and needs_path_reconfig(phy_dist, orig_src_modu) # 如果这一对节点可能是源节点或者是目的节点的话，继续判断物理路径和恢复之前的调制格式状态
            reconfig_dst = (v == dst) and needs_path_reconfig(phy_dist, orig_dst_modu) # 如果物理路径过长但是原调制格式是16QAM的话，这个时候就需要进行重构
            needs_reconfig = reconfig_src or reconfig_dst # 如果源节点或者是目的节点之中的任意一个点需要重构，都判断为需要重构

            # 两种策略对“需要重构”的候选处理不同：
            #
            # 策略1（force_strategy1=True）：
            #   要求完全复用原有端点配置，不允许任何端点重构。
            #   因此只要 needs_reconfig 为 True，就说明该候选与原配置不兼容，
            #   必须立刻丢弃，直接 continue 看下一个候选。
            #
            # 策略2（force_strategy1=False）：
            #   允许端点重构，所以不会丢弃该候选；
            #   但为了体现“重构是有代价的”，会在基础代价上额外增加 RECONFIG_PENALTY。
            #
            # 也就是说：
            # - continue 的作用不是结束整条虚拟边的搜索；
            # - 它只是跳过“当前这个不满足策略1约束的候选路径”。
            if force_strategy1 and needs_reconfig:
                continue
            if (not force_strategy1) and needs_reconfig:
                # 策略2下，这条路径依然可选，只是因为触发了端点重构，
                # 在排序时会被额外惩罚，从而尽量优先选择那些无需重构的候选。
                cost += RECONFIG_PENALTY

            # 经典“扫描式最优值更新”：
            # 若当前候选的总代价更小，则把它记为当前虚拟边最优候选。
            # 记录的信息包括：
            # - path: 物理路径节点序列
            # - dist: 物理距离
            # - modu: 按距离推断的天然调制
            # - cost: 最终比较时使用的总代价
            if cost < best_cost:
                best_cost = cost
                best_candidate = {"path": cand["path"], "dist": phy_dist, "modu": native_modu, "cost": cost}

        # 若至少存在一个合法候选，则将这条虚拟边写入虚拟拓扑邻接矩阵，
        # 同时把对应的最优物理映射缓存到 best_phy_map，供后续恢复/分配逻辑直接使用。
        # 若 best_candidate 仍为 None，说明：
        # - 要么所有候选都因节点能力不足或策略1限制被过滤掉；
        # - 要么该虚拟边本来就没有可用候选。
        if best_candidate is not None:
            virtual_adj[u][v] = best_cost
            best_phy_map[(u, v)] = best_candidate

    return virtual_adj, best_phy_map


def manage_s1_reservation_by_copy(flow_list: List[Any], metadata_map: Dict[int, Dict[str, Any]],
                                  flow_acc_DP: np.ndarray, node_P2MP_DP: np.ndarray, P2MP_SC_DP: np.ndarray,
                                  P2MP_FS_DP: np.ndarray, flow_path_DP: np.ndarray,
                                  new_node_flow: np.ndarray, new_node_P2MP: np.ndarray, new_P2MP_SC_1: np.ndarray,
                                  new_P2MP_FS: np.ndarray, new_link_FS: np.ndarray,
                                  action: str = "reserve") -> None:
    """
    S1 预占位/回滚（仅端到端首尾端点）：
      - reserve: 只预占该原始流“源节点侧 hub”与“目的节点侧 leaf”的 SC/容量
      - rollback: 只回滚上述首尾端点的 SC/容量

    注意：这里以 DP 的 subflow_id 来匹配 P2MP_SC[*][*][*][1] 的 entry[0]
    """
    if action not in ("reserve", "rollback"):
        raise ValueError(f"Unknown action={action}")

    # S1 只处理原始业务首尾两端的端口资源：
    # 1. 源端对应的 hub 端口；
    # 2. 宿端对应的 leaf 端口。
    # reserve 时从 DP 状态复制 SC 占用并扣减端口剩余容量 [4]，
    # rollback 时删除对应 SC 占用并恢复端口剩余容量 [4]。
    # 这里不修改 node_P2MP 的其他字段（例如 [2]、[5]）。
    # 这里仍保留 metadata_map / P2MP_FS_DP / flow_path_DP / new_P2MP_FS / new_link_FS
    # 等参数，主要是为了与外部统一接口保持一致。

    for flow in flow_list:
        f_id = int(flow[0])
        flow_src = int(flow[1])
        flow_dst = int(flow[2])
        rows = flow_acc_DP[flow_acc_DP[:, 6] == f_id]
        if len(rows) == 0:
            raise ValueError(f"metadata incomplete for flow {f_id}")

        for row in rows:
            sub_id = int(row[0])
            src = int(row[1])
            dst = int(row[2])
            band = float(row[3])
            hub_p = int(row[7])
            leaf_p = int(row[8])
            sc0 = int(row[9])
            sc1 = int(row[10])
            l0 = int(row[13])
            l1 = int(row[14])

            # 当前 subflow 的起点如果就是原始流源点，则这一行描述的是源端 hub 侧资源。
            if src == flow_src and hub_p >= 0 and sc0 >= 0 and sc1 >= 0:
                if action == "reserve":
                    # 只扣减源端 hub 端口的剩余容量 [4]。
                    new_node_P2MP[src][hub_p][4] -= band
                    # 将该 subflow 在 [sc0, sc1] 范围内的 SC 占用记录从 DP 解复制到新状态。
                    for s in range(sc0, sc1 + 1):
                        _copy_sc_records_by_subflow_id(P2MP_SC_DP[src][hub_p][s], new_P2MP_SC_1[src][hub_p][s], sub_id)
                else:
                    # 回滚源端 hub 侧预留：恢复容量并删除该 subflow 的 SC 记录。
                    new_node_P2MP[src][hub_p][4] += band
                    for s in range(sc0, sc1 + 1):
                        _remove_sc_records_by_subflow_id(new_P2MP_SC_1[src][hub_p][s], sub_id)

            # 当前 subflow 的终点如果就是原始流宿点，则这一行描述的是宿端 leaf 侧资源。
            if dst == flow_dst and leaf_p >= 0 and l0 >= 0 and l1 >= 0:
                if action == "reserve":
                    # 只扣减宿端 leaf 端口的剩余容量 [4]。
                    new_node_P2MP[dst][leaf_p][4] -= band
                    # 将该 subflow 在 [l0, l1] 范围内的 SC 占用记录从 DP 解复制到新状态。
                    for s in range(l0, l1 + 1):
                        _copy_sc_records_by_subflow_id(P2MP_SC_DP[dst][leaf_p][s], new_P2MP_SC_1[dst][leaf_p][s], sub_id)
                else:
                    # 回滚宿端 leaf 侧预留：恢复容量并删除该 subflow 的 SC 记录。
                    new_node_P2MP[dst][leaf_p][4] += band
                    for s in range(l0, l1 + 1):
                        _remove_sc_records_by_subflow_id(new_P2MP_SC_1[dst][leaf_p][s], sub_id)


