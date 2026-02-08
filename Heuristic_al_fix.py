#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : ILP.py 
# @File    : Heuristic_al_fix.py
# @Author  : Wumh
# @Time    : 2026/1/4 13:56
# /mnt/data/Heuristic_algorithm.py
from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import topology as tp
from k_shortest_path import k_shortest_path
from modu_format_Al import modu_format_Al
from Path_Link import get_links_from_path

from assign_hop import try_assign_one_hop_s1
from other_fx_fix import (
    extract_old_flow_info,
    build_virtual_topology,
    manage_s1_reservation_by_copy,
    build_fs_meta_np_from_p2mp_sc,
)


def logical_path_is_valid(path_1b: List[int], break_node_0b: int) -> bool:
    """逻辑路径(1-based)必须简单且不能包含 break_node."""
    if not path_1b or len(path_1b) < 2:
        return False
    # break_node 0-based -> compare with 1-based path
    if (break_node_0b + 1) in path_1b:
        return False
    return len(set(path_1b)) == len(path_1b)


def _infer_hw_modu_from_cap(cap: float) -> int:
    """
    你工程里 DP 的 row[4] 实际是 SC_cap(<=500 => 25, >500 => 12.5/12)。
    映射到 hardware_modu:
      1 -> short (16QAM)
      2 -> long  (QPSK)
    """
    return 1 if cap >= 20 else 2


def _build_flow_metadata_map(
    affected_flow: List[Any],
    flow_acc_DP: np.ndarray,
    P2MP_SC_DP: np.ndarray,
) -> Dict[int, Dict[str, Any]]:
    meta: Dict[int, Dict[str, Any]] = {}
    for f in affected_flow:
        fid, src, dst = int(f[0]), int(f[1]), int(f[2])
        meta[fid] = extract_old_flow_info(fid, src, dst, flow_acc_DP, P2MP_SC_DP)
    return meta


def _build_subflow_lookup(new_flow_acc: np.ndarray) -> Dict[Tuple[int, int, int], int]:
    """
    (orig_flow_id, a, b) -> subflow_id(row index)
    new_flow_acc 行结构同 DP：row[0]=subflow_id, row[1]=a, row[2]=b, row[6]=orig_flow_id
    """
    m: Dict[Tuple[int, int, int], int] = {}
    for row in new_flow_acc:
        sub_id = int(row[0])
        a = int(row[1])
        b = int(row[2])
        orig = int(row[6])
        m[(orig, a, b)] = sub_id
    return m


def _ensure_sc_cap_left_for_nodes(
    new_P2MP_SC_1: np.ndarray,
    nodes: List[int],
    sc_cap: float,
) -> None:
    """
    解决“空 SC cap_left=0 导致无法新开”的问题：
    对 nodes 上所有 P2MP/SC，如果 used_list 为空且 cap_left 不存在或为 0，则把 cap_left 初始化为 sc_cap。
    注意：不写 cell[0]，避免破坏 0/1/2/4 列对齐。
    """
    for u in nodes:
        for p in range(new_P2MP_SC_1.shape[1]):
            for s in range(16):
                cell = new_P2MP_SC_1[u, p, s]
                if not isinstance(cell[1], list):
                    cell[1] = []
                if not isinstance(cell[3], list) or len(cell[3]) == 0:
                    cell[3] = [0.0]
                if len(cell[1]) == 0 and float(cell[3][0]) <= 0.0:
                    cell[3] = [float(sc_cap)]


def _apply_hop_plan(
    flow: Any,
    a: int,
    b: int,
    plan: Dict[str, Any],
    subflow_lookup: Dict[Tuple[int, int, int], int],
    new_flow_acc: np.ndarray,
    new_node_flow: np.ndarray,
    new_node_P2MP: np.ndarray,
    new_P2MP_SC_1: np.ndarray,
    new_link_FS: np.ndarray,
    new_flow_path: np.ndarray,
) -> None:
    """
    将 try_assign_one_hop_s1 返回的 plan 写入资源矩阵。
    """
    fid = int(flow[0])
    band = float(flow[3])

    sub_id = subflow_lookup[(fid, a, b)]
    sp = int(plan["sender_p2mp"])
    rp = int(plan["receiver_p2mp"])
    s_sc0, s_sc1 = map(int, plan["sender_sc"])
    r_sc0, r_sc1 = map(int, plan["receiver_sc"])
    s_base = int(plan["sender_base_fs"])
    r_base = int(plan["receiver_base_fs"])
    fs_s_abs, fs_e_abs = map(int, plan["fs_abs"])
    used_links_0b = list(plan["used_links"])
    phy_path_1b = plan.get("phy_path_1based", [])

    # 更新 node_P2MP 状态
    if new_node_P2MP[a][sp][5] == -1:
        new_node_P2MP[a][sp][5] = s_base
    if new_node_P2MP[b][rp][5] == -1:
        new_node_P2MP[b][rp][5] = r_base

    new_node_P2MP[a][sp][4] -= band
    new_node_P2MP[b][rp][4] -= band

    if new_node_P2MP[a][sp][2] == 0:
        new_node_P2MP[a][sp][2] = 1
    if new_node_P2MP[b][rp][2] == 0:
        new_node_P2MP[b][rp][2] = 2

    # 子载波容量（由该 hop 的距离决定）
    phy_dist = float(plan.get("phy_dist", 0.0))
    sc_cap, _ = modu_format_Al(phy_dist, band)

    def allocate_on_sc(node: int, p: int, sc0: int, sc1: int, peer: int) -> None:
        remain = band
        for s in range(sc0, sc1 + 1):
            cell = new_P2MP_SC_1[node, p, s]
            for k in (0, 1, 2, 4):
                if not isinstance(cell[k], list):
                    cell[k] = []
            if not isinstance(cell[3], list) or len(cell[3]) == 0:
                cell[3] = [float(sc_cap)]

            cap_left = float(cell[3][0])
            use = min(cap_left, remain)
            if use <= 0:
                continue

            # 保持列对齐：每次新增 used_list 同步 append [0],[2],[4]
            cell[0].append(float(sc_cap))
            cell[1].append([sub_id, float(use), fid])
            cell[2].append(list(phy_path_1b))
            cell[4].append(peer)

            cell[3] = [cap_left - use]
            remain -= use
            if remain <= 1e-9:
                break

        if remain > 1e-6:
            raise RuntimeError(f"SC allocation insufficient for flow={fid} hop {a}->{b}, remain={remain}")

    allocate_on_sc(a, sp, s_sc0, s_sc1, b)
    allocate_on_sc(b, rp, r_sc0, r_sc1, a)

    # 更新 node_flow
    subflow_info = [sub_id, a, b, band, sc_cap, (s_sc1 - s_sc0 + 1), fid]
    new_node_flow[a][sp][0].append(subflow_info)
    new_node_flow[b][rp][0].append(subflow_info)

    # 更新 flow_acc（记录 hub/leaf 与 FS/SC）
    new_flow_acc[sub_id][7] = sp
    new_flow_acc[sub_id][8] = rp
    new_flow_acc[sub_id][9] = s_sc0
    new_flow_acc[sub_id][10] = s_sc1
    new_flow_acc[sub_id][11] = fs_s_abs
    new_flow_acc[sub_id][12] = fs_e_abs

    # 更新 link_FS
    for l0 in used_links_0b:
        new_link_FS[l0][fs_s_abs:fs_e_abs + 1] += 1

    # 更新 flow_path（保持与 DP 一致：link id 存 1-based）
    new_flow_path[sub_id][0] = [sub_id]
    new_flow_path[sub_id][1] = [int(x) + 1 for x in used_links_0b]
    new_flow_path[sub_id][2] = list(phy_path_1b)
    new_flow_path[sub_id][3] = [fid]


def Heuristic_algorithm(
    affected_flow: List[Any],
    new_flow_acc: np.ndarray,
    new_node_flow: np.ndarray,
    new_node_P2MP: np.ndarray,
    break_node: int,
    Tbox_num: int,
    Tbox_P2MP: int,
    new_P2MP_SC_1: np.ndarray,
    new_link_FS: np.ndarray,
    node_flow_DP: np.ndarray,
    node_P2MP_DP: np.ndarray,
    flow_acc_DP: np.ndarray,
    link_FS_DP: np.ndarray,
    P2MP_SC_DP: np.ndarray,
    flow_path_DP: np.ndarray,
):
    topo_num, _, topo_dis, link_num, link_index = tp.topology(1)

    # 参数
    K_PHY_CANDIDATES = 3
    K_LOGICAL_PATHS = 5
    RECONFIG_PENALTY = 5000
    HOP_PENALTY = 200

    # 预计算物理候选池
    phy_pool: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for u in range(topo_num):
        for v in range(topo_num):
            if u == v or u == break_node or v == break_node:
                continue
            try:
                cand_list = []
                k_paths = k_shortest_path(topo_dis, u + 1, v + 1, K_PHY_CANDIDATES)
                for p_nodes, dist in k_paths:
                    p_nodes = [x - 1 for x in p_nodes]
                    cand_list.append({"path": p_nodes, "dist": float(dist)})
                if cand_list:
                    phy_pool[(u, v)] = cand_list
            except Exception:
                continue

    flow_metadata_map = _build_flow_metadata_map(affected_flow, flow_acc_DP, P2MP_SC_DP)
    subflow_lookup = _build_subflow_lookup(new_flow_acc)

    # 初始 new_flow_path
    new_flow_path = copy.deepcopy(flow_path_DP)

    # 为每条流构建 S1/S2 的虚拟拓扑
    tier1_flows: List[Any] = []
    tier2_flows: List[Any] = []
    flow_vmap_s1: Dict[int, Tuple[np.ndarray, Dict[Tuple[int, int], Dict[str, Any]]]] = {}
    flow_vmap_s2: Dict[int, Tuple[np.ndarray, Dict[Tuple[int, int], Dict[str, Any]]]] = {}

    for f in affected_flow:
        fid, src, dst, band = int(f[0]), int(f[1]), int(f[2]), float(f[3])
        meta = flow_metadata_map[fid]
        src_modu = _infer_hw_modu_from_cap(float(meta.get("hub_cap", 0.0)))
        dst_modu = _infer_hw_modu_from_cap(float(meta.get("leaf_cap", 0.0)))

        v_adj_s1, v_phy_s1 = build_virtual_topology(
            src, dst, topo_num, phy_pool, break_node, new_node_P2MP, band,
            True, src_modu, dst_modu, RECONFIG_PENALTY, HOP_PENALTY
        )
        flow_vmap_s1[fid] = (v_adj_s1, v_phy_s1)

        # 判断是否存在路径
        ok_s1 = False
        try:
            paths = k_shortest_path(v_adj_s1, src + 1, dst + 1, 1)
            for p_nodes, _ in paths:
                if logical_path_is_valid(p_nodes, break_node):
                    ok_s1 = True
                    break
        except Exception:
            ok_s1 = False

        if ok_s1:
            tier1_flows.append(f)
        else:
            tier2_flows.append(f)

        # S2 always build
        v_adj_s2, v_phy_s2 = build_virtual_topology(
            src, dst, topo_num, phy_pool, break_node, new_node_P2MP, band,
            False, src_modu, dst_modu, RECONFIG_PENALTY, HOP_PENALTY
        )
        flow_vmap_s2[fid] = (v_adj_s2, v_phy_s2)

    # --- Tier1 reservation (copy from DP) ---
    manage_s1_reservation_by_copy(
        tier1_flows, flow_metadata_map,
        flow_acc_DP, node_P2MP_DP, P2MP_SC_DP,
        new_node_P2MP, new_P2MP_SC_1,
        action="reserve"
    )

    # rebuild meta
    fs_total = int(new_link_FS.shape[1])
    new_P2MP_FS_1, new_link_FS_meta = build_fs_meta_np_from_p2mp_sc(new_P2MP_SC_1, new_node_P2MP, fs_total, link_index)

    tier1_restored, tier1_failed = [], []

    # --- Tier1 restore ---
    for f in tier1_flows:
        fid = int(f[0])
        src = int(f[1])
        dst = int(f[2])
        band = float(f[3])

        # rollback this flow reservation before trying
        manage_s1_reservation_by_copy(
            [f], flow_metadata_map,
            flow_acc_DP, node_P2MP_DP, P2MP_SC_DP,
            new_node_P2MP, new_P2MP_SC_1,
            action="rollback"
        )
        new_P2MP_FS_1, new_link_FS_meta = build_fs_meta_np_from_p2mp_sc(new_P2MP_SC_1, new_node_P2MP, fs_total, link_index)

        v_adj, v_phy = flow_vmap_s1[fid]

        restored = False

        try:
            logical_paths = k_shortest_path(v_adj, src + 1, dst + 1, K_LOGICAL_PATHS)
        except Exception:
            logical_paths = []

        # snapshot for each flow
        snap = (
            copy.deepcopy(new_flow_acc),
            copy.deepcopy(new_node_flow),
            copy.deepcopy(new_node_P2MP),
            copy.deepcopy(new_P2MP_SC_1),
            copy.deepcopy(new_link_FS),
            copy.deepcopy(new_flow_path),
        )

        for lp_nodes_1b, _ in logical_paths:
            if not logical_path_is_valid(lp_nodes_1b, break_node):
                continue
            lp0 = [x - 1 for x in lp_nodes_1b]

            # restore snapshot
            (work_flow_acc, work_node_flow, work_node_P2MP, work_P2MP_SC_1, work_link_FS, work_flow_path) = copy.deepcopy(snap)
            work_P2MP_FS_1, work_link_FS_meta = build_fs_meta_np_from_p2mp_sc(work_P2MP_SC_1, work_node_P2MP, fs_total, link_index)

            ok_path = True
            for a, b in zip(lp0[:-1], lp0[1:]):
                if (a, b) not in v_phy:
                    ok_path = False
                    break

                phy_cand = v_phy[(a, b)]
                phy_path_1b = list(phy_cand["path"])
                phy_dist = float(phy_cand["dist"])

                sc_cap, _ = modu_format_Al(phy_dist, band)
                _ensure_sc_cap_left_for_nodes(work_P2MP_SC_1, [a, b], sc_cap)

                ok_hop, plan = try_assign_one_hop_s1(
                    flow=f,
                    a=a,
                    b=b,
                    phy_candidate=phy_cand,
                    phy_path_1based=phy_path_1b,
                    flow_metadata_map=flow_metadata_map,
                    link_index=link_index,
                    new_link_FS=work_link_FS,
                    new_node_P2MP=work_node_P2MP,
                    new_P2MP_SC_1=work_P2MP_SC_1,
                    new_node_flow=work_node_flow,
                    new_flow_acc=work_flow_acc,
                    new_P2MP_FS_1=work_P2MP_FS_1,
                    new_link_FS_meta=work_link_FS_meta,
                    strict_s1=True,
                )

                if not ok_hop or plan is None:
                    ok_path = False
                    break

                plan["phy_path_1based"] = phy_path_1b
                plan["phy_dist"] = phy_dist

                _apply_hop_plan(
                    flow=f, a=a, b=b, plan=plan,
                    subflow_lookup=subflow_lookup,
                    new_flow_acc=work_flow_acc,
                    new_node_flow=work_node_flow,
                    new_node_P2MP=work_node_P2MP,
                    new_P2MP_SC_1=work_P2MP_SC_1,
                    new_link_FS=work_link_FS,
                    new_flow_path=work_flow_path,
                )

                work_P2MP_FS_1, work_link_FS_meta = build_fs_meta_np_from_p2mp_sc(work_P2MP_SC_1, work_node_P2MP, fs_total, link_index)

            if ok_path:
                new_flow_acc, new_node_flow, new_node_P2MP, new_P2MP_SC_1, new_link_FS, new_flow_path = \
                    work_flow_acc, work_node_flow, work_node_P2MP, work_P2MP_SC_1, work_link_FS, work_flow_path
                restored = True
                break

        if restored:
            tier1_restored.append(f)
        else:
            tier1_failed.append(f)
            tier2_flows.append(f)  # downgrade to tier2

    # --- Tier2 restore ---
    tier2_restored, tier2_failed = [], []

    for f in tier2_flows:
        fid = int(f[0])
        src = int(f[1])
        dst = int(f[2])
        band = float(f[3])

        v_adj, v_phy = flow_vmap_s2[fid]

        try:
            logical_paths = k_shortest_path(v_adj, src + 1, dst + 1, K_LOGICAL_PATHS)
        except Exception:
            logical_paths = []

        snap = (
            copy.deepcopy(new_flow_acc),
            copy.deepcopy(new_node_flow),
            copy.deepcopy(new_node_P2MP),
            copy.deepcopy(new_P2MP_SC_1),
            copy.deepcopy(new_link_FS),
            copy.deepcopy(new_flow_path),
        )

        restored = False

        for lp_nodes_1b, _ in logical_paths:
            if not logical_path_is_valid(lp_nodes_1b, break_node):
                continue
            lp0 = [x - 1 for x in lp_nodes_1b]

            (work_flow_acc, work_node_flow, work_node_P2MP, work_P2MP_SC_1, work_link_FS, work_flow_path) = copy.deepcopy(snap)
            work_P2MP_FS_1, work_link_FS_meta = build_fs_meta_np_from_p2mp_sc(work_P2MP_SC_1, work_node_P2MP, fs_total,
                                                                              link_index)

            ok_path = True
            for a, b in zip(lp0[:-1], lp0[1:]):
                if (a, b) not in v_phy:
                    ok_path = False
                    break

                phy_cand = v_phy[(a, b)]
                phy_path_1b = list(phy_cand["path"])
                phy_dist = float(phy_cand["dist"])

                sc_cap, _ = modu_format_Al(phy_dist, band)
                _ensure_sc_cap_left_for_nodes(work_P2MP_SC_1, [a, b], sc_cap)

                ok_hop, plan = try_assign_one_hop_s1(
                    flow=f,
                    a=a,
                    b=b,
                    phy_candidate=phy_cand,
                    phy_path_1based=phy_path_1b,
                    flow_metadata_map=flow_metadata_map,
                    link_index=link_index,
                    new_link_FS=work_link_FS,
                    new_node_P2MP=work_node_P2MP,
                    new_P2MP_SC_1=work_P2MP_SC_1,
                    new_node_flow=work_node_flow,
                    new_flow_acc=work_flow_acc,
                    new_P2MP_FS_1=work_P2MP_FS_1,
                    new_link_FS_meta=work_link_FS_meta,
                    strict_s1=False,
                )

                if not ok_hop or plan is None:
                    ok_path = False
                    break

                plan["phy_path_1based"] = phy_path_1b
                plan["phy_dist"] = phy_dist

                _apply_hop_plan(
                    flow=f, a=a, b=b, plan=plan,
                    subflow_lookup=subflow_lookup,
                    new_flow_acc=work_flow_acc,
                    new_node_flow=work_node_flow,
                    new_node_P2MP=work_node_P2MP,
                    new_P2MP_SC_1=work_P2MP_SC_1,
                    new_link_FS=work_link_FS,
                    new_flow_path=work_flow_path,
                )

                work_P2MP_FS_1, work_link_FS_meta = build_fs_meta_np_from_p2mp_sc(work_P2MP_SC_1, work_node_P2MP, fs_total, link_index)

            if ok_path:
                new_flow_acc, new_node_flow, new_node_P2MP, new_P2MP_SC_1, new_link_FS, new_flow_path = \
                    work_flow_acc, work_node_flow, work_node_P2MP, work_P2MP_SC_1, work_link_FS, work_flow_path
                restored = True
                break

        if restored:
            tier2_restored.append(f)
        else:
            tier2_failed.append(f)

    return new_node_flow, new_node_P2MP, new_flow_acc, new_link_FS, new_P2MP_SC_1, new_flow_path
