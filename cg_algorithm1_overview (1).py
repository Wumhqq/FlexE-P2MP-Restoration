#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按照论文 Algorithm 1（Overall Procedure of hAG-S1F-CP-CG+）实现的
总控程序。

这份文件把论文里的“总览算法”拆成了几个可单独复用的步骤：

1. 构建物理候选路径池 P_{u,v}
2. 为每条流构建 AG，并计算 K_lgc 条候选逻辑路径 L_r
3. 用 AG-S1F 思想做 warm start，得到初始恢复集合 Z^0
4. 初始化活动列池 Omega_r^0 与候选库 Lambda_r^(0)
5. 在当前 path rank j 下，用 Algorithm 3 生成新的候选列库
6. 循环求 LP-RMP、提取对偶、挑选 reduced cost 最小且为负的列加入活动池
7. 当边际改善 Delta^(j) 小于阈值 epsilon 时停止 path-rank 扩张
8. 最终求解整数 RMP，输出所选列

-----------------------------------------------------------------------
设计原则
-----------------------------------------------------------------------
- 尽量复用你已经上传的代码风格和数据结构；
- 把“列池生成”（Algorithm 3）和“总览调度”（Algorithm 1）分成两个文件；
- 优先使用现有模块，如果某些外部模块在当前环境不存在，则允许通过回调或预计算结果注入；
- 对 warm start 保留一个“完全可跑的贪心版”和一个“可接你现有 AG_S1F 的可选接口”。

-----------------------------------------------------------------------
最重要的输入
-----------------------------------------------------------------------
你至少需要准备：

1. affected_flows
   受影响流列表，每个元素形如 [flow_id, src, dst, bw, ...]

2. state_after_failure
   失败后的网络状态 C0：
   {
       "flow_acc": ...,
       "node_flow": ...,
       "node_P2MP": ...,
       "P2MP_SC": ...,
       "link_FS": ...,
       "P2MP_FS": ...,
       "flow_path": ...,
   }

3. rmp_sets
   用来构建 master problem 的集合与参数，格式需兼容 RMPData：
   {
       "vg_nodes": ...,
       "eo_links": ...,
       "fibers": ...,
       "H_u": ...,
       "K_up": ...,
       "W_up": ...,
       "b": ...,
       "U": ...,
       "beta": ...,
       "big_m": ...,
   }

4. candidate_logical_paths_by_flow / mapping_tables_by_flow
   如果你已经预计算好了每条流的候选逻辑路径和 AG 映射表，
   直接传进来即可；
   否则，可以让本文件尝试通过你的 k_shortest_path / build_virtual_topology
   自动生成。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from rmp_master_problem import (
    Column,
    RMPData,
    compute_reduced_cost,
    get_selected_columns,
    solve_rmp_ip,
    solve_rmp_lp,
)

from cg_algorithm3_generate_pool import (
    GeneratedColumn,
    _column_signature,
    _extract_flow_metadata_from_base,
    generate_candidate_column_pool_on_path,
    try_candidate_paths_for_one_flow,
)


# =====================================================================
# 一些可选外部依赖：如果你的工程环境里有，就自动复用；没有也不强依赖。
# =====================================================================

try:
    from k_shortest_path import k_shortest_path as _k_shortest_path
except Exception:
    _k_shortest_path = None

try:
    from other_fx_fix import build_virtual_topology as _build_virtual_topology
except Exception:
    _build_virtual_topology = None

try:
    import AG_S1F as _ag_s1f
    # 你上传的 AG_S1F.py 里 sc_effective_cap 使用了 math 和 TS_UNIT，
    # 但文件顶部没有显式补全；这里在导入后补丁一下，避免运行时 NameError。
    if not hasattr(_ag_s1f, "math"):
        _ag_s1f.math = math
    if not hasattr(_ag_s1f, "TS_UNIT"):
        _ag_s1f.TS_UNIT = 5
except Exception:
    _ag_s1f = None


# =====================================================================
# 数据结构
# =====================================================================

@dataclass
class WarmStartResult:
    restored_columns: Dict[Any, GeneratedColumn]
    state_after: Dict[str, Any]
    tier1_flow_ids: List[Any]
    tier2_flow_ids: List[Any]


@dataclass
class CPGRunResult:
    warm_start: WarmStartResult
    omega: Dict[Any, List[Column]]
    lambda_repo: Dict[Any, List[GeneratedColumn]]
    lp_objectives: Dict[int, float]
    improvements: Dict[int, float]
    final_rmp: Any
    final_selected_columns: Dict[Any, List[Tuple[str, float]]]


# =====================================================================
# 基础函数
# =====================================================================


def _deepcopy_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "flow_acc": copy.deepcopy(state["flow_acc"]),
        "node_flow": copy.deepcopy(state["node_flow"]),
        "node_P2MP": copy.deepcopy(state["node_P2MP"]),
        "P2MP_SC": copy.deepcopy(state["P2MP_SC"]),
        "link_FS": copy.deepcopy(state["link_FS"]),
        "P2MP_FS": copy.deepcopy(state["P2MP_FS"]),
        "flow_path": copy.deepcopy(state["flow_path"]),
    }


def _flow_id(flow: Sequence[Any]) -> Any:
    return int(flow[0])


def _build_rmp_data(
    affected_flows: Sequence[Sequence[Any]],
    omega: Mapping[Any, Sequence[Column]],
    rmp_sets: Mapping[str, Any],
) -> RMPData:
    """把当前活动列池 Omega 转成 rmp_master_problem.py 需要的 RMPData。"""
    flow_ids = [_flow_id(f) for f in affected_flows]
    return RMPData(
        flows=flow_ids,
        vg_nodes=rmp_sets["vg_nodes"],
        eo_links=rmp_sets["eo_links"],
        fibers=rmp_sets["fibers"],
        H_u=rmp_sets["H_u"],
        K_up=rmp_sets["K_up"],
        W_up=rmp_sets["W_up"],
        b=rmp_sets["b"],
        U=rmp_sets["U"],
        beta=float(rmp_sets["beta"]),
        big_m=float(rmp_sets["big_m"]),
        columns=omega,
    )


def _merge_generated_columns(
    target: MutableMapping[Any, List[GeneratedColumn]],
    flow_id: Any,
    new_cols: Sequence[GeneratedColumn],
) -> None:
    """按列签名去重后，把新列并入 Lambda 仓库。"""
    if flow_id not in target:
        target[flow_id] = []

    seen = {_column_signature(gc.column) for gc in target[flow_id]}
    for gc in new_cols:
        sig = _column_signature(gc.column)
        if sig in seen:
            continue
        target[flow_id].append(gc)
        seen.add(sig)


def _column_ids_in_omega(omega: Mapping[Any, Sequence[Column]], flow_id: Any) -> set[str]:
    return {str(col.col_id) for col in omega.get(flow_id, [])}


def _sum_subflow_sc_usage(P2MP_SC: np.ndarray, node: int, p: int, sc: int, sub_id: int, orig_id: int) -> float:
    used = 0.0
    try:
        entries = P2MP_SC[node][p][sc][1]
    except Exception:
        entries = []
    for ent in entries or []:
        try:
            if int(ent[0]) == int(sub_id) or (len(ent) > 2 and int(ent[2]) == int(orig_id)):
                used += float(ent[1])
        except Exception:
            continue
    return float(used)


def _build_column_from_state_for_flow(
    flow: Sequence[Any],
    state: Mapping[str, Any],
    *,
    base_state: Optional[Mapping[str, Any]] = None,
    col_id_prefix: str = "warm",
) -> Optional[Column]:
    """
    把现有状态里某个原始流的恢复结果抽取成一条 Column。

    这个函数不重写分配逻辑，只负责把 AG_S1F / Initial_net 已经落到状态表里的结果，
    重新整理成 master problem 需要的列系数。
    """
    fid = int(flow[0])
    flow_acc = np.asarray(state["flow_acc"], dtype=object)
    node_P2MP = np.asarray(state["node_P2MP"], dtype=object)
    P2MP_SC = np.asarray(state["P2MP_SC"], dtype=object)
    flow_path = np.asarray(state["flow_path"], dtype=object)

    rows = []
    for row in flow_acc:
        try:
            if int(row[6]) == fid and int(row[7]) >= 0 and int(row[8]) >= 0:
                rows.append(row)
        except Exception:
            continue
    if not rows:
        return None

    col = Column(col_id=f"{col_id_prefix}_{fid}", f_rf_clr=0)

    old_meta = None
    if base_state is not None:
        try:
            old_meta = _extract_flow_metadata_from_base(
                flow,
                np.asarray(base_state["flow_acc"], dtype=object),
                np.asarray(base_state["P2MP_SC"], dtype=object),
            )
        except Exception:
            old_meta = None

    strict_s1 = True

    for row in rows:
        sub_id = int(row[0])
        a = int(row[1])
        b = int(row[2])
        sc_cap = float(row[4])
        hub_p = int(row[7])
        leaf_p = int(row[8])
        hub_sc_s = int(row[9])
        hub_sc_e = int(row[10])
        fs_s_glo = int(row[11])
        fs_e_glo = int(row[12])
        leaf_sc_s = int(row[13])
        leaf_sc_e = int(row[14])

        if old_meta is not None:
            if a == int(flow[1]) and old_meta.get("hub_idx", -1) != -1:
                if hub_p != int(old_meta["hub_idx"]):
                    strict_s1 = False
                hub_range = old_meta.get("hub_sc_range")
                if hub_range is not None and (hub_sc_s, hub_sc_e) != (int(hub_range[0]), int(hub_range[1])):
                    strict_s1 = False
            if b == int(flow[2]) and old_meta.get("leaf_idx", -1) != -1:
                if leaf_p != int(old_meta["leaf_idx"]):
                    strict_s1 = False
                leaf_range = old_meta.get("leaf_sc_range")
                if leaf_range is not None and (leaf_sc_s, leaf_sc_e) != (int(leaf_range[0]), int(leaf_range[1])):
                    strict_s1 = False

        col.g_tx[(a, hub_p)] = 1.0
        col.g_rx[(b, leaf_p)] = 1.0

        phys_nodes_1b = []
        try:
            phys_nodes_1b = list(flow_path[sub_id][2])
        except Exception:
            phys_nodes_1b = []
        phys_path0 = [int(x) - 1 for x in phys_nodes_1b]
        used_pairs = list(zip(phys_path0[:-1], phys_path0[1:]))

        hub_base = int(node_P2MP[a][hub_p][5]) if int(node_P2MP[a][hub_p][5]) >= 0 else fs_s_glo
        leaf_base = int(node_P2MP[b][leaf_p][5]) if int(node_P2MP[b][leaf_p][5]) >= 0 else fs_s_glo

        for sc in range(hub_sc_s, hub_sc_e + 1):
            col.e_tx[(a, hub_p, sc)] = 1.0
            col.xi[(a, hub_p, sc)] = 1.0
            col.a[(a, hub_p, sc)] = max(float(col.a.get((a, hub_p, sc), 0.0)), sc_cap)
            used_amt = _sum_subflow_sc_usage(P2MP_SC, a, hub_p, sc, sub_id, fid)
            if used_amt > 1e-9:
                col.n_tx[(a, hub_p, sc)] = float(col.n_tx.get((a, hub_p, sc), 0.0)) + float(used_amt) / 5.0
            for i, j in used_pairs:
                col.h[(a, hub_p, sc, i, j)] = 1.0

        for sc in range(leaf_sc_s, leaf_sc_e + 1):
            col.e_rx[(b, leaf_p, sc)] = 1.0
            col.xi[(b, leaf_p, sc)] = 1.0
            col.a[(b, leaf_p, sc)] = max(float(col.a.get((b, leaf_p, sc), 0.0)), sc_cap)
            used_amt = _sum_subflow_sc_usage(P2MP_SC, b, leaf_p, sc, sub_id, fid)
            if used_amt > 1e-9:
                col.n_rx[(b, leaf_p, sc)] = float(col.n_rx.get((b, leaf_p, sc), 0.0)) + float(used_amt) / 5.0
            for i, j in used_pairs:
                col.h[(b, leaf_p, sc, i, j)] = 1.0

        for fs_abs in range(fs_s_glo, fs_e_glo + 1):
            w_hub = int(fs_abs) - int(hub_base)
            w_leaf = int(fs_abs) - int(leaf_base)
            col.sigma[(a, hub_p, w_hub)] = 1.0
            col.sigma[(b, leaf_p, w_leaf)] = 1.0
            for i, j in used_pairs:
                col.tau[(a, hub_p, w_hub, i, j)] = 1.0
                col.tau[(b, leaf_p, w_leaf, i, j)] = 1.0
                col.varsigma_tx[(i, j, fs_abs, a, hub_p)] = 1.0
                col.varsigma_rx[(i, j, fs_abs, b, leaf_p)] = 1.0

    col.f_rf_clr = 0 if strict_s1 else 1
    col.flow_id = fid
    return col


def _warm_start_via_existing_heuristic(
    *,
    affected_flows: Sequence[Sequence[Any]],
    break_node: int,
    candidate_logical_paths_by_flow: Mapping[Any, Sequence[Sequence[int]]],
    path_weights_by_flow: Mapping[Any, Sequence[float]],
    state_after_failure: Mapping[str, Any],
    base_state: Mapping[str, Any],
    reconfig_penalty: float,
) -> WarmStartResult:
    """优先直接复用 AG_S1F.py 里的 Heuristic_algorithm 作为 warm start。"""
    if _ag_s1f is None:
        raise RuntimeError("AG_S1F 不可用")

    tier1_ids = []
    tier2_ids = []
    for flow in affected_flows:
        fid = _flow_id(flow)
        weights = list(path_weights_by_flow.get(fid, []))
        if weights and min(weights) <= float(reconfig_penalty):
            tier1_ids.append(fid)
        else:
            tier2_ids.append(fid)

    node_P2MP_arr = np.asarray(state_after_failure["node_P2MP"], dtype=object)
    p2mp_total = int(node_P2MP_arr.shape[1])

    out = _ag_s1f.Heuristic_algorithm(
        affected_flow=[list(f) for f in affected_flows],
        new_flow_acc=copy.deepcopy(np.asarray(state_after_failure["flow_acc"], dtype=object)),
        new_node_flow=copy.deepcopy(np.asarray(state_after_failure["node_flow"], dtype=object)),
        new_node_P2MP=copy.deepcopy(node_P2MP_arr),
        break_node=int(break_node),
        Tbox_num=1,
        Tbox_P2MP=p2mp_total,
        new_P2MP_SC=copy.deepcopy(np.asarray(state_after_failure["P2MP_SC"], dtype=object)),
        new_link_FS=copy.deepcopy(np.asarray(state_after_failure["link_FS"], dtype=object)),
        new_P2MP_FS=copy.deepcopy(np.asarray(state_after_failure["P2MP_FS"], dtype=object)),
        node_flow_base=np.asarray(base_state["node_flow"], dtype=object),
        node_P2MP_base=np.asarray(base_state["node_P2MP"], dtype=object),
        flow_acc_base=np.asarray(base_state["flow_acc"], dtype=object),
        link_FS_base=np.asarray(base_state["link_FS"], dtype=object),
        P2MP_SC_base=np.asarray(base_state["P2MP_SC"], dtype=object),
        P2MP_FS_base=np.asarray(base_state["P2MP_FS"], dtype=object),
        flow_path_base=np.asarray(base_state["flow_path"], dtype=object),
        new_flow_path=copy.deepcopy(np.asarray(state_after_failure["flow_path"], dtype=object)),
    )

    new_state = {
        "node_flow": out[0],
        "node_P2MP": out[1],
        "flow_acc": out[2],
        "link_FS": out[3],
        "P2MP_SC": out[4],
        "P2MP_FS": out[5],
        "flow_path": out[6],
    }
    failed_orig = out[7]
    tier1_restored_ids = out[8]
    tier2_restored_ids = out[9]

    restored_columns: Dict[Any, GeneratedColumn] = {}
    for flow in affected_flows:
        fid = _flow_id(flow)
        col = _build_column_from_state_for_flow(flow, new_state, base_state=base_state, col_id_prefix="warm")
        if col is None:
            continue
        logical_paths = list(candidate_logical_paths_by_flow.get(fid, []))
        chosen_path = list(logical_paths[0]) if logical_paths else []
        restored_columns[fid] = GeneratedColumn(column=col, hop_schemes=[], logical_path=chosen_path, state_after=None)

    return WarmStartResult(
        restored_columns=restored_columns,
        state_after=new_state,
        tier1_flow_ids=tier1_restored_ids,
        tier2_flow_ids=tier2_restored_ids,
    )


# =====================================================================
# Step 1：构建物理路径池 P_{u,v}
# =====================================================================


def build_physical_path_pool(
    *,
    topo_num: int,
    topo_dis: np.ndarray,
    break_node: int,
    K_phy: int,
    ksp_func: Optional[Callable[..., Any]] = None,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """
    对每个 (u,v) 预先计算最多 K_phy 条物理候选 lightpath。

    返回格式：
    {
        (u,v): [
            {"path": [0-based nodes], "dist": float},
            ...
        ]
    }
    """
    if ksp_func is None:
        ksp_func = _k_shortest_path
    if ksp_func is None:
        raise ImportError("需要 k_shortest_path 函数，或通过参数显式传入 ksp_func")

    phy_pool: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}

    for u in range(topo_num):
        for v in range(topo_num):
            if u == v or u == break_node or v == break_node:
                continue
            try:
                cand_list = []
                k_paths = ksp_func(topo_dis, u + 1, v + 1, K_phy)
                for p_nodes, dist in k_paths:
                    p_nodes0 = [int(x) - 1 for x in p_nodes]
                    cand_list.append({"path": p_nodes0, "dist": float(dist)})
                if cand_list:
                    phy_pool[(u, v)] = cand_list
            except Exception:
                continue
    return phy_pool


# =====================================================================
# Step 2：构建每条流的 AG 并生成 K_lgc 条逻辑候选路径
# =====================================================================


def prepare_candidate_logical_paths(
    *,
    affected_flows: Sequence[Sequence[Any]],
    break_node: int,
    state_after_failure: Mapping[str, Any],
    topo_num: int,
    phy_pool: Mapping[Tuple[int, int], List[Dict[str, Any]]],
    K_lgc: int,
    reconfig_penalty: float,
    hop_penalty: float,
    base_state: Optional[Mapping[str, Any]] = None,
    build_ag_func: Optional[Callable[..., Any]] = None,
    ksp_func: Optional[Callable[..., Any]] = None,
) -> Tuple[Dict[Any, List[List[int]]], Dict[Any, Dict[Tuple[int, int], Dict[str, Any]]], Dict[Any, List[float]]]:
    """
    为每条流生成：
    - candidate_logical_paths_by_flow[fid] = [[...], [...], ...]
    - mapping_tables_by_flow[fid] = M_{r,u,v}
    - path_weights_by_flow[fid] = [w1, w2, ...]

    说明：
    - 如果你的工程里 already 有 build_virtual_topology，就直接用；
    - 否则请把 build_ag_func 显式传进来。
    """
    if build_ag_func is None:
        build_ag_func = _build_virtual_topology
    if ksp_func is None:
        ksp_func = _k_shortest_path
    if build_ag_func is None or ksp_func is None:
        raise ImportError("需要 build_virtual_topology 和 k_shortest_path，或通过参数显式传入")

    candidate_paths: Dict[Any, List[List[int]]] = {}
    mapping_tables: Dict[Any, Dict[Tuple[int, int], Dict[str, Any]]] = {}
    path_weights: Dict[Any, List[float]] = {}

    for flow in affected_flows:
        fid = _flow_id(flow)
        src = int(flow[1])
        dst = int(flow[2])
        band = float(flow[3])

        # 为了构建 AG 的重构代价，需要知道源/宿原始调制格式。
        # 这里沿用 AG_S1F.py 的思路：从 base_state 里提取旧端口信息，然后粗略推断硬件调制级别。
        src_modu = 1
        dst_modu = 1
        if base_state is not None:
            meta = _extract_flow_metadata_from_base(
                flow,
                np.asarray(base_state["flow_acc"], dtype=object),
                np.asarray(base_state["P2MP_SC"], dtype=object),
            )
            src_modu = 1 if float(meta.get("hub_cap", 0.0)) >= 20 else 2
            dst_modu = 1 if float(meta.get("leaf_cap", 0.0)) >= 20 else 2

        # 这里按照 Algorithm 2/Algorithm 1 的逻辑，构建一张 flow-specific AG。
        # 为了让候选路径集合覆盖 Strategy 1/2，两类情况都允许，因此 strict=False。
        v_adj, v_phy = build_ag_func(
            src,
            dst,
            topo_num,
            phy_pool,
            break_node,
            state_after_failure["node_P2MP"],
            band,
            False,
            src_modu,
            dst_modu,
            reconfig_penalty,
            hop_penalty,
        )

        mapping_tables[fid] = v_phy

        paths_this_flow: List[List[int]] = []
        weights_this_flow: List[float] = []

        try:
            k_paths = ksp_func(v_adj, src + 1, dst + 1, K_lgc)
            for p_nodes, w in k_paths:
                paths_this_flow.append([int(x) for x in p_nodes])  # 保持 1-based
                weights_this_flow.append(float(w))
        except Exception:
            pass

        candidate_paths[fid] = paths_this_flow
        path_weights[fid] = weights_this_flow

    return candidate_paths, mapping_tables, path_weights


# =====================================================================
# Step 3：Warm Start（对应 Algorithm 4，供 Algorithm 1 第 7 步调用）
# =====================================================================


def warm_start_ag_s1f(
    *,
    affected_flows: Sequence[Sequence[Any]],
    break_node: int,
    candidate_logical_paths_by_flow: Mapping[Any, Sequence[Sequence[int]]],
    mapping_tables_by_flow: Mapping[Any, Mapping[Tuple[int, int], Mapping[str, Any]]],
    path_weights_by_flow: Mapping[Any, Sequence[float]],
    state_after_failure: Mapping[str, Any],
    link_index: np.ndarray,
    reconfig_penalty: float,
    base_state: Optional[Mapping[str, Any]] = None,
    reserve_strategy1_resources_func: Optional[Callable[..., Any]] = None,
    rollback_one_reservation_func: Optional[Callable[..., Any]] = None,
) -> WarmStartResult:
    """
    Strategy-1-first 的 warm start。

    当前实现强制直接调用 AG_S1F.Heuristic_algorithm 作为 warm start，
    不再使用本文件中的 fallback 贪心恢复逻辑。
    """
    if _ag_s1f is None:
        raise RuntimeError("AG_S1F 不可用，无法执行 warm start")
    if base_state is None:
        raise ValueError("warm_start_ag_s1f 需要提供 base_state 才能调用 AG_S1F")

    return _warm_start_via_existing_heuristic(
        affected_flows=affected_flows,
        break_node=int(break_node),
        candidate_logical_paths_by_flow=candidate_logical_paths_by_flow,
        path_weights_by_flow=path_weights_by_flow,
        state_after_failure=state_after_failure,
        base_state=base_state,
        reconfig_penalty=float(reconfig_penalty),
    )


# =====================================================================
# Step 4-8：Algorithm 1 总控
# =====================================================================


def run_hag_s1f_cp_cg_plus(
    *,
    affected_flows: Sequence[Sequence[Any]],
    break_node: int,
    state_after_failure: Mapping[str, Any],
    rmp_sets: Mapping[str, Any],
    link_index: np.ndarray,
    candidate_logical_paths_by_flow: Mapping[Any, Sequence[Sequence[int]]],
    mapping_tables_by_flow: Mapping[Any, Mapping[Tuple[int, int], Mapping[str, Any]]],
    path_weights_by_flow: Mapping[Any, Sequence[float]],
    base_state: Optional[Mapping[str, Any]] = None,
    K_lgc: Optional[int] = None,
    warm_start_reconfig_penalty: float = 5000.0,
    epsilon_a: float = 1e-4,
    epsilon_r: float = 1e-4,
    I_max: int = 50,
    max_columns_per_path: Optional[int] = None,
    max_schemes_per_hop: Optional[int] = None,
    verbose: bool = False,
) -> CPGRunResult:
    """
    论文 Algorithm 1 的主入口。

    参数里最关键的是：
    - candidate_logical_paths_by_flow
    - mapping_tables_by_flow
    - path_weights_by_flow

    如果这三者你还没有预计算好，可以先调用 prepare_candidate_logical_paths()。
    """
    flow_ids = [_flow_id(f) for f in affected_flows]
    if K_lgc is None:
        K_lgc = max((len(candidate_logical_paths_by_flow.get(fid, [])) for fid in flow_ids), default=0)

    # --------------------------------------------------------------
    # Warm start：得到 Z^0，并转成初始活动列池 Omega^0
    # --------------------------------------------------------------
    warm = warm_start_ag_s1f(
        affected_flows=affected_flows,
        break_node=break_node,
        candidate_logical_paths_by_flow=candidate_logical_paths_by_flow,
        mapping_tables_by_flow=mapping_tables_by_flow,
        path_weights_by_flow=path_weights_by_flow,
        state_after_failure=state_after_failure,
        link_index=link_index,
        reconfig_penalty=float(warm_start_reconfig_penalty),
        base_state=base_state,
        reserve_strategy1_resources_func=None,
        rollback_one_reservation_func=None,
    )

    omega: Dict[Any, List[Column]] = {fid: [] for fid in flow_ids}
    lambda_repo: Dict[Any, List[GeneratedColumn]] = {fid: [] for fid in flow_ids}

    for fid, gc in warm.restored_columns.items():
        omega[fid].append(gc.column)

    # --------------------------------------------------------------
    # 初始化 LP-RMP，并记录 J^(0)
    # --------------------------------------------------------------
    rmp_data = _build_rmp_data(affected_flows, omega, rmp_sets)
    lp_rmp, duals = solve_rmp_lp(rmp_data, verbose=verbose)
    J_prev = float(lp_rmp.model.ObjVal)
    epsilon = max(float(epsilon_a), float(epsilon_r) * abs(float(J_prev)))

    lp_objectives: Dict[int, float] = {0: J_prev}
    improvements: Dict[int, float] = {}

    cg_iter = 0
    path_rank = 1

    # --------------------------------------------------------------
    # 外层：path-rank 扩张
    # --------------------------------------------------------------
    while path_rank <= K_lgc and cg_iter < I_max:

        # ----------------------------------------------------------
        # 对每条流，在当前第 j 条逻辑路径上调用 Algorithm 3 生成候选列
        # ----------------------------------------------------------
        for flow in affected_flows:
            fid = _flow_id(flow)
            paths = list(candidate_logical_paths_by_flow.get(fid, []))
            if len(paths) < path_rank:
                continue
            digamma = paths[path_rank - 1]
            if not digamma:
                continue

            new_cols = generate_candidate_column_pool_on_path(
                flow=flow,
                logical_path=digamma,
                mapping_table=mapping_tables_by_flow[fid],
                state_after_failure=state_after_failure,
                link_index=link_index,
                strategy_mode="both",
                base_state=base_state,
                flow_metadata=None,
                max_columns=max_columns_per_path,
                max_schemes_per_hop=max_schemes_per_hop,
                keep_state_after=False,
                col_id_prefix=f"f{fid}_j{path_rank}",
            )
            _merge_generated_columns(lambda_repo, fid, new_cols)

        # ----------------------------------------------------------
        # 内层：标准 CG（在当前 Lambda 下不断把负 reduced cost 列加入 Omega）
        # ----------------------------------------------------------
        added = True
        while added and cg_iter < I_max:
            rmp_data = _build_rmp_data(affected_flows, omega, rmp_sets)
            lp_rmp, duals = solve_rmp_lp(rmp_data, verbose=verbose)
            added = False

            for flow in affected_flows:
                fid = _flow_id(flow)
                active_ids = _column_ids_in_omega(omega, fid)

                best_gc: Optional[GeneratedColumn] = None
                best_rc: Optional[float] = None

                for gc in lambda_repo.get(fid, []):
                    if str(gc.column.col_id) in active_ids:
                        continue
                    rc = compute_reduced_cost(rmp_data, fid, gc.column, duals)
                    if best_rc is None or float(rc) < float(best_rc):
                        best_rc = float(rc)
                        best_gc = gc

                if best_gc is not None and best_rc is not None and float(best_rc) < -1e-9:
                    omega[fid].append(best_gc.column)
                    added = True

            if added:
                cg_iter += 1

        # ----------------------------------------------------------
        # 当前 rank 下收敛后，记录新的 LP 目标值 J^(j)
        # ----------------------------------------------------------
        rmp_data = _build_rmp_data(affected_flows, omega, rmp_sets)
        lp_rmp, duals = solve_rmp_lp(rmp_data, verbose=verbose)
        J_curr = float(lp_rmp.model.ObjVal)
        lp_objectives[path_rank] = J_curr
        improvements[path_rank] = float(J_prev) - float(J_curr)

        # 若改善已很小，则终止后续 path-rank 扩张
        if improvements[path_rank] < epsilon:
            break

        J_prev = J_curr
        path_rank += 1

    # --------------------------------------------------------------
    # 最终整数 RMP
    # --------------------------------------------------------------
    final_data = _build_rmp_data(affected_flows, omega, rmp_sets)
    final_rmp = solve_rmp_ip(final_data, verbose=verbose)
    final_selected = get_selected_columns(final_rmp)

    return CPGRunResult(
        warm_start=warm,
        omega=omega,
        lambda_repo=lambda_repo,
        lp_objectives=lp_objectives,
        improvements=improvements,
        final_rmp=final_rmp,
        final_selected_columns=final_selected,
    )


# =====================================================================
# CLI：让它成为“可执行文件”
# =====================================================================


def _load_pickle(path: str) -> Any:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(path: str, obj: Any) -> None:
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Algorithm 1: hAG-S1F-CP-CG+ 总控程序")
    parser.add_argument("--input", required=True, help="输入 pickle 文件路径")
    parser.add_argument("--output", required=True, help="输出 pickle 文件路径")
    parser.add_argument("--eps-a", type=float, default=1e-4)
    parser.add_argument("--eps-r", type=float, default=1e-4)
    parser.add_argument("--imax", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    payload = _load_pickle(args.input)

    # 推荐的输入 pickle 结构：
    # {
    #   "affected_flows": ...,
    #   "break_node": ...,
    #   "state_after_failure": ...,
    #   "rmp_sets": ...,
    #   "link_index": ...,
    #   "candidate_logical_paths_by_flow": ...,
    #   "mapping_tables_by_flow": ...,
    #   "path_weights_by_flow": ...,
    #   "base_state": ... (optional)
    # }
    out = run_hag_s1f_cp_cg_plus(
        affected_flows=payload["affected_flows"],
        break_node=payload["break_node"],
        state_after_failure=payload["state_after_failure"],
        rmp_sets=payload["rmp_sets"],
        link_index=payload["link_index"],
        candidate_logical_paths_by_flow=payload["candidate_logical_paths_by_flow"],
        mapping_tables_by_flow=payload["mapping_tables_by_flow"],
        path_weights_by_flow=payload["path_weights_by_flow"],
        base_state=payload.get("base_state"),
        K_lgc=payload.get("K_lgc"),
        warm_start_reconfig_penalty=float(payload.get("warm_start_reconfig_penalty", 5000.0)),
        epsilon_a=args.eps_a,
        epsilon_r=args.eps_r,
        I_max=args.imax,
        max_columns_per_path=payload.get("max_columns_per_path"),
        max_schemes_per_hop=payload.get("max_schemes_per_hop"),
        verbose=args.verbose,
    )

    _save_pickle(args.output, out)
