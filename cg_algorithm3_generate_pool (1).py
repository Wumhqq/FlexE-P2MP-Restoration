#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按照论文 Algorithm 3（Pricing Routine: Generate Columns on Path）实现的
“路径级候选列池生成器”。

这份文件的目标不是直接替代你的 ILP，而是把论文里的 Algorithm 3
落成一个可以独立调用的 Python 模块：

1. 固定一条受影响流 r；
2. 固定一条逻辑路径 digamma；
3. 在给定失败后网络状态 C0 上，用 DFS 按 hop 逐跳尝试资源分配；
4. 对每个可行的完整恢复方案，转换成一个 master problem 可用的 Column；
5. 返回该路径对应的候选列池 Lambda_{r,digamma}^{(j)}。

-----------------------------------------------------------------------
依赖说明
-----------------------------------------------------------------------
这份文件会优先复用你现有工程中的两个文件：
- Initial_net.py：复用已有的 SC/FS 可行性检查与分配规则；
- rmp_master_problem.py：复用 Column 数据结构，保证和 master problem 对齐。

如果你的运行目录里已经有这些文件，直接 import 即可。
如果路径不同，请自行调整 PYTHONPATH 或 import 语句。

-----------------------------------------------------------------------
输入数据约定（核心）
-----------------------------------------------------------------------
1. flow
   一条受影响流，格式与现有工程保持一致：
   [flow_id, src, dst, bandwidth, ...]

2. logical_path
   固定的逻辑路径 digamma。
   默认使用 1-based 节点编号（与论文/现有 k_shortest_path 输出一致）。
   例如：[1, 4, 6] 表示 digamma = 1 -> 4 -> 6。

3. mapping_table
   逻辑-物理路径映射表 M_{r,u,v}。
   这里约定为字典：
       mapping_table[(u, v)] = {
           "path": [u0, x1, x2, v0],   # 0-based 物理节点序列
           "dist": 123.0
       }
   注意：key 中的 (u, v) 使用 0-based，与内部状态索引一致。

4. state
   失败后的网络状态 C0，使用字典包装，字段如下：
       state = {
           "flow_acc": ...,
           "node_flow": ...,
           "node_P2MP": ...,
           "P2MP_SC": ...,
           "link_FS": ...,
           "P2MP_FS": ...,
           "flow_path": ...,
       }

5. link_index
   物理拓扑的链路索引矩阵，和 Initial_net / AG_S1F 中一致。

6. base_state / flow_metadata
   如果要生成 Strategy 1（不重构端资源）的列，需要知道原流的端口/SC 使用信息。
   这里支持两种方式：
   - 直接传 flow_metadata；
   - 或者传 base_state，由本文件自动从 base_state 中提取 metadata。

-----------------------------------------------------------------------
输出
-----------------------------------------------------------------------
返回值是一个列表，每个元素是 GeneratedColumn：
- column：可直接放进 master problem 的 Column
- hop_schemes：该列对应的逐跳恢复方案 Pi
- logical_path：对应的 digamma
- state_after：执行完整 Pi 后的状态快照（可选用于 warm start）

作者这里特意保留了 state_after，原因是：
- Algorithm 3 生成 repository 时，一般只需要 column；
- 但 Algorithm 4/Algorithm 1 的 warm start / try-path 流程中，
  往往还需要把成功方案真正写回网络状态。
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------
# 复用你已有的 master problem 列结构
# ---------------------------------------------------------------------
from rmp_master_problem import Column

# ---------------------------------------------------------------------
# 复用你已有的 Initial_net 里的底层分配逻辑
# 这些都是你现有代码里已经验证过的约束实现，Algorithm 3 这里尽量不重写规则，
# 而是直接复用，确保“列池生成”和“你当前资源分配语义”一致。
# ---------------------------------------------------------------------
from Initial_net import (
    _TYPE_INFO,
    _apply_leaf_usage,
    _apply_p2mp_fs_usage,
    _clear_p2mp_sc,
    _find_free_fs_block,
    _sc_can_use_on_hub,
    _sc_fs_path_ok,
    _sc_used_bw_for_path,
    _find_leaf_sc_segment_with_base,
    _infer_leaf_base_and_find_sc,
    _alloc_leaf_fixed_usage,
)

# 优先复用你已经上传的 AG_S1F / other_fx_fix 中的工具函数。
try:
    from AG_S1F import _build_subflow_lookup as _ag_build_subflow_lookup
except Exception:
    _ag_build_subflow_lookup = None

try:
    from other_fx_fix import extract_old_flow_info as _extract_old_flow_info
except Exception:
    _extract_old_flow_info = None

try:
    # 这个函数如果你工程里存在，就直接复用。
    from Initial_net import sc_effective_cap as _sc_effective_cap
except Exception:
    import math

    TS_UNIT = 5

    def _sc_effective_cap(sc_cap: float) -> int:
        """把 SC 容量向下取整到 5G TS 粒度。"""
        return int(math.floor(float(sc_cap) / TS_UNIT) * TS_UNIT)


# =====================================================================
# 数据结构
# =====================================================================

@dataclass
class HopScheme:
    """
    一条逻辑 hop 的一个可行恢复方案 q。

    这正好对应论文 Algorithm 3 里的 q in Q_{u,v}。
    DFS 的时候，Pi 就是一串 HopScheme 组成的列表。
    """

    sub_id: int
    orig_flow_id: int
    src_node: int
    dst_node: int
    bandwidth: float
    sc_cap: float

    hub_p: int
    hub_type: int
    hub_base_fs: int

    leaf_p: int
    leaf_type: int
    leaf_base_fs: int

    hub_sc_start: int
    hub_sc_end: int
    leaf_sc_start: int
    leaf_sc_end: int

    fs_abs_start: int
    fs_abs_end: int

    # 例如 {sc_idx: used_bw}
    hub_usage: Dict[int, float]
    leaf_usage: Dict[int, float]

    # 1-based 物理路径节点序列；用于和现有状态结构保持一致
    phys_nodes_1b: List[int]

    # 0-based 物理链路索引列表
    used_links_0b: List[int]

    # 0-based 物理有向边列表 (i,j)
    used_link_pairs_0b: List[Tuple[int, int]]

    # 该 hop 是否属于 Strategy 1（不发生端资源重构）
    strict_s1: bool

    # 该 hop 对应的逻辑 hop
    logical_hop_1b: Tuple[int, int]


@dataclass
class GeneratedColumn:
    """Algorithm 3 生成的一条完整列及其附带信息。"""

    column: Column
    hop_schemes: List[HopScheme]
    logical_path: List[int]
    state_after: Optional[Dict[str, Any]] = None


# =====================================================================
# 基础工具函数
# =====================================================================


def _deepcopy_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    """对网络状态做深拷贝，保证 DFS 分支彼此隔离。"""
    return {
        "flow_acc": copy.deepcopy(state["flow_acc"]),
        "node_flow": copy.deepcopy(state["node_flow"]),
        "node_P2MP": copy.deepcopy(state["node_P2MP"]),
        "P2MP_SC": copy.deepcopy(state["P2MP_SC"]),
        "link_FS": copy.deepcopy(state["link_FS"]),
        "P2MP_FS": copy.deepcopy(state["P2MP_FS"]),
        "flow_path": copy.deepcopy(state["flow_path"]),
    }


def _build_subflow_lookup(flow_acc: np.ndarray) -> Dict[Tuple[int, int, int], int]:
    """建立 (原始流ID, hop_a, hop_b) -> subflow_id 的映射。优先复用 AG_S1F.py。"""
    if _ag_build_subflow_lookup is not None:
        return _ag_build_subflow_lookup(flow_acc)

    m: Dict[Tuple[int, int, int], int] = {}
    for row in flow_acc:
        try:
            sub_id = int(row[0])
            a = int(row[1])
            b = int(row[2])
            orig = int(row[6])
            m[(orig, a, b)] = sub_id
        except Exception:
            continue
    return m


def _extract_flow_metadata_from_base(
    flow: Sequence[Any],
    flow_acc_base: np.ndarray,
    P2MP_SC_base: np.ndarray,
) -> Dict[str, Any]:
    """
    从 base_state 中提取 Strategy 1 所需的端点元信息。

    优先复用 other_fx_fix.py 里的 extract_old_flow_info；
    如果外部函数不可用，再退回本地兜底实现。
    """
    if _extract_old_flow_info is not None:
        return _extract_old_flow_info(int(flow[0]), int(flow[1]), int(flow[2]), flow_acc_base, P2MP_SC_base)

    fid = int(flow[0])
    src = int(flow[1])
    dst = int(flow[2])

    meta: Dict[str, Any] = {
        "hub_idx": -1,
        "leaf_idx": -1,
        "hub_cap": 0.0,
        "leaf_cap": 0.0,
        "hub_sc_range": None,
        "leaf_sc_range": None,
        "hub_sc_usage": {},
        "leaf_sc_usage": {},
    }

    src_row = None
    dst_row = None
    for row in flow_acc_base:
        try:
            orig_id = int(row[6])
            a = int(row[1])
            b = int(row[2])
        except Exception:
            continue
        if orig_id != fid:
            continue
        if a == src and src_row is None:
            src_row = row
        if b == dst and dst_row is None:
            dst_row = row

    if src_row is not None:
        hub_p = int(src_row[7])
        sc_s = int(src_row[9])
        sc_e = int(src_row[10])
        sub_id = int(src_row[0])
        meta["hub_idx"] = hub_p
        meta["hub_sc_range"] = (sc_s, sc_e)
        meta["hub_cap"] = float(src_row[4]) if len(src_row) > 4 else 0.0
        usage: Dict[int, float] = {}
        for sc in range(sc_s, sc_e + 1):
            try:
                entries = P2MP_SC_base[src][hub_p][sc][1]
            except Exception:
                entries = []
            used_amt = 0.0
            for ent in entries or []:
                try:
                    if int(ent[0]) == sub_id or int(ent[2]) == fid:
                        used_amt += float(ent[1])
                except Exception:
                    continue
            if used_amt > 1e-9:
                usage[int(sc)] = float(used_amt)
        meta["hub_sc_usage"] = usage

    if dst_row is not None:
        leaf_p = int(dst_row[8])
        sc_s = int(dst_row[13])
        sc_e = int(dst_row[14])
        sub_id = int(dst_row[0])
        meta["leaf_idx"] = leaf_p
        meta["leaf_sc_range"] = (sc_s, sc_e)
        meta["leaf_cap"] = float(dst_row[4]) if len(dst_row) > 4 else 0.0
        usage = {}
        for sc in range(sc_s, sc_e + 1):
            try:
                entries = P2MP_SC_base[dst][leaf_p][sc][1]
            except Exception:
                entries = []
            used_amt = 0.0
            for ent in entries or []:
                try:
                    if int(ent[0]) == sub_id or int(ent[2]) == fid:
                        used_amt += float(ent[1])
                except Exception:
                    continue
            if used_amt > 1e-9:
                usage[int(sc)] = float(used_amt)
        meta["leaf_sc_usage"] = usage

    return meta


def _compute_used_links_and_pairs(
    phy_path0: Sequence[int],
    link_index: np.ndarray,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    把物理路径上的节点序列转成：
    - used_links_0b：0-based 链路索引
    - used_link_pairs_0b：0-based 有向边对 (i,j)
    """
    used_links: List[int] = []
    used_pairs: List[Tuple[int, int]] = []

    for u, v in zip(phy_path0[:-1], phy_path0[1:]):
        lk = int(link_index[u][v])
        if lk <= 0:
            raise ValueError(f"link_index[{u},{v}] 非法，无法映射到物理链路")
        used_links.append(lk - 1)
        used_pairs.append((int(u), int(v)))

    return used_links, used_pairs


# =====================================================================
# leaf 侧枚举：相比 Initial_net._alloc_new_leaf，这里返回“所有可行方案”而不是第一个。
# =====================================================================


def _enumerate_leaf_allocations(
    node_P2MP: np.ndarray,
    node_flow: np.ndarray,
    P2MP_SC: np.ndarray,
    P2MP_FS: np.ndarray,
    *,
    dest: int,
    src_node: int,
    src_p: int,
    fs_s_glo: int,
    fs_e_glo: int,
    span_len: int,
    phys_nodes: List[int],
    des: int,
    usage: Dict[int, float],
    sc_cap: float,
    fixed_leaf: Optional[int] = None,
    fixed_leaf_sc_range: Optional[Tuple[int, int]] = None,
    fixed_leaf_usage: Optional[Dict[int, float]] = None,
    max_leaf_plans: Optional[int] = None,
) -> List[Tuple[int, int, int, int, Dict[int, float], bool]]:
    """
    返回所有可行的 leaf 分配方案。

    返回元素格式：
    (leaf_p, leaf_type, leaf_base_fs, leaf_usage_start, leaf_usage_dict, is_new_leaf)

    说明：
    - 为了与 hub 侧绝对 FS 完全对齐，会优先在已启用 leaf 中找；
    - 如果找不到，再尝试空闲 leaf；
    - 如果给了 fixed_leaf，则只允许该端口。
    """
    results: List[Tuple[int, int, int, int, Dict[int, float], bool]] = []
    _p2mp_total = node_P2MP.shape[1]

    if fixed_leaf is not None and fixed_leaf_usage is not None and fixed_leaf_sc_range is not None:
        leaf_p, leaf_sc_s, leaf_sc_e, leaf_usage = _alloc_leaf_fixed_usage(
            node_P2MP,
            P2MP_SC,
            P2MP_FS,
            dest,
            src_node,
            src_p,
            fs_s_glo,
            fs_e_glo,
            phys_nodes,
            des,
            sc_cap,
            sum(float(v) for v in fixed_leaf_usage.values()),
            fixed_leaf,
            fixed_leaf_sc_range,
            fixed_leaf_usage,
        )
        if leaf_p is not None:
            leaf_type = int(node_P2MP[dest][leaf_p][3])
            leaf_base_fs = int(node_P2MP[dest][leaf_p][5])
            results.append((int(leaf_p), leaf_type, leaf_base_fs, int(leaf_sc_s), dict(leaf_usage), False))
        return results

    # 1) 先枚举已启用 leaf
    p_iter: Iterable[int]
    if fixed_leaf is not None:
        p_iter = [int(fixed_leaf)]
    else:
        p_iter = range(_p2mp_total)

    for p in p_iter:
        if int(node_P2MP[dest][p][2]) != 1:
            continue
        leaf_type = int(node_P2MP[dest][p][3])
        leaf_base = int(node_P2MP[dest][p][5])
        seg = _find_leaf_sc_segment_with_base(
            P2MP_SC,
            P2MP_FS,
            dest,
            p,
            leaf_type,
            leaf_base,
            fs_s_glo,
            fs_e_glo,
            span_len,
            src_node,
            src_p,
            phys_nodes,
            des,
            usage,
            sc_cap,
        )
        if seg is None:
            continue
        leaf_sc_s, leaf_sc_e, leaf_usage = seg
        results.append((int(p), leaf_type, leaf_base, int(leaf_sc_s), dict(leaf_usage), False))
        if max_leaf_plans is not None and len(results) >= max_leaf_plans:
            return results

    # 2) 再枚举空闲 leaf
    if fixed_leaf is None:
        free_iter = range(_p2mp_total)
    else:
        free_iter = [int(fixed_leaf)]

    for p in free_iter:
        if int(node_P2MP[dest][p][2]) != 0:
            continue
        leaf_type = int(node_P2MP[dest][p][3])
        seg = _infer_leaf_base_and_find_sc(
            P2MP_SC,
            P2MP_FS,
            dest,
            p,
            leaf_type,
            src_node,
            src_p,
            fs_s_glo,
            fs_e_glo,
            span_len,
            phys_nodes,
            des,
            usage,
            sc_cap,
        )
        if seg is None:
            continue
        leaf_sc_s, leaf_sc_e, leaf_base_fs, leaf_usage = seg
        results.append((int(p), leaf_type, int(leaf_base_fs), int(leaf_sc_s), dict(leaf_usage), True))
        if max_leaf_plans is not None and len(results) >= max_leaf_plans:
            return results

    return results


# =====================================================================
# hub+SC+leaf 枚举：这是 Algorithm 3 中“Generate hop restoration schemes Q_{u,v}”的核心。
# =====================================================================


def _enumerate_sc_plans(
    *,
    src: int,
    hub_p: int,
    hub_type: int,
    hub_fs0: int,
    dest: int,
    phys_nodes_1b: List[int],
    bw: float,
    sc_cap: float,
    flow_ids_on_hub: list,
    flow_acc: np.ndarray,
    P2MP_SC: np.ndarray,
    node_P2MP: np.ndarray,
    node_flow: np.ndarray,
    link_FS: np.ndarray,
    P2MP_FS: np.ndarray,
    used_links: List[int],
    used_link_pairs: List[Tuple[int, int]],
    flow_path: np.ndarray,
    fixed_hub_sc_range: Optional[Tuple[int, int]] = None,
    fixed_hub_usage: Optional[Dict[int, float]] = None,
    fixed_leaf: Optional[int] = None,
    fixed_leaf_sc_range: Optional[Tuple[int, int]] = None,
    fixed_leaf_usage: Optional[Dict[int, float]] = None,
    max_plans: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    枚举给定 hub 上所有可行的 SC / FS / leaf 方案。

    它相当于把 Initial_net._plan_sc_allocation 从“返回第一条可行方案”
    改写成“返回所有可行方案（或在 max_plans 处截断）”。
    """
    results: List[Dict[str, Any]] = []

    max_sc_idx = _TYPE_INFO[int(hub_type)][0] - 1
    sc_cap_use = float(_sc_effective_cap(sc_cap))

    if fixed_hub_sc_range is not None and fixed_hub_usage is not None:
        sc_start_candidates = [int(fixed_hub_sc_range[0])]
        fixed_sc_end = int(fixed_hub_sc_range[1])
    else:
        sc_start_candidates = list(range(0, max_sc_idx + 1))
        fixed_sc_end = None

    for sc_start in sc_start_candidates:
        usage: Optional[Dict[int, float]] = None
        sc_end: Optional[int] = None

        if fixed_hub_sc_range is not None and fixed_hub_usage is not None:
            # ----------------------------------------------------------
            # 固定 hub SC 方案（对应 Strategy 1）
            # ----------------------------------------------------------
            usage = {int(k): float(v) for k, v in fixed_hub_usage.items()}
            sc_end = int(fixed_sc_end)

            if sc_start < 0 or sc_end < sc_start or sc_end > max_sc_idx:
                usage = None
            else:
                expected_sc = set(range(int(sc_start), int(sc_end) + 1))
                if set(usage.keys()) != expected_sc:
                    usage = None
                elif abs(sum(float(v) for v in usage.values()) - float(bw)) > 1e-6:
                    usage = None

            if usage is not None:
                for sc_check in range(int(sc_start), int(sc_end) + 1):
                    if not _sc_can_use_on_hub(P2MP_SC, src, hub_p, int(sc_check), phys_nodes_1b, dest):
                        usage = None
                        break
                    if not _sc_fs_path_ok(P2MP_FS, src, hub_p, int(hub_type), int(sc_check), phys_nodes_1b):
                        usage = None
                        break
                    used_in_sc = _sc_used_bw_for_path(src, hub_p, int(sc_check), phys_nodes_1b, P2MP_SC)
                    if float(used_in_sc) + float(usage[int(sc_check)]) > float(sc_cap_use) + 1e-9:
                        usage = None
                        break
        else:
            # ----------------------------------------------------------
            # 非固定模式：从 sc_start 开始，按连续 SC 贪心装入 bw。
            # ----------------------------------------------------------
            remaining = float(bw)
            usage = {}
            cur_sc = int(sc_start)
            while remaining > 1e-9:
                if cur_sc > max_sc_idx:
                    usage = None
                    break
                if not _sc_can_use_on_hub(P2MP_SC, src, hub_p, int(cur_sc), phys_nodes_1b, dest):
                    usage = None
                    break
                if not _sc_fs_path_ok(P2MP_FS, src, hub_p, int(hub_type), int(cur_sc), phys_nodes_1b):
                    usage = None
                    break

                used_in_sc = _sc_used_bw_for_path(src, hub_p, int(cur_sc), phys_nodes_1b, P2MP_SC)
                cur_spare = float(sc_cap_use) - float(used_in_sc)
                if cur_spare <= 1e-9:
                    usage = None
                    break

                take = min(float(remaining), float(cur_spare))
                usage[int(cur_sc)] = usage.get(int(cur_sc), 0.0) + float(take)
                remaining -= float(take)
                if remaining <= 1e-9:
                    break
                cur_sc += 1

            if usage is not None:
                sc_end = int(cur_sc)

        if usage is None or sc_end is None:
            continue

        # --------------------------------------------------------------
        # 计算 hub 侧相对/绝对 FS 区间
        # --------------------------------------------------------------
        try:
            from SC_FS import sc_fs  # 用户工程里通常已有这个模块
        except Exception as exc:
            raise ImportError("需要 SC_FS.sc_fs 才能把 SC 区间映射到 FS 区间") from exc

        fs_s_rel = int(sc_fs(int(hub_type), int(sc_start), 1))
        fs_e_rel = int(sc_fs(int(hub_type), int(sc_end), 2))
        fs_s_glo = int(hub_fs0) + int(fs_s_rel)
        fs_e_glo = int(hub_fs0) + int(fs_e_rel)

        # --------------------------------------------------------------
        # 链路层 non-overlap 检查：沿用 Initial_net._plan_sc_allocation 的逻辑
        # --------------------------------------------------------------
        tree_links = set()
        for f0 in flow_ids_on_hub:
            fid0 = int(f0[0])
            try:
                for l0 in flow_path[fid0][1]:
                    tree_links.add(int(l0))
            except Exception:
                continue

        new_links = set(used_links) - set(tree_links)
        new_fs_abs_list: List[int] = []
        for fs_rel in range(int(fs_s_rel), int(fs_e_rel) + 1):
            try:
                used_list = P2MP_FS[src][hub_p][int(fs_rel)][1]
            except Exception:
                used_list = []
            if not used_list:
                new_fs_abs_list.append(int(hub_fs0) + int(fs_rel))

        old_links = set(used_links) & set(tree_links)
        ok_links = True

        if new_fs_abs_list and old_links:
            for l in old_links:
                for fs_abs in new_fs_abs_list:
                    if int(link_FS[int(l)][int(fs_abs)]) != 0:
                        ok_links = False
                        break
                if not ok_links:
                    break

        if ok_links and new_links:
            for l in new_links:
                for fs_abs in range(int(fs_s_glo), int(fs_e_glo) + 1):
                    if int(link_FS[int(l)][int(fs_abs)]) != 0:
                        ok_links = False
                        break
                if not ok_links:
                    break

        if not ok_links:
            continue

        # --------------------------------------------------------------
        # 检查在该 hub 相对 FS 上，是否已经隐含出了固定 leaf
        # --------------------------------------------------------------
        fixed_leaf_from_fs = None
        fixed_leaf_invalid = False
        for fs_rel in range(int(fs_s_rel), int(fs_e_rel) + 1):
            try:
                dst_node_list = P2MP_FS[src][hub_p][int(fs_rel)][7]
                dst_p_list = P2MP_FS[src][hub_p][int(fs_rel)][8]
            except Exception:
                dst_node_list = []
                dst_p_list = []

            if not dst_node_list or not dst_p_list:
                continue

            for idx in range(min(len(dst_node_list), len(dst_p_list))):
                try:
                    dst_i = int(dst_node_list[idx])
                    leaf_i = int(dst_p_list[idx])
                except Exception:
                    continue
                if dst_i != int(dest):
                    fixed_leaf_invalid = True
                    break
                if leaf_i < 0:
                    fixed_leaf_invalid = True
                    break
                if fixed_leaf_from_fs is None:
                    fixed_leaf_from_fs = int(leaf_i)
                elif int(fixed_leaf_from_fs) != int(leaf_i):
                    fixed_leaf_invalid = True
                    break
            if fixed_leaf_invalid:
                break

        if fixed_leaf_invalid:
            continue

        if fixed_leaf is not None and fixed_leaf_from_fs is not None and int(fixed_leaf) != int(fixed_leaf_from_fs):
            continue

        leaf_fixed_use = int(fixed_leaf) if fixed_leaf is not None else fixed_leaf_from_fs

        # --------------------------------------------------------------
        # 枚举所有可行 leaf 分配
        # --------------------------------------------------------------
        leaf_allocs = _enumerate_leaf_allocations(
            node_P2MP,
            node_flow,
            P2MP_SC,
            P2MP_FS,
            dest=dest,
            src_node=src,
            src_p=hub_p,
            fs_s_glo=fs_s_glo,
            fs_e_glo=fs_e_glo,
            span_len=int(sc_end) - int(sc_start) + 1,
            phys_nodes=phys_nodes_1b,
            des=dest,
            usage=usage,
            sc_cap=sc_cap,
            fixed_leaf=leaf_fixed_use,
            fixed_leaf_sc_range=fixed_leaf_sc_range,
            fixed_leaf_usage=fixed_leaf_usage,
            max_leaf_plans=None,
        )

        for leaf_p, leaf_type, leaf_base_fs, leaf_sc_s, leaf_usage, is_new_leaf in leaf_allocs:
            leaf_sc_e = int(leaf_sc_s) + len(leaf_usage) - 1
            results.append(
                {
                    "hub_sc_start": int(sc_start),
                    "hub_sc_end": int(sc_end),
                    "hub_usage": dict(usage),
                    "fs_abs_start": int(fs_s_glo),
                    "fs_abs_end": int(fs_e_glo),
                    "leaf_p": int(leaf_p),
                    "leaf_type": int(leaf_type),
                    "leaf_base_fs": int(leaf_base_fs),
                    "leaf_sc_start": int(leaf_sc_s),
                    "leaf_sc_end": int(leaf_sc_e),
                    "leaf_usage": dict(leaf_usage),
                    "is_new_leaf": bool(is_new_leaf),
                }
            )
            if max_plans is not None and len(results) >= max_plans:
                return results

    return results


# =====================================================================
# 单 hop 候选方案生成（Q_{u,v}）
# =====================================================================


def generate_hop_restoration_schemes(
    *,
    flow: Sequence[Any],
    hop_src_0b: int,
    hop_dst_0b: int,
    phy_path0: Sequence[int],
    phy_dist: float,
    link_index: np.ndarray,
    state: Mapping[str, Any],
    strict_s1: bool,
    flow_metadata: Optional[Mapping[str, Any]] = None,
    subflow_lookup: Optional[Mapping[Tuple[int, int, int], int]] = None,
    max_schemes_per_hop: Optional[int] = None,
) -> List[HopScheme]:
    """
    生成论文 Algorithm 3 中当前 hop 的候选集合 Q_{u,v}。

    这是整个文件里最关键的接口之一：
    - 输入一个固定逻辑 hop；
    - 输入当前残余状态 C；
    - 返回所有可行的 hop 级恢复方案 q。
    """
    if subflow_lookup is None:
        subflow_lookup = _build_subflow_lookup(state["flow_acc"])

    if flow_metadata is None:
        flow_metadata = {}

    if not phy_path0 or len(phy_path0) < 2:
        return []

    fid = int(flow[0])
    band = float(flow[3])
    sub_id = subflow_lookup.get((fid, int(hop_src_0b), int(hop_dst_0b)))
    if sub_id is None:
        return []
    sub_id = int(sub_id)
    orig_fid = int(state["flow_acc"][sub_id][6]) if int(state["flow_acc"][sub_id][6]) >= 0 else fid

    used_links, used_link_pairs = _compute_used_links_and_pairs(phy_path0, link_index)
    phys_nodes_1b = [int(x) + 1 for x in phy_path0]

    sc_cap = float(_sc_effective_cap(float(phy_dist if False else 0.0)))  # 占位，下面会被覆盖
    try:
        from modu_format_Al import modu_format_Al
        sc_cap_raw, _ = modu_format_Al(float(phy_dist), float(band))
        sc_cap = float(_sc_effective_cap(float(sc_cap_raw)))
    except Exception:
        # 如果用户本地环境还没把 modu_format_Al 放进路径，这里至少不要直接崩掉。
        # 退化策略：尽量从 flow_acc[sub_id][4] 中拿已经预计算好的 SC_cap。
        try:
            sc_cap = float(state["flow_acc"][sub_id][4])
        except Exception:
            raise ImportError("需要 modu_format_Al 或者 flow_acc[sub_id][4] 中存在可用 SC_cap")

    # --------------------------------------------------------------
    # Strategy 1：端点资源固定
    # --------------------------------------------------------------
    fixed_hub = None
    fixed_leaf = None
    fixed_hub_sc_range = None
    fixed_leaf_sc_range = None
    fixed_hub_usage = None
    fixed_leaf_usage = None

    if strict_s1:
        if int(flow[1]) == int(hop_src_0b) and int(flow_metadata.get("hub_idx", -1)) != -1:
            fixed_hub = int(flow_metadata["hub_idx"])
        if int(flow[2]) == int(hop_dst_0b) and int(flow_metadata.get("leaf_idx", -1)) != -1:
            fixed_leaf = int(flow_metadata["leaf_idx"])

        if fixed_hub is not None and fixed_leaf is not None:
            hub_cap = float(flow_metadata.get("hub_cap", 0.0))
            leaf_cap = float(flow_metadata.get("leaf_cap", 0.0))
            hub_eff = float(_sc_effective_cap(hub_cap)) if hub_cap > 0 else 0.0
            leaf_eff = float(_sc_effective_cap(leaf_cap)) if leaf_cap > 0 else 0.0
            if hub_eff <= 0 or leaf_eff <= 0 or abs(hub_eff - leaf_eff) > 1e-9:
                return []
            sc_cap = hub_eff
        elif fixed_hub is not None:
            old_cap = float(flow_metadata.get("hub_cap", 0.0))
            if old_cap > 0:
                sc_cap = float(_sc_effective_cap(old_cap))
        elif fixed_leaf is not None:
            old_cap = float(flow_metadata.get("leaf_cap", 0.0))
            if old_cap > 0:
                sc_cap = float(_sc_effective_cap(old_cap))

        if fixed_hub is not None:
            hub_range = flow_metadata.get("hub_sc_range")
            if hub_range is not None:
                fixed_hub_sc_range = (int(hub_range[0]), int(hub_range[1]))
            hub_usage = flow_metadata.get("hub_sc_usage")
            if hub_usage is not None:
                fixed_hub_usage = {int(k): float(v) for k, v in hub_usage.items()}

        if fixed_leaf is not None:
            leaf_range = flow_metadata.get("leaf_sc_range")
            if leaf_range is not None:
                fixed_leaf_sc_range = (int(leaf_range[0]), int(leaf_range[1]))
            leaf_usage = flow_metadata.get("leaf_sc_usage")
            if leaf_usage is not None:
                fixed_leaf_usage = {int(k): float(v) for k, v in leaf_usage.items()}

    node_P2MP = state["node_P2MP"]
    node_flow = state["node_flow"]
    P2MP_SC = state["P2MP_SC"]
    link_FS = state["link_FS"]
    P2MP_FS = state["P2MP_FS"]
    flow_acc = state["flow_acc"]
    flow_path = state["flow_path"]

    _p2mp_total = node_P2MP.shape[1]
    hub_candidates = [fixed_hub] if fixed_hub is not None else list(range(_p2mp_total))

    results: List[HopScheme] = []

    # --------------------------------------------------------------
    # A. 先枚举已启用 hub
    # --------------------------------------------------------------
    for hub_p in hub_candidates:
        if hub_p is None:
            continue
        if int(node_P2MP[hop_src_0b][hub_p][2]) != 1:
            continue

        used_bw = sum(float(x[3]) for x in node_flow[hop_src_0b][hub_p][0])
        if used_bw + band > 400 + 1e-9:
            continue

        hub_type = int(node_P2MP[hop_src_0b][hub_p][3])
        hub_fs0 = int(node_P2MP[hop_src_0b][hub_p][5])
        if hub_fs0 < 0:
            continue

        flow_ids_on_hub = node_flow[hop_src_0b][hub_p][0]
        plans = _enumerate_sc_plans(
            src=int(hop_src_0b),
            hub_p=int(hub_p),
            hub_type=int(hub_type),
            hub_fs0=int(hub_fs0),
            dest=int(hop_dst_0b),
            phys_nodes_1b=phys_nodes_1b,
            bw=float(band),
            sc_cap=float(sc_cap),
            flow_ids_on_hub=flow_ids_on_hub,
            flow_acc=flow_acc,
            P2MP_SC=P2MP_SC,
            node_P2MP=node_P2MP,
            node_flow=node_flow,
            link_FS=link_FS,
            P2MP_FS=P2MP_FS,
            used_links=used_links,
            used_link_pairs=used_link_pairs,
            flow_path=flow_path,
            fixed_hub_sc_range=fixed_hub_sc_range,
            fixed_hub_usage=fixed_hub_usage,
            fixed_leaf=fixed_leaf,
            fixed_leaf_sc_range=fixed_leaf_sc_range,
            fixed_leaf_usage=fixed_leaf_usage,
            max_plans=max_schemes_per_hop,
        )

        for plan in plans:
            results.append(
                HopScheme(
                    sub_id=int(sub_id),
                    orig_flow_id=int(orig_fid),
                    src_node=int(hop_src_0b),
                    dst_node=int(hop_dst_0b),
                    bandwidth=float(band),
                    sc_cap=float(sc_cap),
                    hub_p=int(hub_p),
                    hub_type=int(hub_type),
                    hub_base_fs=int(hub_fs0),
                    leaf_p=int(plan["leaf_p"]),
                    leaf_type=int(plan["leaf_type"]),
                    leaf_base_fs=int(plan["leaf_base_fs"]),
                    hub_sc_start=int(plan["hub_sc_start"]),
                    hub_sc_end=int(plan["hub_sc_end"]),
                    leaf_sc_start=int(plan["leaf_sc_start"]),
                    leaf_sc_end=int(plan["leaf_sc_end"]),
                    fs_abs_start=int(plan["fs_abs_start"]),
                    fs_abs_end=int(plan["fs_abs_end"]),
                    hub_usage=dict(plan["hub_usage"]),
                    leaf_usage=dict(plan["leaf_usage"]),
                    phys_nodes_1b=list(phys_nodes_1b),
                    used_links_0b=list(used_links),
                    used_link_pairs_0b=list(used_link_pairs),
                    strict_s1=bool(strict_s1),
                    logical_hop_1b=(int(hop_src_0b) + 1, int(hop_dst_0b) + 1),
                )
            )
            if max_schemes_per_hop is not None and len(results) >= max_schemes_per_hop:
                return results

    # --------------------------------------------------------------
    # B. 再枚举新开 hub
    # --------------------------------------------------------------
    for hub_p in hub_candidates:
        if hub_p is None:
            continue
        if int(node_P2MP[hop_src_0b][hub_p][2]) != 0:
            continue

        hub_type = int(node_P2MP[hop_src_0b][hub_p][3])
        block_size = _TYPE_INFO[int(hub_type)][1]
        hub_fs0 = _find_free_fs_block(link_FS, used_links, block_size)
        if hub_fs0 is None:
            continue

        # 注意：这里只是“枚举”，不能污染外部状态，因此这里复制一份临时状态来测。
        tmp_node_flow = copy.deepcopy(node_flow)
        tmp_node_P2MP = copy.deepcopy(node_P2MP)
        tmp_P2MP_SC = copy.deepcopy(P2MP_SC)

        tmp_node_P2MP[hop_src_0b][hub_p][2] = 1
        tmp_node_P2MP[hop_src_0b][hub_p][5] = int(hub_fs0)
        tmp_node_flow[hop_src_0b][hub_p][0].clear()
        _clear_p2mp_sc(tmp_P2MP_SC, hop_src_0b, hub_p)

        plans = _enumerate_sc_plans(
            src=int(hop_src_0b),
            hub_p=int(hub_p),
            hub_type=int(hub_type),
            hub_fs0=int(hub_fs0),
            dest=int(hop_dst_0b),
            phys_nodes_1b=phys_nodes_1b,
            bw=float(band),
            sc_cap=float(sc_cap),
            flow_ids_on_hub=tmp_node_flow[hop_src_0b][hub_p][0],
            flow_acc=flow_acc,
            P2MP_SC=tmp_P2MP_SC,
            node_P2MP=tmp_node_P2MP,
            node_flow=tmp_node_flow,
            link_FS=link_FS,
            P2MP_FS=P2MP_FS,
            used_links=used_links,
            used_link_pairs=used_link_pairs,
            flow_path=flow_path,
            fixed_hub_sc_range=fixed_hub_sc_range,
            fixed_hub_usage=fixed_hub_usage,
            fixed_leaf=fixed_leaf,
            fixed_leaf_sc_range=fixed_leaf_sc_range,
            fixed_leaf_usage=fixed_leaf_usage,
            max_plans=max_schemes_per_hop,
        )

        for plan in plans:
            results.append(
                HopScheme(
                    sub_id=int(sub_id),
                    orig_flow_id=int(orig_fid),
                    src_node=int(hop_src_0b),
                    dst_node=int(hop_dst_0b),
                    bandwidth=float(band),
                    sc_cap=float(sc_cap),
                    hub_p=int(hub_p),
                    hub_type=int(hub_type),
                    hub_base_fs=int(hub_fs0),
                    leaf_p=int(plan["leaf_p"]),
                    leaf_type=int(plan["leaf_type"]),
                    leaf_base_fs=int(plan["leaf_base_fs"]),
                    hub_sc_start=int(plan["hub_sc_start"]),
                    hub_sc_end=int(plan["hub_sc_end"]),
                    leaf_sc_start=int(plan["leaf_sc_start"]),
                    leaf_sc_end=int(plan["leaf_sc_end"]),
                    fs_abs_start=int(plan["fs_abs_start"]),
                    fs_abs_end=int(plan["fs_abs_end"]),
                    hub_usage=dict(plan["hub_usage"]),
                    leaf_usage=dict(plan["leaf_usage"]),
                    phys_nodes_1b=list(phys_nodes_1b),
                    used_links_0b=list(used_links),
                    used_link_pairs_0b=list(used_link_pairs),
                    strict_s1=bool(strict_s1),
                    logical_hop_1b=(int(hop_src_0b) + 1, int(hop_dst_0b) + 1),
                )
            )
            if max_schemes_per_hop is not None and len(results) >= max_schemes_per_hop:
                return results

    return results


# =====================================================================
# 将一个 hop 方案写入状态（Apply q to C）
# =====================================================================


def apply_hop_scheme_to_state(hs: HopScheme, state: MutableMapping[str, Any]) -> None:
    """
    把单个 hop 方案 q 真正写入当前状态 C。

    这一步对应 Algorithm 3 / Algorithm 5 里的 “Apply q to C”。
    """
    flow_acc = state["flow_acc"]
    node_flow = state["node_flow"]
    node_P2MP = state["node_P2MP"]
    P2MP_SC = state["P2MP_SC"]
    link_FS = state["link_FS"]
    P2MP_FS = state["P2MP_FS"]
    flow_path = state["flow_path"]

    src = int(hs.src_node)
    dst = int(hs.dst_node)
    sub_id = int(hs.sub_id)
    orig_fid = int(hs.orig_flow_id)

    # 若 hub 原本未启用，则现在启用并设置 base FS。
    if int(node_P2MP[src][hs.hub_p][2]) == 0:
        node_P2MP[src][hs.hub_p][2] = 1
        node_P2MP[src][hs.hub_p][5] = int(hs.hub_base_fs)
        node_flow[src][hs.hub_p][0].clear()
        _clear_p2mp_sc(P2MP_SC, src, hs.hub_p)

    # 若 leaf 原本未启用，则现在启用并设置 base FS。
    if int(node_P2MP[dst][hs.leaf_p][2]) == 0:
        node_P2MP[dst][hs.leaf_p][2] = 1
        node_P2MP[dst][hs.leaf_p][5] = int(hs.leaf_base_fs)
        node_flow[dst][hs.leaf_p][0].clear()
        _clear_p2mp_sc(P2MP_SC, dst, hs.leaf_p)

    sc_num = int(hs.hub_sc_end) - int(hs.hub_sc_start) + 1
    subflow_info = [
        int(sub_id),
        int(src),
        int(dst),
        float(hs.bandwidth),
        float(hs.sc_cap),
        int(sc_num),
        int(orig_fid),
    ]

    node_flow[src][hs.hub_p][0].append(subflow_info)
    node_flow[dst][hs.leaf_p][0].append(subflow_info)

    flow_acc[sub_id][7] = int(hs.hub_p)
    flow_acc[sub_id][8] = int(hs.leaf_p)
    flow_acc[sub_id][9] = int(hs.hub_sc_start)
    flow_acc[sub_id][10] = int(hs.hub_sc_end)
    flow_acc[sub_id][11] = int(hs.fs_abs_start)
    flow_acc[sub_id][12] = int(hs.fs_abs_end)
    flow_acc[sub_id][13] = int(hs.leaf_sc_start)
    flow_acc[sub_id][14] = int(hs.leaf_sc_end)

    # hub 侧 SC / FS 记录
    for sc, used_amt in hs.hub_usage.items():
        P2MP_SC[src][hs.hub_p][int(sc)][1].append([int(sub_id), float(used_amt), int(orig_fid)])
        P2MP_SC[src][hs.hub_p][int(sc)][0].append(float(hs.sc_cap))
        P2MP_SC[src][hs.hub_p][int(sc)][2].append(list(hs.phys_nodes_1b))
        P2MP_SC[src][hs.hub_p][int(sc)][4].append(int(dst))
        P2MP_SC[src][hs.hub_p][int(sc)][5].append(int(src))
        P2MP_SC[src][hs.hub_p][int(sc)][6].append(int(hs.hub_p))
        P2MP_SC[src][hs.hub_p][int(sc)][7].append(int(dst))
        P2MP_SC[src][hs.hub_p][int(sc)][8].append(int(hs.leaf_p))

        _apply_p2mp_fs_usage(
            P2MP_FS,
            src,
            hs.hub_p,
            int(hs.hub_type),
            int(sc),
            int(sub_id),
            float(used_amt),
            list(hs.phys_nodes_1b),
            int(dst),
            int(orig_fid),
            int(src),
            int(hs.hub_p),
            int(dst),
            int(hs.leaf_p),
        )

    flow_path[sub_id][0] = [int(sub_id)]
    flow_path[sub_id][1] = list(hs.used_links_0b)
    flow_path[sub_id][2] = list(hs.phys_nodes_1b)
    flow_path[sub_id][3] = [int(orig_fid)]

    _apply_leaf_usage(
        P2MP_SC,
        int(dst),
        int(hs.leaf_p),
        int(hs.leaf_sc_start),
        int(hs.leaf_sc_end),
        hs.leaf_usage,
        float(hs.sc_cap),
        int(sub_id),
        list(hs.phys_nodes_1b),
        int(dst),
        int(orig_fid),
        int(src),
        int(hs.hub_p),
        int(dst),
        int(hs.leaf_p),
    )

    for sc, used_amt in hs.leaf_usage.items():
        _apply_p2mp_fs_usage(
            P2MP_FS,
            dst,
            hs.leaf_p,
            int(hs.leaf_type),
            int(sc),
            int(sub_id),
            float(used_amt),
            list(hs.phys_nodes_1b),
            int(dst),
            int(orig_fid),
            int(src),
            int(hs.hub_p),
            int(dst),
            int(hs.leaf_p),
        )

    for l in set(hs.used_links_0b):
        link_FS[int(l)][int(hs.fs_abs_start): int(hs.fs_abs_end) + 1] += 1


# =====================================================================
# 将完整 Pi 转成 master problem 的 Column
# =====================================================================


def _append_sparse(d: MutableMapping, key: Tuple[Any, ...], value: float) -> None:
    """稀疏字典累加。"""
    d[key] = float(d.get(key, 0.0)) + float(value)


def build_column_from_scheme(
    flow: Sequence[Any],
    logical_path: Sequence[int],
    hop_schemes: Sequence[HopScheme],
    node_P2MP_after: np.ndarray,
    *,
    col_id: str,
) -> Column:
    """
    把一串 hop 方案 Pi 转成一条完整列 k。

    这一步对应论文里：
    "Build a column k from (r, digamma, Pi)"。
    """
    col = Column(col_id=col_id, f_rf_clr=0)

    # 如果任何一个 hop 需要 Strategy 2（重构），则整条列记为 RF-CLR。
    if any(not hs.strict_s1 for hs in hop_schemes):
        col.f_rf_clr = 1

    for hs in hop_schemes:
        src = int(hs.src_node)
        dst = int(hs.dst_node)
        hub_p = int(hs.hub_p)
        leaf_p = int(hs.leaf_p)

        # --------------------------------------------------------------
        # FlexE group 级别
        # --------------------------------------------------------------
        col.g_tx[(src, hub_p)] = 1.0
        col.g_rx[(dst, leaf_p)] = 1.0

        # --------------------------------------------------------------
        # SC 级别（hub 侧 Tx）
        # --------------------------------------------------------------
        for sc, used_amt in hs.hub_usage.items():
            sc = int(sc)
            col.e_tx[(src, hub_p, sc)] = 1.0
            _append_sparse(col.n_tx, (src, hub_p, sc), float(used_amt) / 5.0)
            col.a[(src, hub_p, sc)] = max(float(col.a.get((src, hub_p, sc), 0.0)), float(hs.sc_cap))
            col.xi[(src, hub_p, sc)] = 1.0

            for i, j in hs.used_link_pairs_0b:
                col.h[(src, hub_p, sc, i, j)] = 1.0

        # --------------------------------------------------------------
        # SC 级别（leaf 侧 Rx）
        # --------------------------------------------------------------
        for sc, used_amt in hs.leaf_usage.items():
            sc = int(sc)
            col.e_rx[(dst, leaf_p, sc)] = 1.0
            _append_sparse(col.n_rx, (dst, leaf_p, sc), float(used_amt) / 5.0)
            col.a[(dst, leaf_p, sc)] = max(float(col.a.get((dst, leaf_p, sc), 0.0)), float(hs.sc_cap))
            col.xi[(dst, leaf_p, sc)] = 1.0

            for i, j in hs.used_link_pairs_0b:
                col.h[(dst, leaf_p, sc, i, j)] = 1.0

        # --------------------------------------------------------------
        # FS / sigma / tau / varsigma
        # 注意：论文里的 w 对应的是“相对 FS”，master problem 里的 non-overlap
        # 则要用绝对 FS（这里就是 fs_abs_start ~ fs_abs_end）。
        # --------------------------------------------------------------
        hub_base = int(hs.hub_base_fs)
        leaf_base = int(hs.leaf_base_fs)

        for fs_abs in range(int(hs.fs_abs_start), int(hs.fs_abs_end) + 1):
            w_hub = int(fs_abs) - int(hub_base)
            w_leaf = int(fs_abs) - int(leaf_base)

            col.sigma[(src, hub_p, w_hub)] = 1.0
            col.sigma[(dst, leaf_p, w_leaf)] = 1.0

            for i, j in hs.used_link_pairs_0b:
                col.tau[(src, hub_p, w_hub, i, j)] = 1.0
                col.tau[(dst, leaf_p, w_leaf, i, j)] = 1.0
                col.varsigma_tx[(i, j, fs_abs, src, hub_p)] = 1.0
                col.varsigma_rx[(i, j, fs_abs, dst, leaf_p)] = 1.0

    # 额外附加一些调试元数据，方便你后续排错。
    col.logical_path = list(logical_path)
    col.flow_id = int(flow[0])
    col.hop_scheme_count = len(hop_schemes)

    return col


# =====================================================================
# DFS：Algorithm 3 的主体
# =====================================================================


def _column_signature(col: Column) -> Tuple[Any, ...]:
    """用于 repository 去重。"""
    return (
        int(col.f_rf_clr),
        tuple(sorted(col.g_tx.items())),
        tuple(sorted(col.g_rx.items())),
        tuple(sorted(col.e_tx.items())),
        tuple(sorted(col.e_rx.items())),
        tuple(sorted(col.n_tx.items())),
        tuple(sorted(col.n_rx.items())),
        tuple(sorted(col.a.items())),
        tuple(sorted(col.xi.items())),
        tuple(sorted(col.h.items())),
        tuple(sorted(col.sigma.items())),
        tuple(sorted(col.tau.items())),
        tuple(sorted(col.varsigma_tx.items())),
        tuple(sorted(col.varsigma_rx.items())),
    )


def generate_candidate_column_pool_on_path(
    *,
    flow: Sequence[Any],
    logical_path: Sequence[int],
    mapping_table: Mapping[Tuple[int, int], Mapping[str, Any]],
    state_after_failure: Mapping[str, Any],
    link_index: np.ndarray,
    strategy_mode: str = "both",
    base_state: Optional[Mapping[str, Any]] = None,
    flow_metadata: Optional[Mapping[str, Any]] = None,
    max_columns: Optional[int] = None,
    max_schemes_per_hop: Optional[int] = None,
    keep_state_after: bool = False,
    col_id_prefix: Optional[str] = None,
) -> List[GeneratedColumn]:
    """
    论文 Algorithm 3 的直接实现。

    参数
    ------------------------------------------------------------------
    strategy_mode : str
        - "s1"   : 只生成 Strategy 1 列
        - "s2"   : 只生成 Strategy 2 列
        - "both" : 两种都生成（默认）

    keep_state_after : bool
        True 时，会把每条完整列对应的 state_after 一并保留，便于 warm start。
    """
    if strategy_mode not in {"s1", "s2", "both"}:
        raise ValueError("strategy_mode 只能是 's1' / 's2' / 'both'")

    work_init_state = _deepcopy_state(state_after_failure)

    if flow_metadata is None and base_state is not None:
        flow_metadata = _extract_flow_metadata_from_base(
            flow,
            np.asarray(base_state["flow_acc"], dtype=object),
            np.asarray(base_state["P2MP_SC"], dtype=object),
        )
    elif flow_metadata is None:
        flow_metadata = {}

    subflow_lookup = _build_subflow_lookup(np.asarray(work_init_state["flow_acc"], dtype=object))

    logical_path = list(logical_path)
    if len(logical_path) < 2:
        return []

    results: List[GeneratedColumn] = []
    seen_signatures = set()
    fid = int(flow[0])
    col_id_prefix = col_id_prefix or f"f{fid}_path"

    # 把逻辑路径转成 0-based，便于和状态索引一致。
    logical_path_0b = [int(x) - 1 for x in logical_path]

    strategy_flags: List[bool] = []
    if strategy_mode in {"s1", "both"}:
        strategy_flags.append(True)
    if strategy_mode in {"s2", "both"}:
        strategy_flags.append(False)

    def dfs(hop_idx: int, state: Dict[str, Any], pi: List[HopScheme], strict_s1: bool) -> None:
        nonlocal results, seen_signatures

        if max_columns is not None and len(results) >= max_columns:
            return

        # --------------------------------------------------------------
        # 递归结束：说明一整条 digamma 上的所有 hop 都已成功分配
        # Build a column k from (r, digamma, Pi)
        # --------------------------------------------------------------
        if hop_idx > len(logical_path_0b) - 2:
            col_id = f"{col_id_prefix}_{strict_s1}_{len(results)}"
            col = build_column_from_scheme(
                flow=flow,
                logical_path=logical_path,
                hop_schemes=pi,
                node_P2MP_after=state["node_P2MP"],
                col_id=col_id,
            )
            sig = _column_signature(col)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                results.append(
                    GeneratedColumn(
                        column=col,
                        hop_schemes=copy.deepcopy(pi),
                        logical_path=list(logical_path),
                        state_after=_deepcopy_state(state) if keep_state_after else None,
                    )
                )
            return

        a = int(logical_path_0b[hop_idx])
        b = int(logical_path_0b[hop_idx + 1])
        if (a, b) not in mapping_table:
            return

        phy_cand = mapping_table[(a, b)]
        phy_path0 = list(phy_cand["path"])
        phy_dist = float(phy_cand["dist"])

        q_set = generate_hop_restoration_schemes(
            flow=flow,
            hop_src_0b=a,
            hop_dst_0b=b,
            phy_path0=phy_path0,
            phy_dist=phy_dist,
            link_index=link_index,
            state=state,
            strict_s1=strict_s1,
            flow_metadata=flow_metadata,
            subflow_lookup=subflow_lookup,
            max_schemes_per_hop=max_schemes_per_hop,
        )

        for q in q_set:
            next_state = _deepcopy_state(state)
            apply_hop_scheme_to_state(q, next_state)
            dfs(hop_idx + 1, next_state, pi + [q], strict_s1)
            if max_columns is not None and len(results) >= max_columns:
                return

    for strict_s1 in strategy_flags:
        dfs(0, _deepcopy_state(work_init_state), [], strict_s1)
        if max_columns is not None and len(results) >= max_columns:
            break

    return results


# =====================================================================
# 一个便于 warm start 复用的辅助函数：给定一组路径，找到第一条可行列。
# 它对应 Algorithm 5 / try-paths 的行为。
# =====================================================================


def try_candidate_paths_for_one_flow(
    *,
    flow: Sequence[Any],
    candidate_logical_paths: Sequence[Sequence[int]],
    mapping_table: Mapping[Tuple[int, int], Mapping[str, Any]],
    state: Mapping[str, Any],
    link_index: np.ndarray,
    strict_s1: bool,
    base_state: Optional[Mapping[str, Any]] = None,
    flow_metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, Optional[GeneratedColumn]]:
    """
    按路径顺序逐条尝试，只返回第一条成功列。

    这是为了让 Algorithm 1 / warm start 更容易复用。
    """
    for path in candidate_logical_paths:
        cols = generate_candidate_column_pool_on_path(
            flow=flow,
            logical_path=path,
            mapping_table=mapping_table,
            state_after_failure=state,
            link_index=link_index,
            strategy_mode="s1" if strict_s1 else "s2",
            base_state=base_state,
            flow_metadata=flow_metadata,
            max_columns=1,
            max_schemes_per_hop=1,
            keep_state_after=True,
            col_id_prefix=f"warm_f{int(flow[0])}",
        )
        if cols:
            return True, cols[0]
    return False, None


# =====================================================================
# CLI：为了让它成为“可执行文件”
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

    parser = argparse.ArgumentParser(
        description="Algorithm 3: 在固定逻辑路径上生成候选列池。"
    )
    parser.add_argument("--input", required=True, help="输入 pickle 文件路径")
    parser.add_argument("--output", required=True, help="输出 pickle 文件路径")
    parser.add_argument("--strategy", default="both", choices=["s1", "s2", "both"], help="生成哪类列")
    parser.add_argument("--max-columns", type=int, default=None, help="最多生成多少条列")
    parser.add_argument("--max-hop-schemes", type=int, default=None, help="每个 hop 最多枚举多少个 q")
    args = parser.parse_args()

    payload = _load_pickle(args.input)

    # 约定输入 pickle 至少包含：
    # {
    #   "flow": ...,
    #   "logical_path": ...,
    #   "mapping_table": ...,
    #   "state_after_failure": ...,
    #   "link_index": ...,
    #   "base_state": ... (optional),
    #   "flow_metadata": ... (optional)
    # }
    out = generate_candidate_column_pool_on_path(
        flow=payload["flow"],
        logical_path=payload["logical_path"],
        mapping_table=payload["mapping_table"],
        state_after_failure=payload["state_after_failure"],
        link_index=payload["link_index"],
        strategy_mode=args.strategy,
        base_state=payload.get("base_state"),
        flow_metadata=payload.get("flow_metadata"),
        max_columns=args.max_columns,
        max_schemes_per_hop=args.max_hop_schemes,
        keep_state_after=bool(payload.get("keep_state_after", False)),
        col_id_prefix=payload.get("col_id_prefix"),
    )

    _save_pickle(args.output, out)
