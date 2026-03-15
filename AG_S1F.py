from __future__ import annotations

"""
AG_S1F.py

策略 1 优先（S1 first）的启发式恢复算法。

这个版本在原始实现基础上做了两类改进：
1. 修复了两个会在运行期触发 NameError 的确定性问题；
2. 补充了模块级与关键路径级注释，方便后续维护和联调。

核心思路：
- 先从 DP 基线状态中提取受影响业务的历史资源占用；
- 对每条业务分别构建“逻辑层候选路径 + 物理层候选路径”；
- 在恢复时并列考虑两种算法：
  1) 基于逻辑拓扑的多跳算法；
  2) 直接 src->dst 的单跳算法；
- 优先尝试保持原有端口/SC/硬件约束（strict_s1=True）；
- 若严格复用失败，再进入允许重构的策略 2。

主要输入/输出状态：
- new_flow_acc   : 子流级资源映射表
- new_node_flow  : 每个节点每个 P2MP 端口当前承载的子流列表
- new_node_P2MP  : 每个节点每个 P2MP 端口的启用状态/类型/FS base 等
- new_P2MP_SC    : 端口-时隙级占用详情
- new_P2MP_FS    : 端口-FS 级占用详情
- new_link_FS    : 链路绝对 FS 占用
- new_flow_path  : 子流映射到物理链路/节点路径的记录

注意：
- 本文件默认节点内部索引使用 0-based；
- 调用 k_shortest_path / 某些路径记录时会临时转换为 1-based；
- SC 容量统一按 TS_UNIT 对齐，避免出现 2.5 TS 这类不可实际分配的粒度。
"""

import copy
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import topology as tp
from k_shortest_path import k_shortest_path
from modu_format_Al import modu_format_Al
from SC_FS import sc_fs

from Initial_net import (
    TS_UNIT,
    _TYPE_INFO,
    _apply_leaf_usage,
    _apply_p2mp_fs_usage,
    _clear_p2mp_sc,
    _plan_sc_allocation,
)
from other_fx_fix import (
    build_virtual_topology,
    extract_old_flow_info,
    manage_s1_reservation_by_copy,
)

def sc_effective_cap(sc_cap: float) -> int:
    """把原始 SC 容量压缩到“按 TS 粒度可真正落地”的容量。

    这里不能直接使用 modu_format_Al 返回的理论容量，
    因为资源分配是按 TS_UNIT=5Gbps 的整数倍进行的。
    例如：12.5Gbps 只能承载 floor(12.5/5)=2 个 TS => 10Gbps。
    """
    return int(math.floor(float(sc_cap) / TS_UNIT) * TS_UNIT)

def sc_num_from_bw_cap(bw: float, sc_cap: float) -> int:
    eff = sc_effective_cap(sc_cap)
    if eff <= 0:
        raise ValueError(f"Invalid SC_cap={sc_cap}")
    return int(math.ceil(float(bw) / eff))

def logical_path_is_valid(path_1b: List[int], break_node_0b: int) -> bool:
    """逻辑路径合理时返回 True：非空、不包含断点且不存在重复节点。"""
    if not path_1b or len(path_1b) < 2:
        return False
    if (break_node_0b + 1) in path_1b:
        return False
    if len(set(path_1b)) != len(path_1b):
        return False
    return True


def _find_free_fs_block_candidates(
    link_FS: np.ndarray,
    used_links: List[int],
    block_size: int,
) -> List[int]:
    """返回所有可用的连续 FS block 起点，而不是只返回第一个。"""
    if len(used_links) == 0:
        combined_usage = np.zeros(link_FS.shape[1], dtype=link_FS.dtype)
    else:
        combined_usage = link_FS[used_links].sum(axis=0)

    fs_candidates: List[int] = []
    for fs0 in range(len(combined_usage) - int(block_size) + 1):
        if np.all(combined_usage[fs0:fs0 + int(block_size)] == 0):
            fs_candidates.append(int(fs0))
    return fs_candidates


def _fs_block_is_free(
    link_FS: np.ndarray,
    used_links: List[int],
    fs0: int,
    block_size: int,
) -> bool:
    """检查给定 hub base FS block 是否在 used_links 上全空闲。"""
    fs1 = int(fs0) + int(block_size)
    if fs0 < 0 or fs1 > link_FS.shape[1]:
        return False
    if len(used_links) == 0:
        return True
    return bool(np.all(link_FS[used_links, int(fs0):fs1] == 0))


def _infer_hub_fs0_candidates_from_fixed_leaf(
    dest: int,
    hub_type: int,
    fixed_leaf: int | None,
    fixed_leaf_sc_range: Tuple[int, int] | None,
    node_P2MP: np.ndarray,
) -> List[int]:
    """根据固定 leaf 的 SC 区间直接反推出可行的 hub base FS。"""
    if fixed_leaf is None or fixed_leaf_sc_range is None:
        return []

    leaf_p = int(fixed_leaf)
    if leaf_p < 0 or leaf_p >= node_P2MP.shape[1]:
        return []
    if int(node_P2MP[dest][leaf_p][2]) == 0:
        return []

    leaf_type = int(node_P2MP[dest][leaf_p][3])
    leaf_base = int(node_P2MP[dest][leaf_p][5])
    if leaf_base < 0:
        return []

    leaf_sc_s = int(fixed_leaf_sc_range[0])
    leaf_sc_e = int(fixed_leaf_sc_range[1])
    fs_s_glo = int(leaf_base) + int(sc_fs(int(leaf_type), int(leaf_sc_s), 1))

    max_sc_idx = _TYPE_INFO[int(hub_type)][0] - 1
    hub_fs_candidates: List[int] = []
    for sc0_probe in range(max_sc_idx + 1):
        hub_sc_s = int(sc0_probe)
        hub_fs0 = int(fs_s_glo) - int(sc_fs(int(hub_type), int(hub_sc_s), 1))
        if hub_fs0 < 0:
            continue
        if int(hub_fs0) not in hub_fs_candidates:
            hub_fs_candidates.append(int(hub_fs0))
    return hub_fs_candidates


def _infer_hw_modu_from_cap(cap: float) -> int:
    """根据容量推断硬件调制类型。"""
    return 1 if cap >= 20 else 2


def _build_flow_metadata_map(
    affected_flow: List[Any],
    flow_acc_DP: np.ndarray,
    P2MP_SC_DP: np.ndarray,
) -> Dict[int, Dict[str, Any]]:
    """从 DP 状态提取每条流的元信息便于快速查询。"""
    meta: Dict[int, Dict[str, Any]] = {}
    for f in affected_flow:
        fid, src, dst = int(f[0]), int(f[1]), int(f[2])
        meta[fid] = extract_old_flow_info(fid, src, dst, flow_acc_DP, P2MP_SC_DP)
    return meta


def _assign_one_hop_with_initialnet(
    flow: Any,
    a: int,
    b: int,
    phy_path0: List[int],
    phy_dist: float,
    link_index: np.ndarray,
    strict_s1: bool,
    meta: Dict[str, Any],
    new_flow_acc: np.ndarray,
    new_node_flow: np.ndarray,
    new_node_P2MP: np.ndarray,
    new_P2MP_SC: np.ndarray,
    new_link_FS: np.ndarray,
    new_P2MP_FS: np.ndarray,
    new_flow_path: np.ndarray,
) -> bool:
    """按 Initial_net 的约束为单个逻辑跳分配资源。"""
    # 流程概览：
    # 1) 校验物理路径，提取带宽/原始流ID等关键参数
    # 2) 基于物理路径计算链路占用与 SC 容量，并按策略1固定端口/时隙（若开启 strict_s1）
    # 3) 先尝试已启用的 hub 端口进行 SC 规划；成功则提交资源并返回
    # 4) 若已启用端口均失败，再尝试空闲端口：分配 FS block → SC 规划 → 成功提交，否则回滚端口状态
    # 这个函数它负责回答一件事：“如果现在要把这一跳放到网络里，按当前策略和当前剩余资源，能不能放进去？”
    # 同一条流的各个 hop，带宽需求通常还是这条流的带宽
    # 但每个 hop 的物理路、端口、FS、SC、可复用性、是否受原配置约束，都是可能不同的
    if not phy_path0 or len(phy_path0) < 2:
        return False

    # 基本流参数
    fid = int(flow[0])
    band = float(flow[3])
    orig_fid = fid

    # 物理链路索引（0-based）
    used_links: List[int] = []
    for u, v in zip(phy_path0[:-1], phy_path0[1:]):
        lk = int(link_index[u][v])
        if lk <= 0:
            return False
        used_links.append(lk - 1)

    # 物理路径节点序列（1-based）与容量测算
    phys_nodes_1b = [int(x) + 1 for x in phy_path0]
    sc_cap_raw, _ = modu_format_Al(phy_dist, band)
    sc_cap = float(sc_effective_cap(sc_cap_raw))

    # 策略1：端点硬件/端口固定
    # 这里是为了找sc_cap
    # 如果当前 (a, b) 是中间 hop，则不会固定 hub/leaf，端口可按当前资源自由选择。
    fixed_hub = None
    fixed_leaf = None
    if strict_s1:
        if a == int(flow[1]) and int(meta.get("hub_idx", -1)) != -1:
            fixed_hub = int(meta["hub_idx"])
        if b == int(flow[2]) and int(meta.get("leaf_idx", -1)) != -1:
            fixed_leaf = int(meta["leaf_idx"])
        if fixed_hub is not None and fixed_leaf is not None:
            hub_cap = float(meta.get("hub_cap", 0.0)) if meta is not None else 0.0
            leaf_cap = float(meta.get("leaf_cap", 0.0)) if meta is not None else 0.0
            hub_eff = float(sc_effective_cap(hub_cap)) if hub_cap > 0 else 0.0
            leaf_eff = float(sc_effective_cap(leaf_cap)) if leaf_cap > 0 else 0.0
            if hub_eff <= 0 or leaf_eff <= 0 or abs(hub_eff - leaf_eff) > 1e-9:
                return False
            sc_cap = hub_eff
        elif fixed_hub is not None:
            old_cap = float(meta.get("hub_cap", 0.0)) if meta is not None else 0.0
            if old_cap > 0:
                sc_cap = float(sc_effective_cap(old_cap))
        elif fixed_leaf is not None:
            old_cap = float(meta.get("leaf_cap", 0.0)) if meta is not None else 0.0
            if old_cap > 0:
                sc_cap = float(sc_effective_cap(old_cap))
                
    # 这里是找hub_usage/leaf_usage
    fixed_hub_sc_range = None
    fixed_leaf_sc_range = None
    fixed_hub_usage = None
    fixed_leaf_usage = None
    if strict_s1:
        if fixed_hub is not None:
            hub_range = meta.get("hub_sc_range")
            if hub_range is not None:
                fixed_hub_sc_range = (int(hub_range[0]), int(hub_range[1]))
            hub_usage = meta.get("hub_sc_usage")
            if hub_usage is not None:
                fixed_hub_usage = {int(k): float(v) for k, v in hub_usage.items()}
        if fixed_leaf is not None:
            leaf_range = meta.get("leaf_sc_range")
            if leaf_range is not None:
                fixed_leaf_sc_range = (int(leaf_range[0]), int(leaf_range[1]))
            leaf_usage = meta.get("leaf_sc_usage")
            if leaf_usage is not None:
                fixed_leaf_usage = {int(k): float(v) for k, v in leaf_usage.items()}

    _p2mp_total = new_node_P2MP.shape[1]
    hub_candidates = [fixed_hub] if fixed_hub is not None else list(range(_p2mp_total))

    # 先尝试已启用的 hub 端口
    for hub_p in hub_candidates:
        if hub_p is None:
            continue
        if int(new_node_P2MP[a][hub_p][2]) == 1:
            used_bw = sum(float(x[3]) for x in new_node_flow[a][hub_p][0])
            if used_bw + band > 400 + 1e-9:
                continue
            hub_type = int(new_node_P2MP[a][hub_p][3])
            hub_fs0 = int(new_node_P2MP[a][hub_p][5])
            if hub_fs0 < 0:
                continue
            flow_ids_on_hub = new_node_flow[a][hub_p][0]
            plan = _plan_sc_allocation(
                a, hub_p, hub_type, hub_fs0, b,
                phys_nodes_1b, band, sc_cap, flow_ids_on_hub, new_flow_acc, new_P2MP_SC,
                new_node_P2MP, new_node_flow, new_link_FS, new_P2MP_FS, used_links, new_flow_path,
                fixed_hub_sc_range, fixed_hub_usage, fixed_leaf, fixed_leaf_sc_range, fixed_leaf_usage
            )
            if plan is None:
                continue
            sc_s, sc_e, usage, fs_s_glo, fs_e_glo, leaf_p, leaf_sc_s, leaf_sc_e, leaf_usage = plan
            if fixed_leaf is not None and int(leaf_p) != int(fixed_leaf):
                continue
            _commit_plan(
                orig_fid, a, b, band, sc_cap, hub_p, hub_type, leaf_p,
                sc_s, sc_e, fs_s_glo, fs_e_glo, leaf_sc_s, leaf_sc_e,
                usage, leaf_usage, phys_nodes_1b, used_links,
                new_flow_acc, new_node_flow, new_node_P2MP, new_P2MP_SC, new_link_FS, new_P2MP_FS, new_flow_path
            )
            return True

    # 再尝试空闲端口，先分配 FS block 再走同样的规划
    for hub_p in hub_candidates:
        if hub_p is None:
            continue
        if int(new_node_P2MP[a][hub_p][2]) == 0:
            hub_type = int(new_node_P2MP[a][hub_p][3])
            block_size = _TYPE_INFO[int(hub_type)][1]
            hub_fs_candidates = _infer_hub_fs0_candidates_from_fixed_leaf(
                dest=b,
                hub_type=hub_type,
                fixed_leaf=fixed_leaf,
                fixed_leaf_sc_range=fixed_leaf_sc_range,
                node_P2MP=new_node_P2MP,
            )
            if not hub_fs_candidates:
                hub_fs_candidates = _find_free_fs_block_candidates(new_link_FS, used_links, block_size)
            if not hub_fs_candidates:
                continue

            for hub_fs0 in hub_fs_candidates:
                new_node_P2MP[a][hub_p][2] = 1
                new_node_P2MP[a][hub_p][5] = int(hub_fs0)
                new_node_flow[a][hub_p][0].clear()
                _clear_p2mp_sc(new_P2MP_SC, a, hub_p)

                flow_ids_on_hub = new_node_flow[a][hub_p][0]
                plan = _plan_sc_allocation(
                    a, hub_p, hub_type, int(hub_fs0), b,
                    phys_nodes_1b, band, sc_cap, flow_ids_on_hub, new_flow_acc, new_P2MP_SC,
                    new_node_P2MP, new_node_flow, new_link_FS, new_P2MP_FS, used_links, new_flow_path,
                    fixed_hub_sc_range, fixed_hub_usage, fixed_leaf, fixed_leaf_sc_range, fixed_leaf_usage
                )
                if plan is None:
                    new_node_P2MP[a][hub_p][2] = 0
                    new_node_P2MP[a][hub_p][5] = -1
                    new_node_flow[a][hub_p][0].clear()
                    _clear_p2mp_sc(new_P2MP_SC, a, hub_p)
                    continue
                sc_s, sc_e, usage, fs_s_glo, fs_e_glo, leaf_p, leaf_sc_s, leaf_sc_e, leaf_usage = plan
                if fixed_leaf is not None and int(leaf_p) != int(fixed_leaf):
                    new_node_P2MP[a][hub_p][2] = 0
                    new_node_P2MP[a][hub_p][5] = -1
                    new_node_flow[a][hub_p][0].clear()
                    _clear_p2mp_sc(new_P2MP_SC, a, hub_p)
                    continue
                _commit_plan(
                    orig_fid, a, b, band, sc_cap, hub_p, hub_type, leaf_p,
                    sc_s, sc_e, fs_s_glo, fs_e_glo, leaf_sc_s, leaf_sc_e,
                    usage, leaf_usage, phys_nodes_1b, used_links,
                    new_flow_acc, new_node_flow, new_node_P2MP, new_P2MP_SC, new_link_FS, new_P2MP_FS, new_flow_path
                )
                return True

    return False


def _append_restored_subflow(
    new_flow_acc: np.ndarray,
    new_flow_path: np.ndarray,
    orig_fid: int,
    a: int,
    b: int,
    band: float,
    sc_cap: float,
    sc_num: int,
) -> int:
    """为恢复得到的新 hop 追加一条子流记录，并返回新的 subflow_id。"""
    sub_id = int(new_flow_acc.shape[0])

    new_flow_acc.resize((sub_id + 1, new_flow_acc.shape[1]), refcheck=False)
    new_flow_acc[sub_id] = [
        int(sub_id), int(a), int(b), float(band), float(sc_cap), int(sc_num), int(orig_fid),
        -1, -1, -1, -1, -1, -1, -1, -1,
    ]

    new_flow_path.resize((sub_id + 1, new_flow_path.shape[1]), refcheck=False)
    new_flow_path[sub_id][0] = []
    new_flow_path[sub_id][1] = []
    new_flow_path[sub_id][2] = []
    new_flow_path[sub_id][3] = []
    return sub_id


def _commit_plan(
    orig_fid: int,
    a: int,
    b: int,
    band: float,
    sc_cap: float,
    hub_p: int,
    hub_type: int,
    leaf_p: int,
    sc_s: int,
    sc_e: int,
    fs_s_glo: int,
    fs_e_glo: int,
    leaf_sc_s: int,
    leaf_sc_e: int,
    usage: Dict[int, float],
    leaf_usage: Dict[int, float],
    phys_nodes_1b: List[int],
    used_links: List[int],
    new_flow_acc: np.ndarray,
    new_node_flow: np.ndarray,
    new_node_P2MP: np.ndarray,
    new_P2MP_SC: np.ndarray,
    new_link_FS: np.ndarray,
    new_P2MP_FS: np.ndarray,
    new_flow_path: np.ndarray,
) -> None:
    """将规划结果一次性写回所有资源状态表。"""
    sc_num = int(sc_e) - int(sc_s) + 1
    sub_id = _append_restored_subflow(new_flow_acc, new_flow_path, orig_fid, a, b, band, sc_cap, sc_num)

    subflow_info = [int(sub_id), int(a), int(b), float(band), float(sc_cap), int(sc_num), int(orig_fid)]
    new_node_flow[a][hub_p][0].append(subflow_info)
    new_node_flow[b][leaf_p][0].append(subflow_info)
    new_node_P2MP[a][hub_p][4] = float(new_node_P2MP[a][hub_p][4]) - float(band)
    new_node_P2MP[b][leaf_p][4] = float(new_node_P2MP[b][leaf_p][4]) - float(band)
    if int(new_node_P2MP[a][hub_p][2]) == 0:
        new_node_P2MP[a][hub_p][2] = 1
    if int(new_node_P2MP[b][leaf_p][2]) == 0:
        new_node_P2MP[b][leaf_p][2] = 1

    new_flow_acc[sub_id][0] = int(sub_id)
    new_flow_acc[sub_id][1] = int(a)
    new_flow_acc[sub_id][2] = int(b)
    new_flow_acc[sub_id][3] = float(band)
    new_flow_acc[sub_id][4] = float(sc_cap)
    new_flow_acc[sub_id][5] = int(sc_num)
    new_flow_acc[sub_id][6] = int(orig_fid)
    new_flow_acc[sub_id][7] = int(hub_p)
    new_flow_acc[sub_id][8] = int(leaf_p)
    new_flow_acc[sub_id][9] = int(sc_s)
    new_flow_acc[sub_id][10] = int(sc_e)
    new_flow_acc[sub_id][11] = int(fs_s_glo)
    new_flow_acc[sub_id][12] = int(fs_e_glo)
    new_flow_acc[sub_id][13] = int(leaf_sc_s)
    new_flow_acc[sub_id][14] = int(leaf_sc_e)

    for sc, used_amt in usage.items():
        new_P2MP_SC[a][hub_p][int(sc)][1].append([int(sub_id), float(used_amt), int(orig_fid)])
        new_P2MP_SC[a][hub_p][int(sc)][0].append(float(sc_cap))
        new_P2MP_SC[a][hub_p][int(sc)][2].append(list(phys_nodes_1b))
        new_P2MP_SC[a][hub_p][int(sc)][4].append(int(b))
        new_P2MP_SC[a][hub_p][int(sc)][5].append(int(a))
        new_P2MP_SC[a][hub_p][int(sc)][6].append(int(hub_p))
        new_P2MP_SC[a][hub_p][int(sc)][7].append(int(b))
        new_P2MP_SC[a][hub_p][int(sc)][8].append(int(leaf_p))
        _apply_p2mp_fs_usage(
            new_P2MP_FS, a, hub_p, int(hub_type), int(sc),
            int(sub_id), float(used_amt), list(phys_nodes_1b), int(b), int(orig_fid),
            int(a), int(hub_p), int(b), int(leaf_p),
        )

    new_flow_path[sub_id][0] = [int(sub_id)]
    new_flow_path[sub_id][1] = list(used_links)
    new_flow_path[sub_id][2] = list(phys_nodes_1b)
    new_flow_path[sub_id][3] = [int(orig_fid)]

    _apply_leaf_usage(
        new_P2MP_SC, int(b), int(leaf_p),
        int(leaf_sc_s), int(leaf_sc_e),
        leaf_usage, float(sc_cap),
        int(sub_id), list(phys_nodes_1b), int(b), int(orig_fid),
        int(a), int(hub_p), int(b), int(leaf_p),
    )
    leaf_type = int(new_node_P2MP[b][leaf_p][3])
    for sc, used_amt in leaf_usage.items():
        _apply_p2mp_fs_usage(
            new_P2MP_FS, b, leaf_p, int(leaf_type), int(sc),
            int(sub_id), float(used_amt), list(phys_nodes_1b), int(b), int(orig_fid),
            int(a), int(hub_p), int(b), int(leaf_p),
        )

    for l in set(used_links):
        new_link_FS[int(l)][int(fs_s_glo):int(fs_e_glo) + 1] += 1


def _attempt_restore_flow_multihop(
    flow: Any,
    break_node: int,
    v_adj: np.ndarray,
    v_phy: Dict[Tuple[int, int], Dict[str, Any]],
    K_LOGICAL_PATHS: int,
    flow_metadata_map: Dict[int, Dict[str, Any]],
    link_index: np.ndarray,
    strict_s1: bool,
    state: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[bool, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """多跳算法：枚举逻辑路径并逐跳提交资源以尝试恢复该流。"""
    # 流程概览：
    # 1) 提取源宿节点，计算 K 条逻辑候选路径（1-based）
    # 2) 为每条逻辑路径创建独立快照，逐跳尝试分配资源
    # 3) 任一逻辑路径全跳成功则返回新状态，否则回退到原状态
    src = int(flow[1])
    dst = int(flow[2])

    try:
        # 生成逻辑层 k 短路候选路径（节点编号为 1-based）
        logical_paths = k_shortest_path(v_adj, src + 1, dst + 1, K_LOGICAL_PATHS)
    except Exception:
        # 路径搜索异常时视为无可用逻辑路径
        logical_paths = []

    print(f"[AG_S1F][multihop] flow={int(flow[0])} logical_paths={len(logical_paths)}")
    for idx, (p_nodes, p_weight) in enumerate(logical_paths, start=1):
        print(f"  path{idx}: nodes={list(p_nodes)}, weight={float(p_weight)}")

    # 保存原始状态快照，保证路径失败不污染外部状态
    snap = tuple(copy.deepcopy(x) for x in state)
    # 获取该流的元数据，用于跳级资源分配
    meta = flow_metadata_map.get(int(flow[0]), {})

    # 枚举每条逻辑路径，逐跳尝试分配资源
    for lp_nodes_1b, _ in logical_paths:
        # 逻辑路径有效性检查（例如绕开断点）
        is_valid = logical_path_is_valid(lp_nodes_1b, break_node)
        if not is_valid:
            continue
        # 逻辑路径节点改为 0-based 以匹配内部索引
        lp0 = [x - 1 for x in lp_nodes_1b]

        # 对当前逻辑路径建立工作副本，失败可随时丢弃
        work_flow_acc, work_node_flow, work_node_P2MP, work_P2MP_SC, work_link_FS, work_P2MP_FS, work_flow_path = copy.deepcopy(snap)

        ok_path = True
        # 逐跳处理逻辑路径中的相邻节点对
        for a, b in zip(lp0[:-1], lp0[1:]):
            # 若逻辑跳无法映射到物理候选，当前路径直接失败
            if (a, b) not in v_phy:
                ok_path = False
                break

            # 取出该逻辑跳对应的物理路径与距离
            phy_cand = v_phy[(a, b)]
            phy_path0 = list(phy_cand["path"])
            phy_dist = float(phy_cand["dist"])

            # 在该物理跳上尝试分配资源（成功则更新工作副本）
            ok_hop = _assign_one_hop_with_initialnet(
                flow=flow,
                a=a,
                b=b,
                phy_path0=phy_path0,
                phy_dist=phy_dist,
                link_index=link_index,
                strict_s1=strict_s1,
                meta=meta,
                new_flow_acc=work_flow_acc,
                new_node_flow=work_node_flow,
                new_node_P2MP=work_node_P2MP,
                new_P2MP_SC=work_P2MP_SC,
                new_link_FS=work_link_FS,
                new_P2MP_FS=work_P2MP_FS,
                new_flow_path=work_flow_path,
            )
            # 单跳失败则该逻辑路径失败
            if not ok_hop:
                print(
                    f"[AG_S1F][multihop] flow={int(flow[0])} hop_failed="
                    f"({int(a)}->{int(b)}) logical_path={lp0} phy_path={phy_path0}"
                )
                ok_path = False
                break

        # 所有跳成功则返回该路径的工作状态
        if ok_path:
            return True, (work_flow_acc, work_node_flow, work_node_P2MP, work_P2MP_SC, work_link_FS, work_P2MP_FS, work_flow_path)

    # 所有候选路径失败，返回原始状态
    return False, state


def _attempt_restore_flow_singlehop(
    flow: Any,
    phy_pool: Dict[Tuple[int, int], List[Dict[str, Any]]],
    flow_metadata_map: Dict[int, Dict[str, Any]],
    link_index: np.ndarray,
    strict_s1: bool,
    state: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[bool, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """单跳算法：直接对 src->dst 的物理候选路径逐条尝试资源分配。"""
    src = int(flow[1])
    dst = int(flow[2])
    meta = flow_metadata_map.get(int(flow[0]), {})
    snap = tuple(copy.deepcopy(x) for x in state)

    direct_candidates = phy_pool.get((src, dst), [])
    for phy_cand in direct_candidates:
        work_flow_acc, work_node_flow, work_node_P2MP, work_P2MP_SC, work_link_FS, work_P2MP_FS, work_flow_path = copy.deepcopy(snap)

        ok_hop = _assign_one_hop_with_initialnet(
            flow=flow,
            a=src,
            b=dst,
            phy_path0=list(phy_cand["path"]),
            phy_dist=float(phy_cand["dist"]),
            link_index=link_index,
            strict_s1=strict_s1,
            meta=meta,
            new_flow_acc=work_flow_acc,
            new_node_flow=work_node_flow,
            new_node_P2MP=work_node_P2MP,
            new_P2MP_SC=work_P2MP_SC,
            new_link_FS=work_link_FS,
            new_P2MP_FS=work_P2MP_FS,
            new_flow_path=work_flow_path,
        )
        if ok_hop:
            return True, (work_flow_acc, work_node_flow, work_node_P2MP, work_P2MP_SC, work_link_FS, work_P2MP_FS, work_flow_path)

    return False, state


def _mode_has_s1_candidate(
    restore_mode: str,
    flow: Any,
    break_node: int,
    v_adj_s1: np.ndarray,
    v_phy_s1: Dict[Tuple[int, int], Dict[str, Any]],
) -> bool:
    """按所选算法判断该流是否具备 S1 尝试资格。"""
    src = int(flow[1])
    dst = int(flow[2])

    if restore_mode == "singlehop":
        return (src, dst) in v_phy_s1

    if restore_mode == "multihop":
        try:
            paths = k_shortest_path(v_adj_s1, src + 1, dst + 1, 1)
            for p_nodes, _ in paths:
                is_valid = logical_path_is_valid(p_nodes, break_node)
                if is_valid:
                    return True
        except Exception:
            return False
        return False

    raise ValueError(f"Unknown restore_mode={restore_mode}")


def _attempt_restore_flow_by_mode(
    restore_mode: str,
    flow: Any,
    break_node: int,
    v_adj: np.ndarray,
    v_phy: Dict[Tuple[int, int], Dict[str, Any]],
    phy_pool: Dict[Tuple[int, int], List[Dict[str, Any]]],
    K_LOGICAL_PATHS: int,
    flow_metadata_map: Dict[int, Dict[str, Any]],
    link_index: np.ndarray,
    strict_s1: bool,
    state: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[bool, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """按指定模式选择单跳算法或多跳算法。"""
    if restore_mode == "singlehop":
        return _attempt_restore_flow_singlehop(
            flow=flow,
            phy_pool=phy_pool,
            flow_metadata_map=flow_metadata_map,
            link_index=link_index,
            strict_s1=strict_s1,
            state=state,
        )

    if restore_mode == "multihop":
        return _attempt_restore_flow_multihop(
            flow=flow,
            break_node=break_node,
            v_adj=v_adj,
            v_phy=v_phy,
            K_LOGICAL_PATHS=K_LOGICAL_PATHS,
            flow_metadata_map=flow_metadata_map,
            link_index=link_index,
            strict_s1=strict_s1,
            state=state,
        )

    raise ValueError(f"Unknown restore_mode={restore_mode}")


def Heuristic_algorithm(
    affected_flow: List[Any],
    new_flow_acc: np.ndarray,
    new_node_flow: np.ndarray,
    new_node_P2MP: np.ndarray,
    break_node: int,
    Tbox_num: int,
    Tbox_P2MP: int,
    new_P2MP_SC: np.ndarray,
    new_link_FS: np.ndarray,
    new_P2MP_FS: np.ndarray,
    node_flow_base: np.ndarray,
    node_P2MP_base: np.ndarray,
    flow_acc_base: np.ndarray,
    link_FS_base: np.ndarray,
    P2MP_SC_base: np.ndarray,
    P2MP_FS_base: np.ndarray,
    flow_path_base: np.ndarray,
    new_flow_path: np.ndarray,
    restore_mode: str = "multihop",
):
    """策略1优先的恢复流程，并复用 Initial_net 资源分配约束。

    返回值依次为：
    new_node_flow, new_node_P2MP, new_flow_acc,
    new_link_FS, new_P2MP_SC, new_P2MP_FS, new_flow_path,
    failed_orig, tier1_restored_ids, tier2_restored_ids
    """
    if restore_mode not in ("singlehop", "multihop"):
        raise ValueError(f"Unknown restore_mode={restore_mode}")

    # 读取拓扑与链路索引，所有路径搜索与资源分配都基于该拓扑
    topo_num, _, topo_dis, link_num, link_index = tp.topology(1)

    # 算法超参数：物理候选数、逻辑候选数、重构惩罚与跳数惩罚
    K_PHY_CANDIDATES = 3
    K_LOGICAL_PATHS = 5
    RECONFIG_PENALTY = 5000
    HOP_PENALTY = 200

    # 预生成每对节点的物理候选路径池（0-based路径节点序列）
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

    # 从 DP 状态提取原有流的历史资源信息
    flow_metadata_map = _build_flow_metadata_map(affected_flow, flow_acc_base, P2MP_SC_base)

    # 分层队列：Tier1 代表可走策略1（硬件/路径兼容），Tier2 代表需要更自由的策略2
    tier1_flows: List[Any] = []
    tier2_flows: List[Any] = []
    flow_vmap_s1: Dict[int, Tuple[np.ndarray, Dict[Tuple[int, int], Dict[str, Any]]]] = {}
    flow_vmap_s2: Dict[int, Tuple[np.ndarray, Dict[Tuple[int, int], Dict[str, Any]]]] = {}

    # 为每条受影响的流构建虚拟拓扑，并判断是否具备策略1资格
    for f in affected_flow:
        fid, src, dst, band = int(f[0]), int(f[1]), int(f[2]), float(f[3])
        meta = flow_metadata_map[fid]
        # 根据原流端口容量推断硬件调制类型
        src_modu = _infer_hw_modu_from_cap(float(meta.get("hub_cap", 0.0)))
        dst_modu = _infer_hw_modu_from_cap(float(meta.get("leaf_cap", 0.0)))

        # 策略1：强制保持原硬件/格式约束
        v_adj_s1, v_phy_s1 = build_virtual_topology(
            src, dst, topo_num, phy_pool, break_node, new_node_P2MP, band,
            True, src_modu, dst_modu, RECONFIG_PENALTY, HOP_PENALTY
        )
        flow_vmap_s1[fid] = (v_adj_s1, v_phy_s1)

        # 按当前算法模式判断该流是否值得进入 S1 队列。
        ok_s1 = _mode_has_s1_candidate(restore_mode, f, break_node, v_adj_s1, v_phy_s1)

        if ok_s1:
            tier1_flows.append(f)
        else:
            tier2_flows.append(f)

        # 策略2：放宽约束，允许硬件/格式重构
        v_adj_s2, v_phy_s2 = build_virtual_topology(
            src, dst, topo_num, phy_pool, break_node, new_node_P2MP, band,
            False, src_modu, dst_modu, RECONFIG_PENALTY, HOP_PENALTY
        )
        flow_vmap_s2[fid] = (v_adj_s2, v_phy_s2)

    # 先对 Tier1 的端口/SC 进行占位，避免资源被后续流抢占
    manage_s1_reservation_by_copy(
        tier1_flows, flow_metadata_map,
        flow_acc_base, node_P2MP_base, P2MP_SC_base, P2MP_FS_base, flow_path_base,
        new_node_flow, new_node_P2MP, new_P2MP_SC, new_P2MP_FS, new_link_FS,
        action="reserve"
    )

    tier1_restored, tier1_failed = [], []

    # 先尝试策略1恢复：逐流回滚本流占位，再按指定模式执行恢复。
    for f in tier1_flows:
        fid = int(f[0])

        # 对当前流撤销占位，避免影响真实分配paths = k_shortest_path(v_adj_s1, src + 1, dst + 1, 1)
        manage_s1_reservation_by_copy(
            [f], flow_metadata_map,
            flow_acc_base, node_P2MP_base, P2MP_SC_base, P2MP_FS_base, flow_path_base,
            new_node_flow, new_node_P2MP, new_P2MP_SC, new_P2MP_FS, new_link_FS,
            action="rollback"
        )

        v_adj, v_phy = flow_vmap_s1[fid]
        # 尝试在策略1约束下恢复。
        restored, new_state = _attempt_restore_flow_by_mode(
            restore_mode=restore_mode,
            flow=f,
            break_node=break_node,
            v_adj=v_adj,
            v_phy=v_phy,
            phy_pool=phy_pool,
            K_LOGICAL_PATHS=K_LOGICAL_PATHS,
            flow_metadata_map=flow_metadata_map,
            link_index=link_index,
            strict_s1=True,
            state=(new_flow_acc, new_node_flow, new_node_P2MP, new_P2MP_SC, new_link_FS, new_P2MP_FS, new_flow_path),
        )
        if restored:
            # 使用成功路径的状态覆盖回主状态
            new_flow_acc, new_node_flow, new_node_P2MP, new_P2MP_SC, new_link_FS, new_P2MP_FS, new_flow_path = new_state

        if restored:
            tier1_restored.append(f)
        else:
            # 策略1失败的流进入 Tier2 尝试更自由的恢复
            tier1_failed.append(f)
            tier2_flows.append(f)

    tier2_restored, tier2_failed = [], []

    # 再尝试策略2恢复：允许重构，并按指定模式执行恢复。
    for f in tier2_flows:
        fid = int(f[0])

        v_adj, v_phy = flow_vmap_s2[fid]
        restored, new_state = _attempt_restore_flow_by_mode(
            restore_mode=restore_mode,
            flow=f,
            break_node=break_node,
            v_adj=v_adj,
            v_phy=v_phy,
            phy_pool=phy_pool,
            K_LOGICAL_PATHS=K_LOGICAL_PATHS,
            flow_metadata_map=flow_metadata_map,
            link_index=link_index,
            strict_s1=False,
            state=(new_flow_acc, new_node_flow, new_node_P2MP, new_P2MP_SC, new_link_FS, new_P2MP_FS, new_flow_path),
        )
        if restored:
            new_flow_acc, new_node_flow, new_node_P2MP, new_P2MP_SC, new_link_FS, new_P2MP_FS, new_flow_path = new_state

        if restored:
            tier2_restored.append(f)
        else:
            tier2_failed.append(f)

    failed_orig = [int(f[0]) for f in tier2_failed]
    failed_orig = list(dict.fromkeys(failed_orig))
    tier1_restored_ids = [int(f[0]) for f in tier1_restored]
    tier1_restored_ids = list(dict.fromkeys(tier1_restored_ids))
    tier2_restored_ids = [int(f[0]) for f in tier2_restored]
    tier2_restored_ids = list(dict.fromkeys(tier2_restored_ids))
    return (
        new_node_flow,
        new_node_P2MP,
        new_flow_acc,
        new_link_FS,
        new_P2MP_SC,
        new_P2MP_FS,
        new_flow_path,
        failed_orig,
        tier1_restored_ids,
        tier2_restored_ids,
    )
