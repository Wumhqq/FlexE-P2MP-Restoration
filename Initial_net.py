#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
顺序式（逐流）FlexE-P2MP 初始网络分配/恢复复用。

与 DP 版本保持一致的关键约束：
- Hub 与 Leaf 必须使用同一段绝对 FS（同一 hub_fs0），即 leaf 的 FS_start 与 hub 完全一致；
- 返回的数据结构与 FlexE_P2MP_DP 一致：
  node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, flow_path

与 FlexE_P2MP_DP 的区别：
- 不使用背包/DP 打包，按流逐个分配（可选排序以贴近 DP 初始化行为）；
- 设计为可复用：核心分配器可在已有状态上工作（如中断后的恢复阶段）。
"""

import math
import numpy as np

from k_shortest_path import k_shortest_path
from SC_FS import sc_fs
from Path_Link import get_links_from_path
from modu_format_Al import modu_format_Al

# 为保持项目约定而保留（此处不直接使用）
TS_UNIT = 5

# P2MP 类型（与 DP 初始化一致）：
# type_id -> (可用SC数量, 对应连续FS块大小)
_TYPE_INFO = {
    1: (1, 1),
    2: (4, 2),
    3: (16, 6),
}

def sc_effective_cap(sc_cap: float) -> int:
    """把 SC 容量映射成可用的整数 TS 容量（不允许 2.5 TS 这种分配）。
    例如：12.5Gbps 只能承载 floor(12.5/5)=2 个 TS => 10Gbps。
    """
    return int(math.floor(float(sc_cap) / TS_UNIT) * TS_UNIT)

def sc_num_from_bw_cap(bw: float, sc_cap: float) -> int:
    eff = sc_effective_cap(sc_cap)
    if eff <= 0:
        raise ValueError(f"Invalid SC_cap={sc_cap}")
    return int(math.ceil(float(bw) / eff))
 
def _clear_p2mp_sc(P2MP_SC: np.ndarray, node: int, p: int) -> None:
    """清空一个 (node, p2mp) 的 16×N SC 记录。"""
    for s in range(16):
        for c in range(P2MP_SC.shape[3]):
            P2MP_SC[node][p][s][c] = []



def _find_free_fs_block(link_FS: np.ndarray, used_links: list[int], block_size: int) -> int | None:
    """在所有 used_links 上寻找首个可用的连续 FS 块作为 hub_fs0。"""
    if len(used_links) == 0:
        combined_usage = link_FS.sum(axis=0) * 0
    else:
        combined_usage = link_FS[used_links].sum(axis=0)

    for i0 in range(len(combined_usage) - block_size + 1):
        if np.all(combined_usage[i0:i0 + block_size] == 0):
            return int(i0)
    return None


def _sc_used_bw_for_path(src: int, hub_p: int, sc: int, phys_nodes: list, P2MP_SC: np.ndarray) -> float:
    """统计该 hub 的指定 SC 在给定路径上的已用带宽。"""
    used = 0.0
    try:
        flows = P2MP_SC[src][hub_p][sc][1]
        paths = P2MP_SC[src][hub_p][sc][2]
    except Exception:
        return 0.0
    if not flows or not paths:
        return 0.0
    for entry, p in zip(flows, paths):
        if p == phys_nodes:
            used += float(entry[1])
    return used


def _sc_can_use_on_hub(P2MP_SC: np.ndarray, src: int, hub_p: int, sc: int,
                       phys_nodes: list, dest: int) -> bool:
    """检查该 SC 是否允许用于当前路径与目的节点。"""
    try:
        paths = P2MP_SC[src][hub_p][sc][2]
        dsts = P2MP_SC[src][hub_p][sc][4]
    except Exception:
        return True
    if not paths:
        return True
    for p, d in zip(paths, dsts):
        if p == phys_nodes and int(d) == int(dest):
            return True
    return False


def _fs_path_ok_on_p2mp(P2MP_FS: np.ndarray, u: int, p: int, fs_rel: int, phys_nodes: list) -> bool:
    """检查该 P2MP 相对 FS 位置是否与给定路径一致或空闲。"""
    try:
        path_list = P2MP_FS[u][p][fs_rel][2]
    except Exception:
        return False
    if not path_list:
        return True
    for pp in path_list:
        if pp != phys_nodes:
            return False
    return True


def _sc_fs_path_ok(P2MP_FS: np.ndarray, u: int, p: int, p_type: int, sc_idx: int, phys_nodes: list) -> bool:
    """检查某个 SC 覆盖的所有 FS 是否都与给定路径一致。"""
    fs_s_rel = int(sc_fs(int(p_type), int(sc_idx), 1))
    fs_e_rel = int(sc_fs(int(p_type), int(sc_idx), 2))
    for fs_rel in range(int(fs_s_rel), int(fs_e_rel) + 1):
        if not _fs_path_ok_on_p2mp(P2MP_FS, u, p, int(fs_rel), phys_nodes):
            return False
    return True


def _apply_p2mp_fs_usage(P2MP_FS: np.ndarray, u: int, p: int, p_type: int, sc_idx: int,
                         fid: int, used_amt: float, phys_nodes: list, dest: int, flow_tag,
                         src_node: int, src_p: int, dst_node: int, dst_p: int) -> None:
    """把该流量在指定 SC 覆盖的 FS 位置写入 P2MP_FS 记录。"""
    fs_s_rel = int(sc_fs(int(p_type), int(sc_idx), 1))
    fs_e_rel = int(sc_fs(int(p_type), int(sc_idx), 2))
    for fs_rel in range(int(fs_s_rel), int(fs_e_rel) + 1):
        cell = P2MP_FS[u][p][int(fs_rel)]
        cell[1].append([int(fid), float(used_amt), flow_tag])
        cell[2].append(phys_nodes)
        cell[4].append(int(dest))
        cell[5].append(int(src_node))
        cell[6].append(int(src_p))
        cell[7].append(int(dst_node))
        cell[8].append(int(dst_p))


def _leaf_sc_segment_ok(P2MP_SC: np.ndarray, dest: int, leaf_p: int, sc_s: int, sc_e: int,
                        phys_nodes: list, des: int) -> bool:
    """检查 leaf 侧 SC 段是否与路径与目的节点一致。"""
    for sc in range(int(sc_s), int(sc_e) + 1):
        try:
            paths = P2MP_SC[dest][leaf_p][sc][2]
            dsts = P2MP_SC[dest][leaf_p][sc][4]
        except Exception:
            paths = []
            dsts = []
        if paths:
            if not any(p == phys_nodes for p in paths):
                return False
            if not dsts:
                return False
            for d in dsts:
                if int(d) != int(des):
                    return False
    return True


def _find_leaf_sc_segment_with_base(P2MP_SC: np.ndarray, P2MP_FS: np.ndarray,
                                    dest: int, leaf_p: int,
                                    leaf_type: int, base_fs: int,
                                    fs_s_glo: int, fs_e_glo: int,
                                    span_len: int,
                                    src_node: int, src_p: int,
                                    phys_nodes: list, des: int,
                                    usage: dict, sc_cap: float,
                                    require_fs_empty: bool = False) -> tuple[int, int, dict] | None:
    """在给定 base_fs 前提下寻找可用的 leaf SC 段。"""
    # base_fs 无效直接失败
    if int(base_fs) < 0:
        return None
    # 该型号允许的最大 SC 索引
    max_sc_idx = _TYPE_INFO[int(leaf_type)][0] - 1
    # hub 侧实际使用的 SC 列表（顺序与 usage 对应）
    hub_sc_list = sorted(int(x) for x in usage.keys())
    # 叶端 span 必须与 hub 侧 span 一致
    if len(hub_sc_list) != int(span_len):
        return None
    # 枚举 leaf 侧所有连续 SC 段
    for sc0 in range(0, max_sc_idx + 1):
        for sc1 in range(sc0, max_sc_idx + 1):
            # 仅考虑长度与 span_len 一致的区间
            if (sc1 - sc0 + 1) != int(span_len):
                continue
            fs_s_rel = sc_fs(int(leaf_type), int(sc0), 1)
            fs_e_rel = sc_fs(int(leaf_type), int(sc1), 2)
            # 绝对 FS 段必须与 hub 侧完全对齐
            if int(base_fs) + int(fs_s_rel) != int(fs_s_glo):
                continue
            if int(base_fs) + int(fs_e_rel) != int(fs_e_glo):
                continue
            # 叶端 FS 一致性：路径一致 + 发送端一致
            fs_ok = True
            for fs_rel in range(int(fs_s_rel), int(fs_e_rel) + 1):
                if not _fs_path_ok_on_p2mp(P2MP_FS, dest, leaf_p, int(fs_rel), phys_nodes):
                    fs_ok = False
                    break
                # 若该 FS 已被占用，则发送端节点与发送端 P2MP 必须一致
                try:
                    src_node_list = P2MP_FS[dest][leaf_p][int(fs_rel)][5]
                    src_p_list = P2MP_FS[dest][leaf_p][int(fs_rel)][6]
                except Exception:
                    src_node_list = []
                    src_p_list = []
                if src_node_list and src_p_list:
                    if len(src_node_list) != len(src_p_list):
                        fs_ok = False
                        break
                    for idx in range(len(src_node_list)):
                        try:
                            sn = int(src_node_list[idx])
                            sp = int(src_p_list[idx])
                        except Exception:
                            fs_ok = False
                            break
                        if sn != int(src_node) or sp != int(src_p):
                            fs_ok = False
                            break
                    if not fs_ok:
                        break
            if not fs_ok:
                continue
            if require_fs_empty:
                # 要求 FS 必须全空，禁止在已占用 FS 上复用
                fs_free = True
                for fs_rel in range(int(fs_s_rel), int(fs_e_rel) + 1):
                    try:
                        used_list = P2MP_FS[dest][leaf_p][int(fs_rel)][1]
                        path_list = P2MP_FS[dest][leaf_p][int(fs_rel)][2]
                    except Exception:
                        fs_free = False
                        break
                    if used_list or path_list:
                        fs_free = False
                        break
                if not fs_free:
                    continue
            # 该段 SC 的路径与目的节点必须一致
            if not _leaf_sc_segment_ok(P2MP_SC, dest, leaf_p, sc0, sc1, phys_nodes, des):
                continue
            
            # 校验叶端每个 SC 的剩余容量是否覆盖对应 usage
            ok_cap = True
            for lsc, hsc in zip(range(int(sc0), int(sc1) + 1), hub_sc_list):
                try:
                    flows = P2MP_SC[dest][leaf_p][int(lsc)][1]
                    paths = P2MP_SC[dest][leaf_p][int(lsc)][2]
                    dsts = P2MP_SC[dest][leaf_p][int(lsc)][4]
                except Exception:
                    flows = []
                    paths = []
                    dsts = []
                used = 0.0
                if flows and paths and dsts:
                    for entry, p, d in zip(flows, paths, dsts):
                        if p == phys_nodes and int(d) == int(des):
                            used += float(entry[1])
                spare = float(sc_cap) - float(used)
                # 叶端当前 SC 的剩余容量必须覆盖 hub 分配到该 SC 的使用量
                if spare + 1e-9 < float(usage[int(hsc)]):
                    ok_cap = False
                    break
            if not ok_cap:
                continue
            # 记录本段 leaf SC 的使用量映射（leaf_sc -> used_bw）
            leaf_usage = {int(lsc): float(usage[int(hsc)]) for lsc, hsc in zip(range(int(sc0), int(sc1) + 1), hub_sc_list)}
            return int(sc0), int(sc1), leaf_usage
    return None


def _infer_leaf_base_and_find_sc(P2MP_SC: np.ndarray, P2MP_FS: np.ndarray, dest: int, leaf_p: int,
                                 leaf_type: int, src_node: int, src_p: int,
                                 fs_s_glo: int, fs_e_glo: int,
                                 span_len: int,
                                 phys_nodes: list, des: int,
                                 usage: dict, sc_cap: float) -> tuple[int, int, int, dict] | None:
    """推断可行的 leaf base_fs 并返回对应的 SC 段。"""
    max_sc_idx = _TYPE_INFO[int(leaf_type)][0] - 1
    # 枚举 leaf 侧起始 SC，反推 base_fs 并验证对应 SC 段是否可用
    for sc0_probe in range(0, max_sc_idx + 1):
        rel_s = int(sc_fs(int(leaf_type), int(sc0_probe), 1))
        base_cand = int(fs_s_glo) - int(rel_s)
        if base_cand < 0:
            continue
        seg = _find_leaf_sc_segment_with_base(
            P2MP_SC, P2MP_FS, dest, leaf_p,
            leaf_type, base_cand,
            fs_s_glo, fs_e_glo,
            span_len,
            src_node, src_p,
            phys_nodes, des,
            usage, sc_cap,
        )
        if seg is None:
            continue
        return int(seg[0]), int(seg[1]), int(base_cand), seg[2]
    return None


def _apply_leaf_usage(P2MP_SC: np.ndarray, dest: int, leaf_p: int,
                      leaf_sc_s: int, leaf_sc_e: int,
                      leaf_usage: dict, sc_cap: float,
                      fid: int, phys_nodes: list, des: int, flow_tag,
                      src_node: int, src_p: int, dst_node: int, dst_p: int) -> bool:
    """把 leaf_usage 写入 leaf 侧的 SC 段记录中。"""
    leaf_sc_list = list(range(int(leaf_sc_s), int(leaf_sc_e) + 1))
    for lsc in leaf_sc_list:
        if int(lsc) not in leaf_usage:
            return False
    for lsc in leaf_sc_list:
        used_amt = leaf_usage[int(lsc)]
        P2MP_SC[dest][leaf_p][int(lsc)][1].append([int(fid), float(used_amt), flow_tag])
        P2MP_SC[dest][leaf_p][int(lsc)][0].append(float(sc_cap))
        P2MP_SC[dest][leaf_p][int(lsc)][2].append(phys_nodes)
        P2MP_SC[dest][leaf_p][int(lsc)][4].append(int(des))
        P2MP_SC[dest][leaf_p][int(lsc)][5].append(int(src_node))
        P2MP_SC[dest][leaf_p][int(lsc)][6].append(int(src_p))
        P2MP_SC[dest][leaf_p][int(lsc)][7].append(int(dst_node))
        P2MP_SC[dest][leaf_p][int(lsc)][8].append(int(dst_p))
    return True


def _alloc_new_leaf(node_P2MP: np.ndarray, node_flow: np.ndarray, P2MP_SC: np.ndarray,
                    P2MP_FS: np.ndarray, dest: int, src_node: int, src_p: int,
                    fs_s_glo: int, fs_e_glo: int, span_len: int,
                    phys_nodes: list, des: int,
                    usage: dict, sc_cap: float,
                    fixed_leaf: int | None = None) -> tuple[int | None, int | None, int | None, dict | None, bool]:
    """分配或复用 leaf 端口，并返回对应 SC 段与是否新建标记。"""
    # 叶端口总数
    _p2mp_total = node_P2MP.shape[1]
    # 如果指定了复用的 leaf，则优先使用该端口
    if fixed_leaf is not None:
        p = int(fixed_leaf)
        if p < 0 or p >= _p2mp_total:
            return None, None, None, None, False
        if int(node_P2MP[dest][p][2]) == 0:
            return None, None, None, None, False
        # 该端口已使用：在既有 base 上寻找可用 SC 段
        leaf_type = int(node_P2MP[dest][p][3])
        leaf_base = int(node_P2MP[dest][p][5])
        if leaf_base < 0:
            return None, None, None, None, False
        seg = _find_leaf_sc_segment_with_base(
            P2MP_SC, P2MP_FS, dest, p,
            leaf_type, leaf_base,
            fs_s_glo, fs_e_glo,
            span_len,
            src_node, src_p,
            phys_nodes, des,
            usage, sc_cap,
        )
        if seg is None:
            return None, None, None, None, False
        return int(p), int(seg[0]), int(seg[1]), seg[2], False

    # 1) 在已使用的 leaf 中找可复用端口（要求 FS 全空）
    for p in range(_p2mp_total):
        if int(node_P2MP[dest][p][2]) == 0:
            continue
        leaf_type = int(node_P2MP[dest][p][3])
        leaf_base = int(node_P2MP[dest][p][5])
        seg = _find_leaf_sc_segment_with_base(
            P2MP_SC, P2MP_FS, dest, p,
            leaf_type, leaf_base,
            fs_s_glo, fs_e_glo,
            span_len,
            src_node, src_p,
            phys_nodes, des,
            usage, sc_cap,
            True,
        )
        if seg is None:
            continue
        return int(p), int(seg[0]), int(seg[1]), seg[2], False

    # 2) 尝试分配新 leaf
    for p in range(_p2mp_total):
        if int(node_P2MP[dest][p][2]) != 0:
            continue
        leaf_type = int(node_P2MP[dest][p][3])
        seg = _infer_leaf_base_and_find_sc(
            P2MP_SC, P2MP_FS, dest, p,
            leaf_type, src_node, src_p,
            fs_s_glo, fs_e_glo,
            span_len,
            phys_nodes, des,
            usage, sc_cap,
        )
        if seg is None:
            continue
        node_P2MP[dest][p][2] = 1
        node_P2MP[dest][p][5] = int(seg[2])
        node_flow[dest][p][0].clear()
        _clear_p2mp_sc(P2MP_SC, dest, p)
        return int(p), int(seg[0]), int(seg[1]), seg[3], True
    return None, None, None, None, False


def _plan_sc_allocation(src: int, hub_p: int, hub_type: int, hub_fs0: int, dest: int,
                        phys_nodes: list, bw: float, sc_cap: float, flow_ids_on_hub: list,
                        flow_acc: np.ndarray, P2MP_SC: np.ndarray,
                        node_P2MP: np.ndarray, node_flow: np.ndarray, link_FS: np.ndarray,
                        P2MP_FS: np.ndarray,
                        used_links: list, flow_path: np.ndarray):
    """
    为单条流在给定 hub 上规划 SC 分配，不直接修改状态。
    返回 (sc_s, sc_e, usage_dict{sc:used_bw}, fs_s_glo, fs_e_glo,
          leaf_p, leaf_sc_s, leaf_sc_e)，失败返回 None。
    """
    # 该 hub 类型允许的最大 SC 索引
    max_sc_idx = _TYPE_INFO[int(hub_type)][0] - 1

    # 当前流在单个 SC 上的可用容量基准
    sc_cap_use = float(sc_cap)

    # 枚举可能的起始 SC
    for sc_start in range(0, max_sc_idx + 1):
        # 需要分配的剩余带宽
        remaining = float(bw)
        # 记录每个 SC 实际分配的带宽
        usage = {}
        # 当前尝试填充的 SC 索引
        cur_sc = int(sc_start)

        # 在连续 SC 上贪心填充剩余带宽
        while remaining > 1e-9:
            if cur_sc > max_sc_idx:
                usage = None
                break
            if not _sc_can_use_on_hub(P2MP_SC, src, hub_p, int(cur_sc), phys_nodes, dest):
                usage = None
                break
            if not _sc_fs_path_ok(P2MP_FS, src, hub_p, int(hub_type), int(cur_sc), phys_nodes):
                usage = None
                break
            # 该 SC 在相同路径上的已占用带宽
            used_in_sc = _sc_used_bw_for_path(src, hub_p, int(cur_sc), phys_nodes, P2MP_SC)
            # 当前 SC 还能容纳的带宽
            cur_spare = float(sc_cap_use) - float(used_in_sc)
            if cur_spare <= 1e-9:
                usage = None
                break
            # 本 SC 分配的带宽不超过剩余需求与可用容量
            take = min(remaining, cur_spare)
            usage[cur_sc] = usage.get(cur_sc, 0.0) + float(take)
            remaining -= float(take)
            if remaining <= 1e-9:
                break
            # 继续尝试下一个 SC
            cur_sc += 1

        # 如果当前 sc_start 不可行，尝试下一个起点
        if usage is None:
            continue

        # 本次分配覆盖的 SC 终点
        sc_end = int(cur_sc)
        # 计算相对 FS 起止与绝对 FS 起止
        fs_s_rel = sc_fs(int(hub_type), int(sc_start), 1)
        fs_e_rel = sc_fs(int(hub_type), int(sc_end), 2)
        fs_s_glo = int(hub_fs0) + int(fs_s_rel)
        fs_e_glo = int(hub_fs0) + int(fs_e_rel)

        # 找出现有 hub 树使用过的链路集合
        tree_links = set()
        for f0 in flow_ids_on_hub:
            fid0 = int(f0[0])
            for l0 in flow_path[fid0][1]:
                tree_links.add(int(l0))
        new_links = set(used_links) - set(tree_links)
        new_fs_abs_list = []
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
                    if link_FS[int(l)][int(fs_abs)] != 0:
                        ok_links = False
                        break
                if not ok_links:
                    break
        if ok_links and new_links:
            for l in new_links:
                for fs_abs in range(int(fs_s_glo), int(fs_e_glo) + 1):
                    if link_FS[int(l)][int(fs_abs)] != 0:
                        ok_links = False
                        break
                if not ok_links:
                    break
        if not ok_links:
            continue

        # 检查同一相对 FS 上已有流的目的节点与目的端口是否一致
        fixed_leaf = None
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
            # 对该 FS 上的每条已占用流进行一致性校验
            for idx in range(min(len(dst_node_list), len(dst_p_list))):
                try:
                    dst_i = int(dst_node_list[idx])
                    leaf_i = int(dst_p_list[idx])
                except Exception:
                    continue
                # 目的节点不一致则不可复用
                if dst_i != int(dest):
                    fixed_leaf_invalid = True
                    break
                # 目的端口无效则不可复用
                if leaf_i < 0:
                    fixed_leaf_invalid = True
                    break
                # 首次遇到的 leaf 作为固定 leaf，后续必须一致
                if fixed_leaf is None:
                    fixed_leaf = leaf_i
                elif fixed_leaf != leaf_i:
                    fixed_leaf_invalid = True
                    break
            # 该 FS 已有占用但不满足一致性，直接判失败
            if fixed_leaf_invalid:
                break
        if fixed_leaf_invalid:
            continue

        # 选择或分配 leaf 端口，并获取 leaf 的 SC 段
        leaf_p, leaf_sc_s, leaf_sc_e, leaf_usage, _ = _alloc_new_leaf(
            node_P2MP, node_flow, P2MP_SC, P2MP_FS,
            dest, src, hub_p,
            fs_s_glo, fs_e_glo, int(sc_end) - int(sc_start) + 1,
            phys_nodes, dest,
            usage, sc_cap,
            fixed_leaf
        )
        if leaf_p is None or leaf_sc_s is None or leaf_sc_e is None or leaf_usage is None:
            continue

        # 返回本次规划的完整结果
        return int(sc_start), sc_end, usage, fs_s_glo, fs_e_glo, int(leaf_p), int(leaf_sc_s), int(leaf_sc_e), leaf_usage
    # 无可行分配方案
    return None


def allocate_flows_sequential(
    flows_info: np.ndarray,
    topo_num: int,
    topo_dis: np.ndarray,
    link_num: int,
    link_index: np.ndarray,
    Tbox_num: int,
    Tbox_P2MP: int,
    *,
    node_flow: np.ndarray,
    node_P2MP: np.ndarray,
    link_FS: np.ndarray,
    P2MP_SC: np.ndarray,
    P2MP_FS: np.ndarray,
    flow_acc: np.ndarray,
    flow_path: np.ndarray,
    stop_on_fail: bool = True,
):
    """
    逐流分配：
    - 若未提供已有状态/表，则在空状态上初始化并分配；
    - 返回 (node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, flow_path, failed_flow_ids)。
    """
    flows_arr = np.array(flows_info, dtype=object)

    flows_num = flows_arr.shape[0]

    _p2mp_total = Tbox_num * Tbox_P2MP
    failed = []

    # 按源节点分组（贴近 DP 初始化行为；有助于同源聚合）
    scr_flows = [[] for _ in range(topo_num)]
    for f in flows_arr:
        scr_flows[int(f[1])].append(f)

    for src in range(topo_num):
        for f in scr_flows[src]:
            fid = int(f[0])
            des = int(f[2])
            bw = float(f[3])
            sc_cap = float(f[4])

            # 物理最短路径（与 DP 初始化一致）
            path = k_shortest_path(topo_dis, src + 1, des + 1, 1)
            phys_nodes = path[0][0]
            used_links = get_links_from_path(phys_nodes, link_index)
            used_links = [int(x) - 1 for x in used_links]  # 0-based links

            placed = False

            # 1) 尝试复用源节点已有 hub
            for hub_p in range(_p2mp_total):
                if int(node_P2MP[src][hub_p][2]) != 1:
                    continue

                # 容量检查（不依赖 node_P2MP[...,4] 的动态值，直接统计已用带宽）
                used_bw = sum(float(x[3]) for x in node_flow[src][hub_p][0])
                if used_bw + bw > 400 + 1e-9:
                    continue

                hub_type = int(node_P2MP[src][hub_p][3])
                hub_fs0 = int(node_P2MP[src][hub_p][5])
                if hub_fs0 < 0:
                    continue
                flow_ids_on_hub = node_flow[src][hub_p][0]
                plan = _plan_sc_allocation(
                    src, hub_p, hub_type, hub_fs0, des,
                    phys_nodes, bw, sc_cap, flow_ids_on_hub, flow_acc, P2MP_SC,
                    node_P2MP, node_flow, link_FS, P2MP_FS, used_links, flow_path
                )
                if plan is None:
                    continue
                sc_s, sc_e, usage, fs_s_glo, fs_e_glo, leaf_p, leaf_sc_s, leaf_sc_e, leaf_usage = plan

                # 提交：hub + leaf
                # hub 侧
                node_flow[src][hub_p][0].append(f)
                # leaf side
                node_flow[des][leaf_p][0].append(f)

                flow_acc[fid][7] = int(hub_p)
                flow_acc[fid][8] = int(leaf_p)
                flow_acc[fid][9] = int(sc_s)
                flow_acc[fid][10] = int(sc_e)
                flow_acc[fid][11] = int(fs_s_glo)
                flow_acc[fid][12] = int(fs_e_glo)
                flow_acc[fid][13] = int(leaf_sc_s)
                flow_acc[fid][14] = int(leaf_sc_e)

                # 更新 hub 的 P2MP_SC
                for sc, used_amt in usage.items():
                    P2MP_SC[src][hub_p][int(sc)][1].append([fid, float(used_amt), f[6]])
                    P2MP_SC[src][hub_p][int(sc)][0].append(float(sc_cap))
                    P2MP_SC[src][hub_p][int(sc)][2].append(phys_nodes)
                    P2MP_SC[src][hub_p][int(sc)][4].append(int(des))
                    P2MP_SC[src][hub_p][int(sc)][5].append(int(src))
                    P2MP_SC[src][hub_p][int(sc)][6].append(int(hub_p))
                    P2MP_SC[src][hub_p][int(sc)][7].append(int(des))
                    P2MP_SC[src][hub_p][int(sc)][8].append(int(leaf_p))
                    _apply_p2mp_fs_usage(
                        P2MP_FS, src, hub_p, int(hub_type), int(sc),
                        fid, float(used_amt), phys_nodes, des, f[6],
                        src, hub_p, des, leaf_p,
                    )

                # 更新路径记录
                flow_path[fid][0].append(fid)
                flow_path[fid][1].extend(used_links)
                flow_path[fid][2].extend(phys_nodes)
                flow_path[fid][3].append(f[6])

                _apply_leaf_usage(
                    P2MP_SC, des, leaf_p,
                    int(leaf_sc_s), int(leaf_sc_e),
                    leaf_usage, sc_cap,
                    fid, phys_nodes, des, f[6],
                    src, hub_p, des, leaf_p,
                )
                leaf_type = int(node_P2MP[des][leaf_p][3])
                for sc, used_amt in leaf_usage.items():
                    _apply_p2mp_fs_usage(
                        P2MP_FS, des, leaf_p, int(leaf_type), int(sc),
                        fid, float(used_amt), phys_nodes, des, f[6],
                        src, hub_p, des, leaf_p,
                    )

                # 在涉及链路上占用光谱（与 DP 初始化一致）
                for l in set(used_links):
                    link_FS[int(l)][int(fs_s_glo):int(fs_e_glo) + 1] += 1

                placed = True
                break

            if placed:
                continue

            # 2) 在源节点开启新的 hub 并尝试放置
            for hub_p in range(_p2mp_total):
                if int(node_P2MP[src][hub_p][2]) != 0:
                    continue

                hub_type = int(node_P2MP[src][hub_p][3])
                block_size = _TYPE_INFO[int(hub_type)][1]
                hub_fs0 = _find_free_fs_block(link_FS, used_links, block_size)
                if hub_fs0 is None:
                    continue

                # 分配 hub
                node_P2MP[src][hub_p][2] = 1
                node_P2MP[src][hub_p][5] = int(hub_fs0)
                node_flow[src][hub_p][0].clear()
                _clear_p2mp_sc(P2MP_SC, src, hub_p)

                flow_ids_on_hub = node_flow[src][hub_p][0]

                plan = _plan_sc_allocation(
                    src, hub_p, hub_type, hub_fs0, des,
                    phys_nodes, bw, sc_cap, flow_ids_on_hub, flow_acc, P2MP_SC,
                    node_P2MP, node_flow, link_FS, P2MP_FS, used_links, flow_path
                )
                if plan is None:
                    # rollback hub and try next
                    node_P2MP[src][hub_p][2] = 0
                    node_P2MP[src][hub_p][5] = -1
                    node_flow[src][hub_p][0].clear()
                    _clear_p2mp_sc(P2MP_SC, src, hub_p)
                    continue
                sc_s, sc_e, usage, fs_s_glo, fs_e_glo, leaf_p, leaf_sc_s, leaf_sc_e, leaf_usage = plan

                # commit
                node_flow[src][hub_p][0].append(f)
                node_flow[des][leaf_p][0].append(f)

                flow_acc[fid][7] = int(hub_p)
                flow_acc[fid][8] = int(leaf_p)
                flow_acc[fid][9] = int(sc_s)
                flow_acc[fid][10] = int(sc_e)
                flow_acc[fid][11] = int(fs_s_glo)
                flow_acc[fid][12] = int(fs_e_glo)
                flow_acc[fid][13] = int(leaf_sc_s)
                flow_acc[fid][14] = int(leaf_sc_e)

                for sc, used_amt in usage.items():
                    P2MP_SC[src][hub_p][int(sc)][1].append([fid, float(used_amt), f[6]])
                    P2MP_SC[src][hub_p][int(sc)][0].append(float(sc_cap))
                    P2MP_SC[src][hub_p][int(sc)][2].append(phys_nodes)
                    P2MP_SC[src][hub_p][int(sc)][4].append(int(des))
                    P2MP_SC[src][hub_p][int(sc)][5].append(int(src))
                    P2MP_SC[src][hub_p][int(sc)][6].append(int(hub_p))
                    P2MP_SC[src][hub_p][int(sc)][7].append(int(des))
                    P2MP_SC[src][hub_p][int(sc)][8].append(int(leaf_p))
                    _apply_p2mp_fs_usage(
                        P2MP_FS, src, hub_p, int(hub_type), int(sc),
                        fid, float(used_amt), phys_nodes, des, f[6],
                        src, hub_p, des, leaf_p,
                    )

                flow_path[fid][0].append(fid)
                flow_path[fid][1].extend(used_links)
                flow_path[fid][2].extend(phys_nodes)
                flow_path[fid][3].append(f[6])

                _apply_leaf_usage(
                    P2MP_SC, des, leaf_p,
                    int(leaf_sc_s), int(leaf_sc_e),
                    leaf_usage, sc_cap,
                    fid, phys_nodes, des, f[6],
                    src, hub_p, des, leaf_p,
                )
                leaf_type = int(node_P2MP[des][leaf_p][3])
                for sc, used_amt in leaf_usage.items():
                    _apply_p2mp_fs_usage(
                        P2MP_FS, des, leaf_p, int(leaf_type), int(sc),
                        fid, float(used_amt), phys_nodes, des, f[6],
                        src, hub_p, des, leaf_p,
                    )

                for l in set(used_links):
                    link_FS[int(l)][int(fs_s_glo):int(fs_e_glo) + 1] += 1

                placed = True
                break

            if not placed:
                failed.append(fid)
                if stop_on_fail:
                    raise RuntimeError(f"Sequential allocation failed for flow_id={fid} (src={src}, des={des}, bw={bw})")

    return node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, P2MP_FS, flow_path


def FlexE_P2MP_Sequential(flows_info, flows_num, topo_num, topo_dis, link_num, link_index, Tbox_num, Tbox_P2MP,
                          node_flow, node_P2MP, link_FS, P2MP_SC, P2MP_FS, flow_acc, flow_path,
                          *, stop_on_fail: bool = True):
    """
    便捷封装：保持与原 DP 初始化器一致的函数签名与返回形式。
    注意：flows_num 应等于 flows_info 的长度（逐跳拆解后的子流表）。
    """
    node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, P2MP_FS, flow_path, _ = allocate_flows_sequential(
        flows_info=np.array(flows_info, dtype=object),
        topo_num=topo_num,
        topo_dis=topo_dis,
        link_num=link_num,
        link_index=link_index,
        Tbox_num=Tbox_num,
        Tbox_P2MP=Tbox_P2MP,
        node_flow=node_flow,
        node_P2MP=node_P2MP,
        link_FS=link_FS,
        P2MP_SC=P2MP_SC,
        P2MP_FS=P2MP_FS,
        flow_acc=flow_acc,
        flow_path=flow_path,
        stop_on_fail=stop_on_fail,
    )
    return node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, P2MP_FS, flow_path


def restore_flows_sequential(
    affected_flow: list,
    topo_num: int,
    topo_dis: np.ndarray,
    link_num: int,
    link_index: np.ndarray,
    Tbox_num: int,
    Tbox_P2MP: int,
    node_flow: np.ndarray,
    node_P2MP: np.ndarray,
    link_FS: np.ndarray,
    P2MP_SC: np.ndarray,
    P2MP_FS: np.ndarray,
    flow_acc: np.ndarray,
    flow_path: np.ndarray,
    break_node: int,
):
    """
    基于已有资源表顺序恢复受影响流：
    - 屏蔽故障节点；
    - 为受影响的原始流直接按单跳流重建基础字段；
    - 复用顺序分配器在当前状态上做重新分配。
    """
    # 屏蔽故障节点：故障节点对其它节点的距离设为无穷大
    topo_dis_rest = np.array(topo_dis, copy=True)
    topo_dis_rest[break_node, :] = np.inf
    topo_dis_rest[:, break_node] = np.inf
    topo_dis_rest[break_node, break_node] = 0

    # 逐条流构建可恢复列表（新流追加到已有表末尾）
    flows_info = []
    new_items = []
    failed = []
    seen_orig = set()
    next_id = int(flow_acc.shape[0])

    for f in affected_flow:
        orig_id = int(f[0])
        # 同一原流只处理一次，避免重复覆盖
        if orig_id in seen_orig:
            continue
        seen_orig.add(orig_id)
        src = int(f[1])
        dst = int(f[2])
        bw = float(f[3])
        
        # 故障节点作为端点的流不可恢复
        if src == break_node or dst == break_node:
            failed.append(orig_id)
            continue
        try:
            # 计算恢复路径（1-based 节点序列）
            path = k_shortest_path(topo_dis_rest, src + 1, dst + 1, 1)
            phys_nodes = path[0][0]
        except Exception:
            failed.append(orig_id)
            continue
        if not phys_nodes or len(phys_nodes) < 2:
            failed.append(orig_id)
            continue
        # 依据恢复路径距离重算调制与 SC 需求
        dist = 0.0
        for a, b in zip(phys_nodes[:-1], phys_nodes[1:]):
            dist += float(topo_dis[int(a) - 1][int(b) - 1])
        sc_cap_raw, _ = modu_format_Al(dist, bw)
        sc_cap = sc_effective_cap(float(sc_cap_raw))
        sc_num = sc_num_from_bw_cap(float(bw), float(sc_cap))

        # 为恢复流分配新的子流编号（追加在末尾）
        sub_id = next_id
        next_id += 1

        # 构建基础字段（与 flow_acc 的 0..6 列一致）
        item = [int(sub_id), src, dst, float(bw), float(sc_cap), int(sc_num), int(orig_id)]
        flows_info.append(item)
        new_items.append(item)

    if not flows_info:
        return node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, P2MP_FS, flow_path, failed

    # 扩容 flow_acc 与 flow_path
    add_n = len(new_items)
    flow_acc_ext = np.zeros((add_n, flow_acc.shape[1]), dtype=object)
    for i in range(add_n):
        flow_acc_ext[i][7:15] = [-1] * 8
    flow_acc = np.vstack([flow_acc, flow_acc_ext])

    flow_path_ext = np.empty((add_n, flow_path.shape[1]), dtype=object)
    for i in range(add_n):
        for c in range(flow_path.shape[1]):
            flow_path_ext[i][c] = []
    flow_path = np.vstack([flow_path, flow_path_ext])

    # 写入新流的基础字段并初始化路径记录
    sub_to_orig = {}
    for item in new_items:
        sid = int(item[0])
        sub_to_orig[sid] = int(item[6])
        flow_acc[sid][0:7] = item
        flow_path[sid][0] = []
        flow_path[sid][1] = []
        flow_path[sid][2] = []
        flow_path[sid][3] = []

    # 复用顺序分配器在现有资源表上进行恢复分配
    node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, P2MP_FS, flow_path, failed_sub = allocate_flows_sequential(
        flows_info=np.array(flows_info, dtype=object),
        topo_num=topo_num,
        topo_dis=topo_dis_rest,
        link_num=link_num,
        link_index=link_index,
        Tbox_num=Tbox_num,
        Tbox_P2MP=Tbox_P2MP,
        node_flow=node_flow,
        node_P2MP=node_P2MP,
        link_FS=link_FS,
        P2MP_SC=P2MP_SC,
        P2MP_FS=P2MP_FS,
        flow_acc=flow_acc,
        flow_path=flow_path,
        stop_on_fail=False,
    )

    failed_orig = list(failed)
    for sid in failed_sub:
        try:
            failed_orig.append(int(sub_to_orig.get(int(sid), int(sid))))
        except Exception:
            continue
    failed_orig = list(dict.fromkeys(failed_orig))
    return node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, P2MP_FS, flow_path, failed_orig
