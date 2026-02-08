#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : ILP.py 
# @File    : assign_leaf.py
# @Author  : Wumh
# @Time    : 2026/1/2 0:37

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class ReuseCtx:
    ref_dst: int
    ref_leaf_p2mp: int
    ref_subflows: Tuple[int, ...]


ChosenLeaf = Tuple[int, int, int, int]  # (leaf_p2mp, r_sc0, r_sc1, r_base_final)


def _acc_get_dst_leaf(new_flow_acc: Any, subflow_id: int) -> Tuple[int, int]:
    """
    兼容 new_flow_acc 常见格式：
    - 旧: dst 在 [2], leaf 在 [7]
    - 新: dst 在 [2], leaf 在 [8]
    """
    row = new_flow_acc[subflow_id]
    dst = int(row[2])
    if len(row) >= 9:
        leaf = int(row[8])
    else:
        leaf = int(row[7])
    return dst, leaf


def _sc_has_capacity(new_P2MP_SC_1: Any, u: int, p: int, s: int, need: int = 5) -> bool:
    """
    一个 SC 被一条流占用消耗 5（与 ILP 中 *5 口径对齐）。
    new_P2MP_SC_1[u][p][s][3][0] 记录剩余容量。
    """
    try:
        cap_left = int(new_P2MP_SC_1[u][p][s][3][0])
    except Exception:
        return False
    return cap_left >= need


def choose_receiver_leaf_for_fs_segment(
    *,
    flow,
    b: int,
    fs_s_abs: int,
    fs_e_abs: int,
    sc_need: int,
    P2MP_reuse_fs: bool,
    ref_flow_id: Sequence[int],
    new_flow_acc: Any,
    new_node_P2MP: Any,
    new_P2MP_SC_1: Any,
    new_P2MP_FS_1: Any,
    # 依赖你文件里已有的函数（这里直接调用，不重写）
    fs_abs_range,                 # callable(r_type, base_fs, sc0, sc1) -> (abs_s, abs_e, rel_s, rel_e)
    sc_fs,                        # callable(r_type, sc, idx) -> rel_fs boundary
    _fs_range_inside_p2mp_block,  # callable(r_type, base_fs, abs_s, abs_e) -> bool
) -> Tuple[Optional[ChosenLeaf], Optional[ReuseCtx]]:
    """
    选择 receiver 侧 leaf P2MP，并确定其 SC 段与最终 base_fs。
    要求最终映射的 FS 绝对段严格等于 [fs_s_abs, fs_e_abs]。

    返回:
      - chosen_leaf: (leaf_p2mp, r_sc0, r_sc1, r_base_final) 或 None
      - reuse_ctx: 复用场景返回 ReuseCtx；否则 None
    """
    b = int(b)
    fs_s_abs = int(fs_s_abs)
    fs_e_abs = int(fs_e_abs)
    sc_need = int(sc_need)
    band = flow[3]

    if sc_need <= 0 or sc_need > 16:
        return None, None
    if fs_s_abs > fs_e_abs:
        return None, None

    def _find_sc_segment_with_base(rp: int, r_type: int, base_fs: int) -> Optional[Tuple[int, int]]:
        """
        在给定 base_fs 下找 (r_sc0,r_sc1)，要求：
          1) 该 SC 区间映射的 FS 绝对段 == [fs_s_abs, fs_e_abs]
          2) 区间内 SC 的“剩余容量总和” >= need_cap（允许 SC 复用/部分占用）
        返回一个可行区间。策略：优先最短区间，其次更小的 r_sc0。
        """
        if not _fs_range_inside_p2mp_block(r_type, base_fs, fs_s_abs, fs_e_abs):
            return None

        need_cap = band

        best: Optional[Tuple[int, int, int]] = None  # (span, sc0, sc1)

        # 不再固定长度；枚举所有 sc0<=sc1
        for r_sc0 in range(0, 16):
            for r_sc1 in range(r_sc0, 16):
                aa_s, aa_e, _, _ = fs_abs_range(r_type, base_fs, r_sc0, r_sc1)
                if int(aa_s) != fs_s_abs or int(aa_e) != fs_e_abs:
                    continue

                # 计算总剩余容量（允许某个 SC 已经部分被用）
                cap_sum = 0
                for s in range(r_sc0, r_sc1 + 1):
                    try:
                        cap_left = int(new_P2MP_SC_1[b][rp][s][3][0])
                    except Exception:
                        cap_left = 0
                    if cap_left > 0:
                        cap_sum += cap_left

                if cap_sum < need_cap:
                    continue

                span = r_sc1 - r_sc0 + 1
                cand = (span, r_sc0, r_sc1)
                if best is None or cand < best:
                    best = cand

        if best is None:
            return None
        return best[1], best[2]

    def _infer_base_and_find_sc(rp: int, r_type: int) -> Optional[Tuple[int, int, int]]:
        """
        base_fs 未知时：不再固定 sc_need 窗口。
        做法：
          - 枚举可能的起始 SC：r_sc0 ∈ [0,15]
          - 用该起点的 rel_s 反推 base_cand = fs_s_abs - rel_s
          - 然后调用 _find_sc_segment_with_base(rp, r_type, base_cand)
            让它在该 base 下枚举所有 (sc0,sc1) 找一个能精确对齐 FS 段且“总剩余容量”足够的区间。
        返回 (r_sc0, r_sc1, base_cand) 或 None
        """
        for r_sc0_probe in range(0, 16):
            rel_s = int(sc_fs(r_type, int(r_sc0_probe), 1))
            base_cand = int(fs_s_abs - rel_s)
            if base_cand < 0:
                continue

            # 快速剪枝：FS 段必须落在该 block 覆盖内
            if not _fs_range_inside_p2mp_block(r_type, base_cand, fs_s_abs, fs_e_abs):
                continue

            seg = _find_sc_segment_with_base(rp, r_type, base_cand)
            if seg is None:
                continue

            # seg 已经保证 FS 段精确对齐且容量总和 >= need_cap
            return int(seg[0]), int(seg[1]), int(base_cand)

        return None

    def _fs_segment_unused_with_base(rp: int, base_fs: int) -> bool:
        """非复用场景：目标 FS 段在该 leaf 上必须无人使用。"""
        for fs_abs in range(fs_s_abs, fs_e_abs + 1):
            fs_rel = int(fs_abs - base_fs)
            used_list = new_P2MP_FS_1[b, rp, fs_rel, 1]
            if used_list and len(used_list) > 0:
                return False
        return True

    # -------------------- 复用场景 --------------------
    if P2MP_reuse_fs:
        if not ref_flow_id:
            return None, None

        ref_dsts: List[int] = []
        ref_leafs: List[int] = []
        for sub_id in ref_flow_id:
            d, lf = _acc_get_dst_leaf(new_flow_acc, int(sub_id))
            ref_dsts.append(int(d))
            ref_leafs.append(int(lf))

        if any(d != ref_dsts[0] for d in ref_dsts):
            return None, None
        if any(lf != ref_leafs[0] for lf in ref_leafs):
            return None, None

        ref_dst = int(ref_dsts[0])
        ref_leaf_p = int(ref_leafs[0])
        if ref_dst != b:
            return None, None

        r_type = int(new_node_P2MP[b][ref_leaf_p][3])
        r_base = int(new_node_P2MP[b][ref_leaf_p][5])

        if r_base >= 0:
            seg = _find_sc_segment_with_base(ref_leaf_p, r_type, r_base)
            if seg is None:
                return None, None
            chosen: ChosenLeaf = (ref_leaf_p, seg[0], seg[1], int(r_base))
        else:
            inf = _infer_base_and_find_sc(ref_leaf_p, r_type)
            if inf is None:
                return None, None
            chosen = (ref_leaf_p, inf[0], inf[1], inf[2])

        ctx = ReuseCtx(
            ref_dst=ref_dst,
            ref_leaf_p2mp=ref_leaf_p,
            ref_subflows=tuple(int(x) for x in ref_flow_id),
        )
        return chosen, ctx

    # -------------------- 非复用场景 --------------------
    # 遍历 b 的所有 leaf P2MP，找第一个可行（你也可以改成“最优”策略）
    for rp in range(len(new_node_P2MP[b])):

        r_type = int(new_node_P2MP[b][rp][3])
        r_base = int(new_node_P2MP[b][rp][5])

        if r_base >= 0:
            if not _fs_range_inside_p2mp_block(r_type, r_base, fs_s_abs, fs_e_abs):
                continue
            if not _fs_segment_unused_with_base(rp, r_base):
                continue
            seg = _find_sc_segment_with_base(rp, r_type, r_base)
            if seg is None:
                continue
            return (rp, seg[0], seg[1], int(r_base)), None
        else:
            inf = _infer_base_and_find_sc(rp, r_type)
            if inf is None:
                continue
            base_cand = inf[2]
            if not _fs_segment_unused_with_base(rp, base_cand):
                continue
            return (rp, inf[0], inf[1], base_cand), None

    return None, None
