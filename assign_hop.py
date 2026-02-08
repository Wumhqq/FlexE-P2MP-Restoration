#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm
# @Project : ILP.py
# @File    : assign_hop.py
# @Author  : Wumh (refactored with assistant)
# @Time    : 2026/01/02

"""FlexE-over-P2MP 恢复（策略1）中“单跳/单段光路”的资源试探分配。

本文件提供 `try_assign_one_hop_s1`：给定逻辑 hop (a->b) 以及一条物理候选路径 `phy_candidate["path"]`
（节点列表为 0-based），尝试找到一组**同时可行**的资源方案：

- 发送端节点 a：选择一个 P2MP block `sp`，并分配连续子载波区间 [s_sc0, s_sc1] 及其 base_fs `s_base`
- 接收端节点 b：选择一个 P2MP block `rp`，并分配连续子载波区间 [r_sc0, r_sc1] 及其 base_fs `r_base`
- 在链路层：确定绝对 FS 段 [fs_s_abs, fs_e_abs]（由 P2MP 类型 + SC 区间映射得到）

核心约束（与你的最新确认严格对齐）：
1) **不区分 hub/leaf**：在本阶段只要该 P2MP block 有足够空闲 SC 资源即可作为发送或接收端。
2) 链路 FS 允许被多个 flow 占用，但必须满足**路径一致性**：同一 link 的同一 FS 若已被占用，
   则该占用记录中的 path 必须与当前 hop 的物理路径一致，否则不可用。
   其中 `path_list` 可能存 1-based（你已确认）；本实现同时兼容 0-based/1-based 两种表示。
3) 发送端 P2MP 的 FS 复用判定：若目标 FS 段上任一 FS 的 `path_list` 非空，则视为复用场景；
   且每个被占用 FS 都必须能匹配到当前 hop 路径，否则失败。
   `used_list[0][0]` 被视为 subflow_id，`used_list[0][2]` 被视为 orig_flow_id（你已确认）。
4) 接收端（b 侧）的可行性与 base 推断由 `assign_leaf.choose_receiver_leaf_for_fs_segment` 统一负责，
   从而保证“发送端/接收端必须成对找到，否则不可恢复”。

返回值：
  (ok: bool, plan: Optional[dict])

注意：本函数仅做“试探分配/选方案”，**不直接写入**任何资源矩阵，方便你的主循环做回退与多候选搜索。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from modu_format_Al import modu_format_Al
from SC_FS import sc_fs

from assign_leaf import ReuseCtx, choose_receiver_leaf_for_fs_segment


# ========================================
# 通用辅助函数（尽量兼容 numpy/object array 的索引与类型）
# ========================================

def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _path_to_tuple(p: Any) -> Optional[Tuple[int, ...]]:
    """将元数据中存储的路径转换为整数元组；失败则返回 None。"""
    if p is None:
        return None
    try:
        if hasattr(p, "tolist"):
            p = p.tolist()
        if isinstance(p, (list, tuple)):
            return tuple(int(x) for x in p)
        # 有时会出现“单个标量”被错误存储的情况
        return None
    except Exception:
        return None


def _iter_paths(paths_obj: Any) -> Iterable[Any]:
    """从 meta 字段中迭代路径：字段可能是 None / 单条路径 / 多条路径列表。"""
    if paths_obj is None:
        return []
    if hasattr(paths_obj, "tolist"):
        paths_obj = paths_obj.tolist()
    # empty
    try:
        if len(paths_obj) == 0:  # type: ignore[arg-type]
            return []
    except Exception:
        pass

    # 若是 int 列表/元组，则视作“单条路径”
    if isinstance(paths_obj, (list, tuple)):
        if len(paths_obj) > 0 and isinstance(paths_obj[0], (int, float)):
            return [paths_obj]
        return list(paths_obj)

    return [paths_obj]


def _path_match(pp: Any, path_t: Tuple[int, ...]) -> bool:
    t = _path_to_tuple(pp)
    return (t is not None) and (t == path_t)


# ========================================
# 频谱映射辅助函数（SC ↔ FS）
# ========================================

def _norm_sc_cap(sc_cap_raw: float) -> int:
    """工程约定：部分模块将 12.5Gbps 视作 10Gbps 做归一化。"""
    if abs(sc_cap_raw - 12.5) < 1e-9:
        return 10
    return int(sc_cap_raw)


def _max_rel_fs(p_type: int) -> int:
    """给定 P2MP 类型，返回其覆盖的最大相对 FS 索引（rel FS 上界）。"""
    p_type = int(p_type)
    if p_type == 1:
        return 0
    if p_type == 2:
        return 1
    return 5  # type>=3


def _fs_range_inside_p2mp_block(p_type: int, base_fs: int, fs_s_abs: int, fs_e_abs: int) -> bool:
    """判断绝对 FS 段 [fs_s_abs,fs_e_abs] 是否落在该 P2MP block 覆盖范围内。"""
    if int(base_fs) < 0:
        return False
    max_rel = _max_rel_fs(p_type)
    return int(fs_s_abs) >= int(base_fs) and int(fs_e_abs) <= int(base_fs) + int(max_rel)


def fs_abs_range(p_type: int, base_fs: int, sc0: int, sc1: int) -> Tuple[int, int, int, int]:
    """将连续 SC 区间映射到绝对 FS 段，同时返回相对边界 rel_s/rel_e。"""
    rel_s = int(sc_fs(int(p_type), int(sc0), 1))
    rel_e = int(sc_fs(int(p_type), int(sc1), 2))
    return int(base_fs + rel_s), int(base_fs + rel_e), rel_s, rel_e


# ========================================
# 容量辅助函数（SC 剩余容量）
# ========================================

def _sc_cap_left(new_P2MP_SC_1: Any, u: int, p: int, s: int) -> int:
    """读取 (u,p,s) 的剩余容量（尽量兼容 list / numpy / object array 的索引方式）。"""
    try:
        return int(new_P2MP_SC_1[u][p][s][3][0])
    except Exception:
        try:
            return int(new_P2MP_SC_1[u, p, s, 3, 0])
        except Exception:
            return 0


def _sum_sc_cap(new_P2MP_SC_1: Any, u: int, p: int, sc0: int, sc1: int) -> int:
    return sum(_sc_cap_left(new_P2MP_SC_1, u, p, s) for s in range(int(sc0), int(sc1) + 1))


def _iter_sc_segments_by_span() -> List[Tuple[int, int]]:
    """按区间长度从小到大枚举 (sc0, sc1)，同长度下 sc0 从小到大。"""
    segs: List[Tuple[int, int]] = []
    for span in range(1, 17):
        for sc0 in range(0, 16 - span + 1):
            segs.append((sc0, sc0 + span - 1))
    return segs


SC_SEGMENTS_BY_SPAN: List[Tuple[int, int]] = _iter_sc_segments_by_span()



def _find_sc_segment_for_abs_fs(
    *,
    u: int,
    p: int,
    p_type: int,
    base_fs: int,
    fs_s_abs: int,
    fs_e_abs: int,
    band: int,
    new_P2MP_SC_1: Any,
) -> Optional[Tuple[int, int]]:
    """当 base_fs 已知时，寻找一个 (sc0,sc1) 使其映射得到的绝对 FS 段恰为 [fs_s_abs, fs_e_abs]。"""
    if base_fs < 0:
        return None
    for sc0, sc1 in SC_SEGMENTS_BY_SPAN:
        s, e, _, _ = fs_abs_range(p_type, base_fs, sc0, sc1)
        if s == fs_s_abs and e == fs_e_abs:
            if _sum_sc_cap(new_P2MP_SC_1, u, p, sc0, sc1) >= band:
                return sc0, sc1
    return None


def _infer_base_and_sc_for_abs_fs(
    *,
    u: int,
    p: int,
    p_type: int,
    fs_s_abs: int,
    fs_e_abs: int,
    band: int,
    new_P2MP_SC_1: Any,
    fs_total: int,
) -> Optional[Tuple[int, int, int]]:
    """当 base_fs 未知时，反推出 (sc0, sc1, base_fs)，要求绝对 FS 段精确对齐 [fs_s_abs, fs_e_abs]。

推导：
  fs_s_abs = base + rel_s(sc0)
  fs_e_abs = base + rel_e(sc1)
因此 base = fs_s_abs - rel_s = fs_e_abs - rel_e，必须一致。

同时需要满足：base 合法、block 覆盖范围不越界、以及该 SC 区间剩余容量 >= band。
"""
    max_rel = _max_rel_fs(p_type)
    for sc0, sc1 in SC_SEGMENTS_BY_SPAN:
        rel_s = int(sc_fs(int(p_type), int(sc0), 1))
        rel_e = int(sc_fs(int(p_type), int(sc1), 2))
        base1 = int(fs_s_abs) - rel_s
        base2 = int(fs_e_abs) - rel_e
        if base1 != base2:
            continue
        base_fs = base1
        if base_fs < 0 or base_fs + max_rel >= fs_total:
            continue
        if _sum_sc_cap(new_P2MP_SC_1, u, p, sc0, sc1) < band:
            continue
        return sc0, sc1, base_fs
    return None


# ========================================
# 路径一致性检查（FS 复用约束）
# ========================================

def check_sender_p2mp_fs_reuse(
    *,
    a: int,
    sp: int,
    fs_s_abs: int,
    fs_e_abs: int,
    s_base: int,
    new_P2MP_FS_1: Any,
    path0_t: Tuple[int, ...],
    path1_t: Tuple[int, ...],
) -> Optional[Tuple[bool, Tuple[int, ...], Tuple[Tuple[int, int], ...]]]:
    """检查发送端 P2MP 的目标 FS 段是否触发“复用”，并验证路径一致性。

返回：
  (reuse_fs, ref_subflow_ids, ref_orig_flow_ids_with_fs_abs)

其中：
- reuse_fs：只要目标 FS 段上任一 FS 的 path_list 非空，则为 True；
- ref_subflow_ids：从 used_list 中提取的 subflow_id（used_list[0][0]）；
- ref_orig_flow_ids_with_fs_abs：提取 orig_flow_id（used_list[0][2]）并附带 fs_abs。

注意：path_list 存储为 1-based（也可能存在 0-based），本实现两者均兼容。
"""
    reuse_fs = False
    ref_flow_ids: List[int] = []
    ref_orig_flow_ids: List[Tuple[int, int]] = []

    if s_base < 0:
        return None

    for fs_abs in range(int(fs_s_abs), int(fs_e_abs) + 1):
        fs_rel = int(fs_abs - s_base)
        try:
            path_list = new_P2MP_FS_1[a, sp, fs_rel, 2]
            used_list = new_P2MP_FS_1[a, sp, fs_rel, 1]
        except Exception:
            # 结构异常/维度不符合预期
            return None

        paths = list(_iter_paths(path_list))
        if paths:
            reuse_fs = True
            if not any(_path_match(pp, path1_t) for pp in paths):
                return None

            if used_list is not None and isinstance(used_list, list) and len(used_list) > 0:
                # used_list[0][0] 为 subflow_id；used_list[0][2] 为 orig_flow_id（按你的确认）
                sub_id = _as_int(used_list[0][0], -1)
                if sub_id >= 0:
                    ref_flow_ids.append(sub_id)
                if len(used_list[0]) >= 3:
                    ref_orig_flow_ids.append((_as_int(used_list[0][2], sub_id), fs_abs))
                else:
                    ref_orig_flow_ids.append((sub_id, fs_abs))

    return reuse_fs, tuple(ref_flow_ids), tuple(ref_orig_flow_ids)


def check_link_fs_path_consistency(
    *,
    used_links_0based: Sequence[int],
    fs_s_abs: int,
    fs_e_abs: int,
    new_link_FS_meta: Any,
    path0_t: Tuple[int, ...],
    path1_t: Tuple[int, ...],
) -> bool:
    """对 hop 物理路径上的每一条链路、以及目标绝对 FS 段内的每个 FS：
若该 (link,fs) 已被占用，则必须满足占用记录的路径与当前 hop 路径一致。
"""
    for l in used_links_0based:
        for fs_abs in range(int(fs_s_abs), int(fs_e_abs) + 1):
            try:
                used_list = new_link_FS_meta[l, fs_abs, 1]
                path_list = new_link_FS_meta[l, fs_abs, 2]
            except Exception:
                return False

            if used_list is None:
                continue

            occupied = False
            if isinstance(used_list, list):
                occupied = len(used_list) > 0
            else:
                # 兼容：某些版本会把占用数量存成标量
                try:
                    occupied = int(used_list) > 0
                except Exception:
                    occupied = False

            if not occupied:
                continue

            paths = list(_iter_paths(path_list))
            if not paths:
                return False
            if not any(_path_match(pp, path1_t) for pp in paths):
                return False

    return True


def _feasible_fs_mask_on_links(
    *,
    used_links_0based: Sequence[int],
    fs_total: int,
    new_link_FS_meta: Any,
    path0_t: Tuple[int, ...],
    path1_t: Tuple[int, ...],
) -> List[bool]:
    """预计算每个绝对 FS 在该 hop 物理路径上的可行性。

对每个 fs_abs：若 hop 路径上的任一链路在该 FS 上被占用且路径不一致，则该 FS 不可用；
否则可用（包含“空闲”或“同一路径复用”两种情况）。
"""
    ok = [True] * fs_total
    for fs_abs in range(fs_total):
        for l in used_links_0based:
            try:
                used_list = new_link_FS_meta[l, fs_abs, 1]
                path_list = new_link_FS_meta[l, fs_abs, 2]
            except Exception:
                ok[fs_abs] = False
                break

            occupied = False
            if isinstance(used_list, list):
                occupied = len(used_list) > 0
            else:
                try:
                    occupied = int(used_list) > 0
                except Exception:
                    occupied = False

            if not occupied:
                continue

            paths = list(_iter_paths(path_list))
            if not paths or (not any(_path_match(pp, path1_t) for pp in paths)):
                ok[fs_abs] = False
                break
    return ok


def _segment_all_true(mask: Sequence[bool], s: int, e: int) -> bool:
    if s < 0 or e >= len(mask) or s > e:
        return False
    # 小段（<=6）直接线性扫描即可
    for i in range(s, e + 1):
        if not mask[i]:
            return False
    return True


def _fs_segment_unused_on_node_p2mp(
    *,
    node: int,
    p: int,
    base_fs: int,
    fs_s_abs: int,
    fs_e_abs: int,
    new_P2MP_FS_1: Any,
) -> bool:
    """非复用场景：接收端该 node/p2mp 上目标 FS 段必须完全空闲。"""
    if base_fs < 0:
        return False
    for fs_abs in range(int(fs_s_abs), int(fs_e_abs) + 1):
        fs_rel = int(fs_abs - base_fs)
        try:
            used_list = new_P2MP_FS_1[node, p, fs_rel, 1]
        except Exception:
            return False
        if used_list is not None and isinstance(used_list, list) and len(used_list) > 0:
            return False
    return True


# ========================================
# 方案对象（仅用于返回/后续 commit）
# ========================================

@dataclass(frozen=True)
class HopPlan:
    f_id: int
    a: int
    b: int

    phy_path0: Tuple[int, ...]
    used_links_0based: Tuple[int, ...]

    # sender
    sp: int
    s_type: int
    s_base: int
    s_sc0: int
    s_sc1: int

    # receiver
    rp: int
    r_type: int
    r_base: int
    r_sc0: int
    r_sc1: int

    # spectrum
    fs_s_abs: int
    fs_e_abs: int

    # reuse info
    P2MP_reuse_fs: bool
    ref_flow_id: Tuple[int, ...]
    ref_orig_flow_id: Tuple[Tuple[int, int], ...]
    reuse_ctx: Optional[ReuseCtx]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # 将 reuse_ctx 展开成普通 dict（或 None）
        if self.reuse_ctx is not None:
            d["reuse_ctx"] = asdict(self.reuse_ctx)
        return d


def _sender_candidates(a: int, new_node_P2MP: Any) -> List[int]:
    """不区分 hub/leaf；仅作为启发式优先尝试 base_fs>=0（可视为已激活）的 block。"""
    active: List[int] = []
    inactive: List[int] = []
    for p in range(len(new_node_P2MP[a])):
        try:
            base_fs = int(new_node_P2MP[a][p][5])
        except Exception:
            base_fs = -1
        if base_fs >= 0:
            active.append(p)
        else:
            inactive.append(p)
    return active + inactive

def choose_sender_p2mp_for_abs_fs(
    *,
    flow: Sequence[Any],
    a: int,
    fs_s_abs: int,
    fs_e_abs: int,
    feasible_mask: Sequence[bool],
    used_links_0based: Sequence[int],
    phy_path0_t: Tuple[int, ...],
    phy_path1_t: Tuple[int, ...],
    new_node_P2MP: Any,
    new_P2MP_SC_1: Any,
    new_P2MP_FS_1: Any,
    new_link_FS_meta: Any,
    fs_total: int,
) -> Optional[Tuple[int, int, int, int, bool, Tuple[int, ...], Tuple[Tuple[int, int], ...]]]:
    """在发送端 a 侧，为给定绝对 FS 段 [fs_s_abs, fs_e_abs] 选择一个可行的 P2MP block 与 SC 区间。

    该过程与接收端的 `choose_receiver_leaf_for_fs_segment` 对称（只是我们目前只实现了接收端选择器）：
      1) 在 a 上枚举 P2MP block（不区分 hub/leaf；优先 base_fs>=0 的 block）。
      2) 若 base_fs 已知：在该 block 内寻找某个 (sc0,sc1) 使映射的绝对 FS 段恰为 [fs_s_abs,fs_e_abs]；
         若 base_fs 未知：反推 base_fs 并同时给出 (sc0,sc1)。
      3) 检查 SC 容量 >= band；检查链路可用性（feasible_mask）；再做：
         - 发送端 P2MP 复用判定 + 路径一致性检查；
         - 链路层路径一致性检查（更严格、保守）。

    返回：
      (sp, s_sc0, s_sc1, s_base, P2MP_reuse_fs, ref_flow_id, ref_orig_flow_id)
      若找不到可行方案则返回 None。
    """
    f_id = _as_int(flow[0])
    band = _as_int(flow[3])

    # 绝对 FS 段必须在链路可行集合中
    if not _segment_all_true(feasible_mask, int(fs_s_abs), int(fs_e_abs)):
        return None

    for sp in _sender_candidates(a, new_node_P2MP):
        s_type = _as_int(new_node_P2MP[a][sp][3])
        s_base0 = _as_int(new_node_P2MP[a][sp][5], -1)

        if s_base0 >= 0:
            seg = _find_sc_segment_for_abs_fs(
                u=a, p=sp, p_type=s_type, base_fs=s_base0,
                fs_s_abs=fs_s_abs, fs_e_abs=fs_e_abs,
                band=band, new_P2MP_SC_1=new_P2MP_SC_1
            )
            if seg is None:
                continue
            s_sc0, s_sc1 = seg
            s_base = s_base0
        else:
            inf = _infer_base_and_sc_for_abs_fs(
                u=a, p=sp, p_type=s_type,
                fs_s_abs=fs_s_abs, fs_e_abs=fs_e_abs,
                band=band, new_P2MP_SC_1=new_P2MP_SC_1,
                fs_total=fs_total
            )
            if inf is None:
                continue
            s_sc0, s_sc1, s_base = inf

        # 再次确认 SC 容量（上面 helper 已经判过，但这里保守冗余一次）
        if _sum_sc_cap(new_P2MP_SC_1, a, sp, s_sc0, s_sc1) < band:
            continue

        ret = check_sender_p2mp_fs_reuse(
            a=a, sp=sp,
            fs_s_abs=fs_s_abs, fs_e_abs=fs_e_abs,
            s_base=s_base, new_P2MP_FS_1=new_P2MP_FS_1,
            path0_t=phy_path0_t, path1_t=phy_path1_t,
        )
        if ret is None:
            continue
        P2MP_reuse_fs, ref_flow_id, ref_orig_flow_id = ret

        if not check_link_fs_path_consistency(
            used_links_0based=used_links_0based,
            fs_s_abs=fs_s_abs, fs_e_abs=fs_e_abs,
            new_link_FS_meta=new_link_FS_meta,
            path0_t=phy_path0_t, path1_t=phy_path1_t,
        ):
            continue

        return int(sp), int(s_sc0), int(s_sc1), int(s_base), bool(P2MP_reuse_fs), ref_flow_id, ref_orig_flow_id

    return None

def _acc_get_src_hub(new_flow_acc: Any, subflow_id: int) -> Tuple[int, int]:
    """
    从 new_flow_acc 中读取“发送端节点 + 发送端 P2MP 索引”（用于从被复用的 subflow 反推 sender）。
    兼容常见格式：
    - src 在 [1]
    - hub(p2mp) 在 [7]
    """
    row = new_flow_acc[subflow_id]
    src = int(row[1])
    hub = int(row[7])
    return src, hub


def check_receiver_p2mp_fs_reuse(
    *,
    b: int,
    rp: int,
    fs_s_abs: int,
    fs_e_abs: int,
    r_base: int,
    new_P2MP_FS_1: Any,
    path0_t: Tuple[int, ...],
    path1_t: Tuple[int, ...],
) -> Optional[Tuple[bool, Tuple[int, ...], Tuple[Tuple[int, int], ...]]]:
    """
    在“固定接收端”场景中，用接收端 b 的 P2MP(rp) 先判断该 FS 段是否处于复用状态。

    判定方式与 sender 侧一致：
    - 只要该 FS 段中任一 FS 的 path_list 非空，则认为这是“复用 FS”场景；
    - 且要求 path_list 中必须存在与当前 hop 物理路径一致的路径（支持 0-based 或 1-based 存储）；
    - 同时收集 used_list 中记录的 subflow_id / orig_flow_id（你确认 used_list[0][0] 与 used_list[0][2]）。

    返回 (reuse_fs, ref_flow_id_tuple, ref_orig_flow_id_tuple)；失败返回 None。
    """
    if r_base < 0:
        return None

    reuse_fs = False
    ref_flow_ids: List[int] = []
    ref_orig_flow_ids: List[Tuple[int, int]] = []

    for fs_abs in range(int(fs_s_abs), int(fs_e_abs) + 1):
        fs_rel = int(fs_abs - r_base)
        try:
            path_list = new_P2MP_FS_1[b, rp, fs_rel, 2]
            used_list = new_P2MP_FS_1[b, rp, fs_rel, 1]
        except Exception:
            return None

        paths = list(_iter_paths(path_list))
        if paths:
            reuse_fs = True
            if not any(_path_match(pp, path1_t) for pp in paths):
                return None

            if used_list is not None and isinstance(used_list, list) and len(used_list) > 0:
                sub_id = _as_int(used_list[0][0], -1)
                if sub_id >= 0:
                    ref_flow_ids.append(sub_id)
                if len(used_list[0]) >= 3:
                    ref_orig_flow_ids.append((_as_int(used_list[0][2], sub_id), fs_abs))
                else:
                    ref_orig_flow_ids.append((sub_id, fs_abs))

    return reuse_fs, tuple(ref_flow_ids), tuple(ref_orig_flow_ids)


# ========================================
# 对外主接口
# ========================================

def try_assign_one_hop_s1(
    flow: Sequence[Any],
    a: int,
    b: int,
    phy_candidate: Dict[str, Any],
    flow_metadata_map: Dict[int, Dict[str, Any]],
    link_index: Any,
    new_link_FS: Any,
    new_node_P2MP: Any,
    new_P2MP_SC_1: Any,
    new_node_flow: Any,
    new_P2MP_FS_1: Any,
    new_link_FS_meta: Any,
    new_flow_acc: Any,
    hop_results: Optional[list] = None,
    *,
    strict_s1: bool = True,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """为单个 hop 试探分配资源；返回 (ok, plan_dict)。

说明：本函数不直接修改任何资源矩阵（纯“试探/选方案”）。若传入 hop_results，则会把 plan_dict 追加进去，
便于你在外层主循环里收集候选方案并做回退/比较。
"""
    f_id = _as_int(flow[0])
    src = _as_int(flow[1])
    dst = _as_int(flow[2])
    band = _as_int(flow[3])

    # 0-based physical path
    phy_path0 = list(phy_candidate.get("path", []))
    if not phy_path0 or len(phy_path0) < 2:
        return False, None

    phy_path0_t = tuple(int(x) for x in phy_path0)
    phy_path1_t = tuple(int(x) + 1 for x in phy_path0)

    # hop 使用的链路集合（0-based 的 link id）
    used_links_0based: List[int] = []
    for u, v in zip(phy_path0[:-1], phy_path0[1:]):
        lk = _as_int(link_index[u][v], 0)
        if lk <= 0:
            return False, None
        used_links_0based.append(lk - 1)

    fs_total = int(len(new_link_FS[0]))

    # 调制格式/SC 需求估计（主要用于 sanity check；接收端选择仍按 band 容量约束）
    phy_dist = phy_candidate.get("dist", phy_candidate.get("cost", None))
    if phy_dist is None:
        return False, None
    sc_cap_raw, sc_need_by_dist = modu_format_Al(phy_dist, band)
    _ = _norm_sc_cap(sc_cap_raw)
    sc_need = int(sc_need_by_dist)
    if sc_need <= 0 or sc_need > 16:
        return False, None

    # 端点固定的元信息（来自 flow_metadata_map）
    meta = flow_metadata_map.get(f_id, {})
    hub_idx = _as_int(meta.get("hub_idx", -1), -1)
    leaf_idx = _as_int(meta.get("leaf_idx", -1), -1)

    fixed_sender = bool(strict_s1 and (a == src) and (hub_idx != -1))
    fixed_receiver = bool(strict_s1 and (b == dst) and (leaf_idx != -1))

    fixed_sender_sc: Optional[Tuple[int, int, int]] = None  # (sp, sc0, sc1)
    fixed_receiver_sc: Optional[Tuple[int, int, int]] = None  # (rp, sc0, sc1)

    if fixed_sender:
        sp0 = hub_idx
        hub_sc = meta.get("hub_sc_range", None)
        # if hub_sc is None:
        #     s_sc0 = _as_int(meta.get("hub_sc_start"))
        #     s_sc1 = _as_int(meta.get("hub_sc_end"))
        # else:
        s_sc0, s_sc1 = _as_int(hub_sc[0]), _as_int(hub_sc[1])
        fixed_sender_sc = (sp0, s_sc0, s_sc1)

    if fixed_receiver:
        rp0 = leaf_idx
        leaf_sc = meta.get("leaf_sc_range", None)
        # if leaf_sc is None:
        #     r_sc0 = _as_int(meta.get("leaf_sc_start"))
        #     r_sc1 = _as_int(meta.get("leaf_sc_end"))
        # else:
        r_sc0, r_sc1 = _as_int(leaf_sc[0]), _as_int(leaf_sc[1])
        fixed_receiver_sc = (rp0, r_sc0, r_sc1)

    # 预计算每个 FS 在 hop 路径上的可行性（加速未激活 block 的 base 扫描）
    feasible_mask = _feasible_fs_mask_on_links(
        used_links_0based=used_links_0based,
        fs_total=fs_total,
        new_link_FS_meta=new_link_FS_meta,
        path0_t=phy_path0_t,
        path1_t=phy_path1_t,
    )

    def _build_plan(
        *,
        sp: int,
        s_sc0: int,
        s_sc1: int,
        s_base: int,
        rp: int,
        r_sc0: int,
        r_sc1: int,
        r_base: int,
        fs_s_abs: int,
        fs_e_abs: int,
        P2MP_reuse_fs: bool,
        ref_flow_id: Sequence[int],
        ref_orig_flow_id: Sequence[Tuple[int, int]],
        reuse_ctx: Optional[ReuseCtx],
    ) -> HopPlan:
        s_type = _as_int(new_node_P2MP[a][sp][3])
        r_type = _as_int(new_node_P2MP[b][rp][3])
        return HopPlan(
            f_id=f_id,
            a=int(a),
            b=int(b),
            phy_path0=phy_path0_t,
            used_links_0based=tuple(int(x) for x in used_links_0based),
            sp=int(sp),
            s_type=int(s_type),
            s_base=int(s_base),
            s_sc0=int(s_sc0),
            s_sc1=int(s_sc1),
            rp=int(rp),
            r_type=int(r_type),
            r_base=int(r_base),
            r_sc0=int(r_sc0),
            r_sc1=int(r_sc1),
            fs_s_abs=int(fs_s_abs),
            fs_e_abs=int(fs_e_abs),
            P2MP_reuse_fs=bool(P2MP_reuse_fs),
            ref_flow_id=tuple(int(x) for x in ref_flow_id),
            ref_orig_flow_id=tuple((int(x[0]), int(x[1])) for x in ref_orig_flow_id),
            reuse_ctx=reuse_ctx,
        )

    # ------------------------------------------------------------
    # Case A：固定发送端（接收端不固定）
    # ------------------------------------------------------------
    if fixed_sender_sc is not None and fixed_receiver_sc is None:
        sp, s_sc0, s_sc1 = fixed_sender_sc
        s_type = _as_int(new_node_P2MP[a][sp][3])
        s_base = _as_int(new_node_P2MP[a][sp][5], -1)
        if s_base < 0:
            return False, None

        fs_s_abs, fs_e_abs, _, _ = fs_abs_range(s_type, s_base, s_sc0, s_sc1)
        if fs_s_abs < 0 or fs_e_abs >= fs_total:
            return False, None
        if not _segment_all_true(feasible_mask, fs_s_abs, fs_e_abs):
            return False, None
        # if _sum_sc_cap(new_P2MP_SC_1, a, sp, s_sc0, s_sc1) < band:
        #     return False, None

        ret = check_sender_p2mp_fs_reuse(
            a=a,
            sp=sp,
            fs_s_abs=fs_s_abs,
            fs_e_abs=fs_e_abs,
            s_base=s_base,
            new_P2MP_FS_1=new_P2MP_FS_1,
            path0_t=phy_path0_t,
            path1_t=phy_path1_t,
        )
        if ret is None:
            return False, None
        P2MP_reuse_fs, ref_flow_id, ref_orig_flow_id = ret

        if not check_link_fs_path_consistency(
            used_links_0based=used_links_0based,
            fs_s_abs=fs_s_abs,
            fs_e_abs=fs_e_abs,
            new_link_FS_meta=new_link_FS_meta,
            path0_t=phy_path0_t,
            path1_t=phy_path1_t,
        ):
            return False, None

        chosen_leaf, reuse_ctx = choose_receiver_leaf_for_fs_segment(
            flow=flow,
            b=b,
            fs_s_abs=fs_s_abs,
            fs_e_abs=fs_e_abs,
            sc_need=sc_need,
            P2MP_reuse_fs=P2MP_reuse_fs,
            ref_flow_id=list(ref_flow_id),
            new_flow_acc=new_flow_acc,
            new_node_P2MP=new_node_P2MP,
            new_P2MP_SC_1=new_P2MP_SC_1,
            new_P2MP_FS_1=new_P2MP_FS_1,
            fs_abs_range=fs_abs_range,
            sc_fs=sc_fs,
            _fs_range_inside_p2mp_block=_fs_range_inside_p2mp_block,
        )
        if chosen_leaf is None:
            return False, None

        rp, r_sc0, r_sc1, r_base = chosen_leaf
        plan = _build_plan(
            sp=sp,
            s_sc0=s_sc0,
            s_sc1=s_sc1,
            s_base=s_base,
            rp=rp,
            r_sc0=r_sc0,
            r_sc1=r_sc1,
            r_base=r_base,
            fs_s_abs=fs_s_abs,
            fs_e_abs=fs_e_abs,
            P2MP_reuse_fs=P2MP_reuse_fs,
            ref_flow_id=ref_flow_id,
            ref_orig_flow_id=ref_orig_flow_id,
            reuse_ctx=reuse_ctx,
        )
        d = plan.to_dict()
        if hop_results is not None:
            hop_results.append(d)
        return True, d

    # ------------------------------------------------------------
        # Case B：固定接收端（与 Case A 对称，只是“固定端”在接收侧，因此需要反向选择发送侧）
    # ------------------------------------------------------------
    # Case B：固定接收端（目标节点 b 固定了 rp 与 [r_sc0,r_sc1]）
    #
    # 你希望它与 Case A 形式上一致：先在“固定端”判断是否复用，再由被复用的流反推另一端的 P2MP。
    # 因此这里的流程是：
    #   1) 用固定接收端推出绝对 FS 段 [fs_s_abs,fs_e_abs]，并先做链路可行性/路径一致性检查；
    #   2) 在接收端 b 的 rp 上检查该 FS 段是否已被复用：
    #        - 若复用：从 used_list 得到 ref_subflow_id，再用 new_flow_acc 反推出发送端 (a, sp)，
    #                 然后在 sp 上找能精确映射到该绝对 FS 段的 (s_sc0,s_sc1,s_base)；
    #        - 若不复用：则在发送端 a 上搜索一个**非复用**的 (sp,s_sc0,s_sc1,s_base) 使其映射到该绝对 FS 段。
    #   3) 最终返回包含 sender+receiver 的联合方案 plan。
    # ------------------------------------------------------------
    if fixed_sender_sc is None and fixed_receiver_sc is not None:
        rp, r_sc0, r_sc1 = fixed_receiver_sc
        r_type = _as_int(new_node_P2MP[b][rp][3])
        r_base0 = _as_int(new_node_P2MP[b][rp][5], -1)

        if r_base0 < 0:
            return False, None
        # if _sum_sc_cap(new_P2MP_SC_1, b, rp, r_sc0, r_sc1) < band:
        #     return False, None

        # 1) 固定接收端 => 推出绝对 FS 段
        fs_s_abs, fs_e_abs, _, _ = fs_abs_range(r_type, r_base0, r_sc0, r_sc1)
        if fs_s_abs < 0 or fs_e_abs >= fs_total:
            return False, None

        # 2) 链路可行性（按 path-consistency 预过滤）
        if not _segment_all_true(feasible_mask, fs_s_abs, fs_e_abs):
            return False, None
        if not check_link_fs_path_consistency(
                used_links_0based=used_links_0based,
                fs_s_abs=fs_s_abs, fs_e_abs=fs_e_abs,
                new_link_FS_meta=new_link_FS_meta,
                path0_t=phy_path0_t, path1_t=phy_path1_t,
        ):
            return False, None

        # 3) 先在“固定端”（接收端 rp）判断是否复用
        ret_r = check_receiver_p2mp_fs_reuse(
            b=b, rp=rp,
            fs_s_abs=fs_s_abs, fs_e_abs=fs_e_abs,
            r_base=r_base0,
            new_P2MP_FS_1=new_P2MP_FS_1,
            path0_t=phy_path0_t, path1_t=phy_path1_t,
        )
        if ret_r is None:
            return False, None
        recv_reuse_fs, recv_ref_flow_id, recv_ref_orig_flow_id = ret_r

        # 分支 1：接收端已复用 => 通过被复用 subflow 反推发送端 sp
        if recv_reuse_fs:
            if not recv_ref_flow_id:
                return False, None

            # 3.1 反推 sender：要求所有 ref subflow 对应的 src==a 且 hub_p2mp 一致
            sp_set = set()

            for sub_id in recv_ref_flow_id:
                row = new_flow_acc[int(sub_id)]
                src0 = _as_int(row[1])
                dst0 = _as_int(row[2])
                leaf0 = _as_int(row[8])

                # 必须确实是 (a -> b) 且 leaf 是固定 (b,rp)
                if src0 != a or dst0 != b or leaf0 != rp:
                    continue  # 注意：这里改成 continue 更稳，避免“混入垃圾 subflow”直接误杀

                src_hub = _acc_get_src_hub(new_flow_acc, int(sub_id))
                if src_hub[0] != a:
                    continue

                sp_set.add(int(src_hub[1]))

            # 如果一个都没收集到，说明 recv_ref_flow_id 虽然非空但没有可用的“参考 subflow”
            if not sp_set:
                return False, None

            # 严格：必须唯一
            if len(sp_set) != 1:
                return False, None

            sp = next(iter(sp_set))
            if sp < 0 or sp >= len(new_node_P2MP[a]):
                return False, None

            s_type = _as_int(new_node_P2MP[a][sp][3])
            s_base0 = _as_int(new_node_P2MP[a][sp][5], -1)


            # 3.2 在该 sender sp 上找能精确映射到 [fs_s_abs,fs_e_abs] 的 (s_sc0,s_sc1,s_base)
            if s_base0 >= 0:
                seg = _find_sc_segment_for_abs_fs(
                    u=a, p=sp, p_type=s_type, base_fs=s_base0,
                    fs_s_abs=fs_s_abs, fs_e_abs=fs_e_abs,
                    band=band, new_P2MP_SC_1=new_P2MP_SC_1
                )
                if seg is None:
                    return False, None
                s_sc0, s_sc1 = seg
                s_base = s_base0
            else:
                inf = _infer_base_and_sc_for_abs_fs(
                    u=a, p=sp, p_type=s_type,
                    fs_s_abs=fs_s_abs, fs_e_abs=fs_e_abs,
                    band=band, new_P2MP_SC_1=new_P2MP_SC_1,
                    fs_total=fs_total
                )
                if inf is None:
                    return False, None
                s_sc0, s_sc1, s_base = inf

            # sender SC 容量校验
            if _sum_sc_cap(new_P2MP_SC_1, a, sp, s_sc0, s_sc1) < band:
                return False, None

            # 3.3 sender 侧复用判定 + 路径一致性（注意：这一步会给出 sender 的 ref_flow，用于最终 plan 记录）
            ret_s = check_sender_p2mp_fs_reuse(
                a=a, sp=sp,
                fs_s_abs=fs_s_abs, fs_e_abs=fs_e_abs,
                s_base=s_base, new_P2MP_FS_1=new_P2MP_FS_1,
                path0_t=phy_path0_t, path1_t=phy_path1_t,
            )
            if ret_s is None:
                return False, None
            P2MP_reuse_fs, ref_flow_id, ref_orig_flow_id = ret_s

            # 在“接收端已复用”前提下，sender 侧应该也处于复用（否则语义不一致）
            if not P2MP_reuse_fs:
                return False, None

            reuse_ctx = ReuseCtx(ref_dst=int(b), ref_leaf_p2mp=int(rp),
                                 ref_subflows=tuple(int(x) for x in recv_ref_flow_id))

            plan = _build_plan(
                sp=sp, s_sc0=s_sc0, s_sc1=s_sc1, s_base=s_base,
                rp=rp, r_sc0=r_sc0, r_sc1=r_sc1, r_base=r_base0,
                fs_s_abs=fs_s_abs, fs_e_abs=fs_e_abs,
                P2MP_reuse_fs=P2MP_reuse_fs,
                ref_flow_id=ref_flow_id,
                ref_orig_flow_id=ref_orig_flow_id,
                reuse_ctx=reuse_ctx,
            )
            d = plan.to_dict()
            if hop_results is not None:
                hop_results.append(d)
            return True, d

        # 分支 2：接收端未复用 => 接收端该 FS 段必须无人使用；发送端也强制选择“非复用”候选
        else:
            if not _fs_segment_unused_on_node_p2mp(
                    node=b, p=rp, base_fs=r_base0,
                    fs_s_abs=fs_s_abs, fs_e_abs=fs_e_abs,
                    new_P2MP_FS_1=new_P2MP_FS_1
            ):
                return False, None

            # 在发送端选一个映射到该绝对 FS 段的方案，并且 must_non_reuse=True（避免 sender 走复用候选）
            ret_send = choose_sender_p2mp_for_abs_fs(
                flow=flow, a=a,
                fs_s_abs=fs_s_abs, fs_e_abs=fs_e_abs,
                feasible_mask=feasible_mask,
                used_links_0based=used_links_0based,
                phy_path0_t=phy_path0_t,
                phy_path1_t=phy_path1_t,
                new_node_P2MP=new_node_P2MP,
                new_P2MP_SC_1=new_P2MP_SC_1,
                new_P2MP_FS_1=new_P2MP_FS_1,
                new_link_FS_meta=new_link_FS_meta,
                fs_total=fs_total,
                must_non_reuse=True,
            )
            if ret_send is None:
                return False, None

            sp, s_sc0, s_sc1, s_base, P2MP_reuse_fs, ref_flow_id, ref_orig_flow_id = ret_send
            if P2MP_reuse_fs:
                # 理论上不会发生（因为 must_non_reuse=True），保险起见
                return False, None

            plan = _build_plan(
                sp=sp, s_sc0=s_sc0, s_sc1=s_sc1, s_base=s_base,
                rp=rp, r_sc0=r_sc0, r_sc1=r_sc1, r_base=r_base0,
                fs_s_abs=fs_s_abs, fs_e_abs=fs_e_abs,
                P2MP_reuse_fs=False,
                ref_flow_id=ref_flow_id,
                ref_orig_flow_id=ref_orig_flow_id,
                reuse_ctx=None,
            )
            d = plan.to_dict()
            if hop_results is not None:
                hop_results.append(d)
            return True, d

    # ------------------------------------------------------------
    # Case C：收发两端都固定（直接 src->dst hop）
    # ------------------------------------------------------------
    if fixed_sender_sc is not None and fixed_receiver_sc is not None:
        sp, s_sc0, s_sc1 = fixed_sender_sc
        rp, r_sc0, r_sc1 = fixed_receiver_sc

        s_type = _as_int(new_node_P2MP[a][sp][3])
        r_type = _as_int(new_node_P2MP[b][rp][3])
        s_base = _as_int(new_node_P2MP[a][sp][5], -1)
        r_base = _as_int(new_node_P2MP[b][rp][5], -1)
        if s_base < 0 or r_base < 0:
            return False, None

        fs_s_abs, fs_e_abs, _, _ = fs_abs_range(s_type, s_base, s_sc0, s_sc1)
        fs_s2, fs_e2, _, _ = fs_abs_range(r_type, r_base, r_sc0, r_sc1)
        if fs_s_abs != fs_s2 or fs_e_abs != fs_e2:
            return False, None
        if fs_s_abs < 0 or fs_e_abs >= fs_total:
            return False, None
        if not _segment_all_true(feasible_mask, fs_s_abs, fs_e_abs):
            return False, None

        # if _sum_sc_cap(new_P2MP_SC_1, a, sp, s_sc0, s_sc1) < band:
        #     return False, None
        # if _sum_sc_cap(new_P2MP_SC_1, b, rp, r_sc0, r_sc1) < band:
        #     return False, None

        if not check_link_fs_path_consistency(
            used_links_0based=used_links_0based,
            fs_s_abs=fs_s_abs,
            fs_e_abs=fs_e_abs,
            new_link_FS_meta=new_link_FS_meta,
            path0_t=phy_path0_t,
            path1_t=phy_path1_t,
        ):
            return False, None

        ret = check_sender_p2mp_fs_reuse(
            a=a,
            sp=sp,
            fs_s_abs=fs_s_abs,
            fs_e_abs=fs_e_abs,
            s_base=s_base,
            new_P2MP_FS_1=new_P2MP_FS_1,
            path0_t=phy_path0_t,
            path1_t=phy_path1_t,
        )
        if ret is None:
            return False, None
        P2MP_reuse_fs, ref_flow_id, ref_orig_flow_id = ret

        reuse_ctx: Optional[ReuseCtx] = None
        if P2MP_reuse_fs:
            if not ref_flow_id:
                return False, None
            for sub_id in ref_flow_id:
                row = new_flow_acc[int(sub_id)]
                dst0 = _as_int(row[2])
                leaf0 = _as_int(row[8])
                if dst0 != b or leaf0 != rp:
                    return False, None
            reuse_ctx = ReuseCtx(ref_dst=int(b), ref_leaf_p2mp=int(rp), ref_subflows=tuple(int(x) for x in ref_flow_id))
        else:
            if not _fs_segment_unused_on_node_p2mp(
                node=b, p=rp, base_fs=r_base,
                fs_s_abs=fs_s_abs, fs_e_abs=fs_e_abs,
                new_P2MP_FS_1=new_P2MP_FS_1
            ):
                return False, None

        plan = _build_plan(
            sp=sp, s_sc0=s_sc0, s_sc1=s_sc1, s_base=s_base,
            rp=rp, r_sc0=r_sc0, r_sc1=r_sc1, r_base=r_base,
            fs_s_abs=fs_s_abs, fs_e_abs=fs_e_abs,
            P2MP_reuse_fs=P2MP_reuse_fs,
            ref_flow_id=ref_flow_id,
            ref_orig_flow_id=ref_orig_flow_id,
            reuse_ctx=reuse_ctx,
        )
        d = plan.to_dict()
        if hop_results is not None:
            hop_results.append(d)
        return True, d

    # ------------------------------------------------------------
    # Case D：中转 hop（两端都不固定）
    # ------------------------------------------------------------
    # Strategy:
    #   1) 枚举发送端 block（优先 base 已知，其次 base 未知）
    #   2) 对每个 (sp, base, sc区间) 推导绝对 FS 段，并要求链路上可行
    #   3) 强制发送端 P2MP 复用逻辑与链路路径一致性
    #   4) 调用 assign_leaf 选择一个可行的接收端 block
    for sp in _sender_candidates(a, new_node_P2MP):
        s_type = _as_int(new_node_P2MP[a][sp][3])
        s_base0 = _as_int(new_node_P2MP[a][sp][5], -1)
        max_rel = _max_rel_fs(s_type)

        if s_base0 >= 0:
            base_candidates = [s_base0]
        else:
            # 扫描 base：保证 block 覆盖范围不越界
            base_candidates = list(range(0, fs_total - max_rel))

        for s_base in base_candidates:
            # 可选剪枝：若 base 未知，可跳过已被 feasible_mask 判为不可行的 base
            # 但对每个 SC 区间仍需精确检查其映射得到的绝对 FS 段
            for s_sc0, s_sc1 in SC_SEGMENTS_BY_SPAN:
                fs_s_abs, fs_e_abs, _, _ = fs_abs_range(s_type, s_base, s_sc0, s_sc1)
                if fs_s_abs < 0 or fs_e_abs >= fs_total:
                    continue
                if not _segment_all_true(feasible_mask, fs_s_abs, fs_e_abs):
                    continue
                if _sum_sc_cap(new_P2MP_SC_1, a, sp, s_sc0, s_sc1) < band:
                    continue

                ret = check_sender_p2mp_fs_reuse(
                    a=a,
                    sp=sp,
                    fs_s_abs=fs_s_abs,
                    fs_e_abs=fs_e_abs,
                    s_base=s_base,
                    new_P2MP_FS_1=new_P2MP_FS_1,
                    path0_t=phy_path0_t,
                    path1_t=phy_path1_t,
                )
                if ret is None:
                    continue
                P2MP_reuse_fs, ref_flow_id, ref_orig_flow_id = ret

                if not check_link_fs_path_consistency(
                    used_links_0based=used_links_0based,
                    fs_s_abs=fs_s_abs,
                    fs_e_abs=fs_e_abs,
                    new_link_FS_meta=new_link_FS_meta,
                    path0_t=phy_path0_t,
                    path1_t=phy_path1_t,
                ):
                    continue

                chosen_leaf, reuse_ctx = choose_receiver_leaf_for_fs_segment(
                    flow=flow,
                    b=b,
                    fs_s_abs=fs_s_abs,
                    fs_e_abs=fs_e_abs,
                    sc_need=sc_need,
                    P2MP_reuse_fs=P2MP_reuse_fs,
                    ref_flow_id=list(ref_flow_id),
                    new_flow_acc=new_flow_acc,
                    new_node_P2MP=new_node_P2MP,
                    new_P2MP_SC_1=new_P2MP_SC_1,
                    new_P2MP_FS_1=new_P2MP_FS_1,
                    fs_abs_range=fs_abs_range,
                    sc_fs=sc_fs,
                    _fs_range_inside_p2mp_block=_fs_range_inside_p2mp_block,
                )
                if chosen_leaf is None:
                    continue
                rp, r_sc0, r_sc1, r_base = chosen_leaf

                plan = _build_plan(
                    sp=sp, s_sc0=s_sc0, s_sc1=s_sc1, s_base=s_base,
                    rp=rp, r_sc0=r_sc0, r_sc1=r_sc1, r_base=r_base,
                    fs_s_abs=fs_s_abs, fs_e_abs=fs_e_abs,
                    P2MP_reuse_fs=P2MP_reuse_fs,
                    ref_flow_id=ref_flow_id,
                    ref_orig_flow_id=ref_orig_flow_id,
                    reuse_ctx=reuse_ctx,
                )
                d = plan.to_dict()
                if hop_results is not None:
                    hop_results.append(d)
                return True, d

    return False, None
