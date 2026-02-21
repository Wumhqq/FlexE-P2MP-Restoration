from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import topology as tp
from k_shortest_path import k_shortest_path
from modu_format_Al import modu_format_Al

from Initial_net import (
    _TYPE_INFO,
    _apply_leaf_usage,
    _apply_p2mp_fs_usage,
    _clear_p2mp_sc,
    _find_free_fs_block,
    _plan_sc_allocation,
)
from other_fx_fix import (
    build_virtual_topology,
    extract_old_flow_info,
    manage_s1_reservation_by_copy,
)


def logical_path_is_valid(path_1b: List[int], break_node_0b: int) -> bool:
    """验证逻辑路径：非空、不含中断节点、无重复节点。"""
    if not path_1b or len(path_1b) < 2:
        return False
    if (break_node_0b + 1) in path_1b:
        return False
    return len(set(path_1b)) == len(path_1b)


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


def _build_subflow_lookup(new_flow_acc: np.ndarray) -> Dict[Tuple[int, int, int], int]:
    """建立 (原始流ID, hop_a, hop_b) 到子流ID的映射。"""
    m: Dict[Tuple[int, int, int], int] = {}
    for row in new_flow_acc:
        sub_id = int(row[0])
        a = int(row[1])
        b = int(row[2])
        orig = int(row[6])
        m[(orig, a, b)] = sub_id
    return m


def _assign_one_hop_with_initialnet(
    flow: Any,
    a: int,
    b: int,
    phy_path0: List[int],
    phy_dist: float,
    link_index: np.ndarray,
    strict_s1: bool,
    meta: Dict[str, Any],
    subflow_lookup: Dict[Tuple[int, int, int], int],
    new_flow_acc: np.ndarray,
    new_node_flow: np.ndarray,
    new_node_P2MP: np.ndarray,
    new_P2MP_SC: np.ndarray,
    new_link_FS: np.ndarray,
    new_P2MP_FS: np.ndarray,
    new_flow_path: np.ndarray,
) -> bool:
    """按 Initial_net 的约束为单个逻辑跳分配资源。"""
    if not phy_path0 or len(phy_path0) < 2:
        return False

    fid = int(flow[0])
    band = float(flow[3])
    sub_id = subflow_lookup.get((fid, a, b), None)
    if sub_id is None:
        return False
    sub_id = int(sub_id)
    orig_fid = int(new_flow_acc[sub_id][6]) if int(new_flow_acc[sub_id][6]) >= 0 else fid

    used_links: List[int] = []
    for u, v in zip(phy_path0[:-1], phy_path0[1:]):
        lk = int(link_index[u][v])
        if lk <= 0:
            return False
        used_links.append(lk - 1)

    phys_nodes_1b = [int(x) + 1 for x in phy_path0]
    sc_cap, _ = modu_format_Al(phy_dist, band)

    fixed_hub = None
    fixed_leaf = None
    if strict_s1:
        if int(flow[1]) == a and int(meta.get("hub_idx", -1)) != -1:
            fixed_hub = int(meta["hub_idx"])
        if int(flow[2]) == b and int(meta.get("leaf_idx", -1)) != -1:
            fixed_leaf = int(meta["leaf_idx"])

    _p2mp_total = new_node_P2MP.shape[1]
    hub_candidates = [fixed_hub] if fixed_hub is not None else list(range(_p2mp_total))

    for hub_p in hub_candidates:
        if hub_p is None:
            continue
        if int(new_node_P2MP[a][hub_p][2]) != 1:
            continue
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
            new_node_P2MP, new_node_flow, new_link_FS, new_P2MP_FS, used_links, new_flow_path
        )
        if plan is None:
            continue
        sc_s, sc_e, usage, fs_s_glo, fs_e_glo, leaf_p, leaf_sc_s, leaf_sc_e, leaf_usage = plan
        if fixed_leaf is not None and int(leaf_p) != int(fixed_leaf):
            continue
        _commit_plan(
            sub_id, orig_fid, a, b, band, sc_cap, hub_p, hub_type, leaf_p,
            sc_s, sc_e, fs_s_glo, fs_e_glo, leaf_sc_s, leaf_sc_e,
            usage, leaf_usage, phys_nodes_1b, used_links,
            new_flow_acc, new_node_flow, new_P2MP_SC, new_link_FS, new_P2MP_FS, new_flow_path
        )
        return True

    for hub_p in hub_candidates:
        if hub_p is None:
            continue
        if int(new_node_P2MP[a][hub_p][2]) != 0:
            continue
        hub_type = int(new_node_P2MP[a][hub_p][3])
        block_size = _TYPE_INFO[int(hub_type)][1]
        hub_fs0 = _find_free_fs_block(new_link_FS, used_links, block_size)
        if hub_fs0 is None:
            continue
        new_node_P2MP[a][hub_p][2] = 1
        new_node_P2MP[a][hub_p][5] = int(hub_fs0)
        new_node_flow[a][hub_p][0].clear()
        _clear_p2mp_sc(new_P2MP_SC, a, hub_p)

        flow_ids_on_hub = new_node_flow[a][hub_p][0]
        plan = _plan_sc_allocation(
            a, hub_p, hub_type, int(hub_fs0), b,
            phys_nodes_1b, band, sc_cap, flow_ids_on_hub, new_flow_acc, new_P2MP_SC,
            new_node_P2MP, new_node_flow, new_link_FS, new_P2MP_FS, used_links, new_flow_path
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
            sub_id, orig_fid, a, b, band, sc_cap, hub_p, hub_type, leaf_p,
            sc_s, sc_e, fs_s_glo, fs_e_glo, leaf_sc_s, leaf_sc_e,
            usage, leaf_usage, phys_nodes_1b, used_links,
            new_flow_acc, new_node_flow, new_P2MP_SC, new_link_FS, new_P2MP_FS, new_flow_path
        )
        return True

    return False


def _commit_plan(
    sub_id: int,
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
    new_P2MP_SC: np.ndarray,
    new_link_FS: np.ndarray,
    new_P2MP_FS: np.ndarray,
    new_flow_path: np.ndarray,
) -> None:
    """将计划的 SC/FS 分配写入全部状态结构。"""
    sc_num = int(sc_e) - int(sc_s) + 1
    subflow_info = [int(sub_id), int(a), int(b), float(band), float(sc_cap), int(sc_num), int(orig_fid)]
    new_node_flow[a][hub_p][0].append(subflow_info)
    new_node_flow[b][leaf_p][0].append(subflow_info)

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


def _attempt_restore_flow(
    flow: Any,
    break_node: int,
    v_adj: np.ndarray,
    v_phy: Dict[Tuple[int, int], Dict[str, Any]],
    K_LOGICAL_PATHS: int,
    flow_metadata_map: Dict[int, Dict[str, Any]],
    link_index: np.ndarray,
    strict_s1: bool,
    subflow_lookup: Dict[Tuple[int, int, int], int],
    state: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[bool, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """枚举逻辑路径并逐跳提交资源以尝试恢复该流。"""
    src = int(flow[1])
    dst = int(flow[2])

    try:
        logical_paths = k_shortest_path(v_adj, src + 1, dst + 1, K_LOGICAL_PATHS)
    except Exception:
        logical_paths = []

    snap = tuple(copy.deepcopy(x) for x in state)
    meta = flow_metadata_map.get(int(flow[0]), {})

    for lp_nodes_1b, _ in logical_paths:
        if not logical_path_is_valid(lp_nodes_1b, break_node):
            continue
        lp0 = [x - 1 for x in lp_nodes_1b]

        work_flow_acc, work_node_flow, work_node_P2MP, work_P2MP_SC, work_link_FS, work_P2MP_FS, work_flow_path = copy.deepcopy(snap)

        ok_path = True
        for a, b in zip(lp0[:-1], lp0[1:]):
            if (a, b) not in v_phy:
                ok_path = False
                break

            phy_cand = v_phy[(a, b)]
            phy_path0 = list(phy_cand["path"])
            phy_dist = float(phy_cand["dist"])

            ok_hop = _assign_one_hop_with_initialnet(
                flow=flow,
                a=a,
                b=b,
                phy_path0=phy_path0,
                phy_dist=phy_dist,
                link_index=link_index,
                strict_s1=strict_s1,
                meta=meta,
                subflow_lookup=subflow_lookup,
                new_flow_acc=work_flow_acc,
                new_node_flow=work_node_flow,
                new_node_P2MP=work_node_P2MP,
                new_P2MP_SC=work_P2MP_SC,
                new_link_FS=work_link_FS,
                new_P2MP_FS=work_P2MP_FS,
                new_flow_path=work_flow_path,
            )
            if not ok_hop:
                ok_path = False
                break

        if ok_path:
            return True, (work_flow_acc, work_node_flow, work_node_P2MP, work_P2MP_SC, work_link_FS, work_P2MP_FS, work_flow_path)

    return False, state


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
):
    """策略1优先的恢复流程，并复用 Initial_net 资源分配约束。"""
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

    # 从 DP 状态提取原有流的历史资源信息，并建立子流索引
    flow_metadata_map = _build_flow_metadata_map(affected_flow, flow_acc_base, P2MP_SC_base)
    subflow_lookup = _build_subflow_lookup(new_flow_acc)

    # 从 DP 复制 flow_path，避免直接修改输入引用
    new_flow_path = copy.deepcopy(flow_path_base)

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

        # 快速判断策略1下是否连通
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

        # 策略2：放宽约束，允许硬件/格式重构
        v_adj_s2, v_phy_s2 = build_virtual_topology(
            src, dst, topo_num, phy_pool, break_node, new_node_P2MP, band,
            False, src_modu, dst_modu, RECONFIG_PENALTY, HOP_PENALTY
        )
        flow_vmap_s2[fid] = (v_adj_s2, v_phy_s2)

    # 先对 Tier1 的端口/SC 进行占位，避免资源被后续流抢占
    manage_s1_reservation_by_copy(
        tier1_flows, flow_metadata_map,
        flow_acc_base, node_P2MP_base, P2MP_SC_base,
        new_node_P2MP, new_P2MP_SC,
        action="reserve"
    )

    tier1_restored, tier1_failed = [], []

    # 先尝试策略1恢复：逐流回滚本流占位，再进行路径搜索与逐跳分配
    for f in tier1_flows:
        fid = int(f[0])

        # 对当前流撤销占位，避免影响真实分配
        manage_s1_reservation_by_copy(
            [f], flow_metadata_map,
            flow_acc_base, node_P2MP_base, P2MP_SC_base,
            new_node_P2MP, new_P2MP_SC,
            action="rollback"
        )

        v_adj, v_phy = flow_vmap_s1[fid]
        # 尝试在策略1约束下恢复：多条逻辑路径逐条尝试，成功即提交
        restored, new_state = _attempt_restore_flow(
            flow=f,
            break_node=break_node,
            v_adj=v_adj,
            v_phy=v_phy,
            K_LOGICAL_PATHS=K_LOGICAL_PATHS,
            flow_metadata_map=flow_metadata_map,
            link_index=link_index,
            strict_s1=True,
            subflow_lookup=subflow_lookup,
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

    # 再尝试策略2恢复：允许重构，逻辑路径搜索与逐跳分配同样走统一流程
    for f in tier2_flows:
        fid = int(f[0])

        v_adj, v_phy = flow_vmap_s2[fid]
        restored, new_state = _attempt_restore_flow(
            flow=f,
            break_node=break_node,
            v_adj=v_adj,
            v_phy=v_phy,
            K_LOGICAL_PATHS=K_LOGICAL_PATHS,
            flow_metadata_map=flow_metadata_map,
            link_index=link_index,
            strict_s1=False,
            subflow_lookup=subflow_lookup,
            state=(new_flow_acc, new_node_flow, new_node_P2MP, new_P2MP_SC, new_link_FS, new_P2MP_FS, new_flow_path),
        )
        if restored:
            new_flow_acc, new_node_flow, new_node_P2MP, new_P2MP_SC, new_link_FS, new_P2MP_FS, new_flow_path = new_state

        if restored:
            tier2_restored.append(f)
        else:
            tier2_failed.append(f)

    # 返回恢复后的资源状态
    return new_node_flow, new_node_P2MP, new_flow_acc, new_link_FS, new_P2MP_SC, new_P2MP_FS, new_flow_path
