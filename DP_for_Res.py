#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import math
import numpy as np

from k_shortest_path import k_shortest_path
from Path_Link import get_links_from_path
from modu_format_Al import modu_format_Al
from Initial_net import allocate_flows_sequential, sc_effective_cap, sc_num_from_bw_cap, TS_UNIT, _clear_p2mp_sc, _apply_p2mp_fs_usage, _plan_sc_allocation, _apply_leaf_usage, _TYPE_INFO
from knapsack_DP import knapsack_DP
from SC_FS import sc_fs

# 将受影响的原始流转为 DP_for_Res 所需的 flows_info 结构
def _build_restore_inputs(affected_flow: list, topo_dis: np.ndarray, new_flow_acc: np.ndarray, new_flow_path: np.ndarray):
    # 生成恢复流的基础字段（不拆分子流）
    affected_flows_info = []
    next_id = int(new_flow_acc.shape[0])
    for f in affected_flow:
        orig_id = int(f[0])
        src = int(f[1])
        dst = int(f[2])
        bw = float(f[3])
        # 基于物理路径累计距离，重新计算调制与 SC 需求
        path = k_shortest_path(topo_dis, src + 1, dst + 1, 1)
        dist = float(path[0][1])
        sc_cap_raw, _ = modu_format_Al(dist, bw)
        sc_cap = sc_effective_cap(float(sc_cap_raw))
        sc_num = sc_num_from_bw_cap(bw, sc_cap)
        item = [int(next_id), src, dst, float(bw), float(sc_cap), int(sc_num), int(orig_id)]
        affected_flows_info.append(item)
        next_id += 1

    add_n = len(affected_flows_info)
    if add_n > 0:
        # 扩容 flow_acc / flow_path 以容纳新增恢复流
        flow_acc_ext = np.zeros((add_n, new_flow_acc.shape[1]), dtype=object)
        for i in range(add_n):
            flow_acc_ext[i][7:15] = [-1] * 8
        new_flow_acc = np.vstack([new_flow_acc, flow_acc_ext])

        # 初始化新增流的路径记录
        flow_path_ext = np.empty((add_n, new_flow_path.shape[1]), dtype=object)
        for i in range(add_n):
            for c in range(new_flow_path.shape[1]):
                flow_path_ext[i][c] = []
        new_flow_path = np.vstack([new_flow_path, flow_path_ext])

        # 写入基础字段并清空路径
        for item in affected_flows_info:
            sid = int(item[0])
            new_flow_acc[sid][0:7] = item
            new_flow_path[sid][0] = []
            new_flow_path[sid][1] = []
            new_flow_path[sid][2] = []
            new_flow_path[sid][3] = []

    return affected_flows_info, new_flow_acc, new_flow_path

def restore_with_dp(
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
    affected_flows_info, flow_acc, flow_path = _build_restore_inputs(
        affected_flow,
        topo_dis,
        flow_acc,
        flow_path,
    )
    flows_num = len(affected_flows_info)
    return FlexE_P2MP_DP(
        affected_flows_info,
        flows_num,
        topo_num,
        topo_dis,
        link_num,
        link_index,
        Tbox_num,
        Tbox_P2MP,
        node_flow,
        node_P2MP,
        link_FS,
        P2MP_SC,
        P2MP_FS,
        flow_acc,
        flow_path,
        break_node,
    )

# 基于故障后资源状态的 DP 恢复入口（源节点维度 + 连续FS块 + SC段背包）
def FlexE_P2MP_DP(
    flows_info,
    flows_num,
    topo_num,
    topo_dis,
    link_num,
    link_index,
    Tbox_num,
    Tbox_P2MP,
    node_flow,
    node_P2MP,
    link_FS,
    P2MP_SC,
    P2MP_FS,
    flow_acc,
    flow_path,
    break_node=None
):
    type_P2MP = [[k, int(_TYPE_INFO[k][0]), int(_TYPE_INFO[k][1])] for k in sorted(_TYPE_INFO.keys())]
    _p2mp_total = int(node_P2MP.shape[1])
    scr_flows = np.empty(topo_num, dtype=object)
    for n in range(topo_num):
        scr_flows[n] = []
        for flow in flows_info:
            if int(flow[1]) == n:
                scr_flows[n].append(flow[0:7])

    # 在路径的所有链路上搜索可用的连续 FS 块
    def _find_free_fs_block_on_links(link_FS, used_links, block_size):
        if not used_links:
            combined_usage = link_FS.sum(axis=0) * 0
        else:
            combined_usage = link_FS[used_links].sum(axis=0)
        for i0 in range(len(combined_usage) - int(block_size) + 1):
            if np.all(combined_usage[i0:i0 + int(block_size)] == 0):
                return int(i0)
        return None

    # 判断该端口内某段相对 FS 是否完全空闲
    def _fs_rel_segment_empty(P2MP_FS, u, p, fs_rel_s, fs_rel_e):
        for fs_rel in range(int(fs_rel_s), int(fs_rel_e) + 1):
            try:
                used_list = P2MP_FS[u][p][int(fs_rel)][1]
                path_list = P2MP_FS[u][p][int(fs_rel)][2]
            except Exception:
                return False
            if used_list or path_list:
                return False
        return True

    # 按源节点逐个恢复
    for n in range(topo_num):
        # 以源节点为单位进行恢复
        # 1. 将待恢复流按 (目的节点, 路径) 分组
        # 预计算每条流的 SC 需求 (基于路径距离)
        # 并在分组时直接存储
        group_map = {}
        
        # 暂存无法通过 P2MP 恢复的流，稍后尝试逐条恢复
        rest_flows = []
        
        # 初始分组
        # 基于最短物理路径计算 SC 需求并形成分组
        for f in scr_flows[n]:
            des = int(f[2])
            bw = float(f[3])
            # 重新计算路径 (K=1)
            path = k_shortest_path(topo_dis, n + 1, des + 1, 1)
            phys_nodes = path[0][0]
            dist = float(path[0][1])
            used_links = get_links_from_path(phys_nodes, link_index)
            used_links = [int(x) - 1 for x in used_links]
            
            # 计算 SC 需求
            sc_cap_raw, _ = modu_format_Al(dist, 0)
            sc_eff = sc_effective_cap(sc_cap_raw)
            sc_needed = sc_num_from_bw_cap(bw, sc_eff)
            
            key = (des, tuple(phys_nodes))
            if key not in group_map:
                group_map[key] = {
                    "flows": [],
                    "phys_nodes": phys_nodes,
                    "used_links": used_links,
                    "sc_eff": sc_eff,  # 该路径下的单 SC 有效容量
                    "sc_cap_raw": float(sc_cap_raw)
                }
            # 存入流信息，增加 sc_needed
            # f 结构: [id, src, dst, bw, sc_cap, sc_num, orig_id]
            # 我们需要构造 knapsack item: [id, sc_needed, bw, ...]
            # 这里直接把 f 和 sc_needed 打包
            group_map[key]["flows"].append({
                "original": f,
                "sc_needed": sc_needed,
                "bw": bw
            })

        # 恢复策略：
        # 1) 先按 (目的节点, 物理路径) 的分组逐个处理，保证同组流优先复用同一资源语义；
        # 2) 对每个分组再遍历源端口，尝试在该端口上打包一批流并立即做 SC/FS 可行性分配；
        # 3) 成功分配的流立即从分组中删除，失败流留在组内继续尝试其它端口。
        for key, group in group_map.items():
            if not group["flows"]:
                continue
            # 分组级固定信息：同组流共享目的节点、物理路径、链路集合与单SC容量口径
            des = int(key[0])
            phys_nodes = group["phys_nodes"]
            used_links = group["used_links"]
            sc_cap_raw = float(group["sc_cap_raw"])

            # 在该分组上依次尝试所有源端口
            for src_p in range(_p2mp_total):
                # 本组流已全部恢复完成则提前结束该组端口遍历
                if not group["flows"]:
                    break
                # 读取端口类型与其FS块大小（用于base_fs校验/搜索）
                t = int(node_P2MP[n][src_p][3])
                spec = type_P2MP[t - 1]
                fs_size = int(spec[2])

                # 确定该端口用于本组的 FS 基址：
                # - 若端口已有基址，直接复用；
                # - 否则在该分组涉及链路上搜索可用连续FS块。
                base_fs = int(node_P2MP[n][src_p][5])
                if base_fs >= 0:
                    fs_start = base_fs
                else:
                    fs_start = _find_free_fs_block_on_links(link_FS, used_links, fs_size)
                    if fs_start is None:
                        continue

                # 单层背包容量：端口剩余接收容量
                cap_bw = int(max(0, int(node_P2MP[n][src_p][4])))
                if cap_bw <= 0:
                    continue

                # 背包物品：[流ID, 带宽]，由 knapsack_DP 按第二列做重量/价值
                items_for_group = []
                for flow_obj in group["flows"]:
                    fid = int(flow_obj["original"][0])
                    bw_int = int(round(float(flow_obj["bw"])))
                    items_for_group.append([fid, bw_int])
                if not items_for_group:
                    continue

                # 先在容量维度选出一批候选流，随后对这批流逐条做SC/FS真实分配
                _, _, _, group_items = knapsack_DP(int(cap_bw), items_for_group)
                if not group_items:
                    continue
                selected_ids = set(int(x) for x in group_items)
                selected_flow_objs = [fo for fo in group["flows"] if int(fo["original"][0]) in selected_ids]
                if not selected_flow_objs:
                    continue

                # 端口准备策略：
                # - 端口已有流：增量写入，保留历史SC记录；
                # - 端口无流：作为候选新启用端口，先清空本端口SC容器；
                # - 端口使用标记在“本轮至少成功放置一条流”后再置为1。
                original_base = int(node_P2MP[n][src_p][5])
                had_existing_flows = len(node_flow[n][src_p][0]) > 0
                if original_base < 0:
                    node_P2MP[n][src_p][5] = int(fs_start)
                if not had_existing_flows:
                    node_flow[n][src_p][0].clear()
                    _clear_p2mp_sc(P2MP_SC, n, src_p)

                placed_any = False
                placed_ids = set()
                # 对背包选中的候选流逐条做规划与提交：
                # - plan成功才提交；
                # - 提交后记录fid，用于从分组中剔除。
                for f_obj in selected_flow_objs:
                    f = f_obj["original"]
                    fid = int(f[0])
                    bw = float(f[3])
                    # _plan_sc_allocation 负责核心可行性搜索：
                    # 包括 hub/leaf 的SC区间、FS范围、路径一致性、链路占用约束等。
                    plan = _plan_sc_allocation(
                        n, src_p, t, int(fs_start), des,
                        phys_nodes, bw, sc_cap_raw, node_flow[n][src_p][0], flow_acc, P2MP_SC,
                        node_P2MP, node_flow, link_FS, P2MP_FS, used_links, flow_path
                    )
                    if plan is None:
                        continue

                    sc_s, sc_e, usage, fs_s_glo, fs_e_glo, leaf_p, leaf_sc_s, leaf_sc_e, leaf_usage = plan
                    # 提交前二次保护：确认相对FS段仍为空闲，避免状态漂移导致冲突
                    fs_rel_s = int(fs_s_glo) - int(fs_start)
                    fs_rel_e = int(fs_e_glo) - int(fs_start)
                    if not _fs_rel_segment_empty(P2MP_FS, n, src_p, fs_rel_s, fs_rel_e):
                        continue

                    # 提交1：双端端口承载关系（hub/leaf）写入
                    node_flow[n][src_p][0].append(f)
                    node_flow[des][leaf_p][0].append(f)

                    # 提交2：flow_acc 记录该流最终端口、SC区间和FS区间
                    flow_acc[fid][7] = int(src_p)
                    flow_acc[fid][8] = int(leaf_p)
                    flow_acc[fid][9] = int(sc_s)
                    flow_acc[fid][10] = int(sc_e)
                    flow_acc[fid][11] = int(fs_s_glo)
                    flow_acc[fid][12] = int(fs_e_glo)
                    flow_acc[fid][13] = int(leaf_sc_s)
                    flow_acc[fid][14] = int(leaf_sc_e)

                    # 提交3：写hub侧SC占用与其映射到P2MP_FS的占用信息
                    for sc, used_amt in usage.items():
                        P2MP_SC[n][src_p][int(sc)][1].append([fid, float(used_amt), f[6]])
                        P2MP_SC[n][src_p][int(sc)][0].append(float(sc_cap_raw))
                        P2MP_SC[n][src_p][int(sc)][2].append(phys_nodes)
                        P2MP_SC[n][src_p][int(sc)][4].append(int(des))
                        P2MP_SC[n][src_p][int(sc)][5].append(int(n))
                        P2MP_SC[n][src_p][int(sc)][6].append(int(src_p))
                        P2MP_SC[n][src_p][int(sc)][7].append(int(des))
                        P2MP_SC[n][src_p][int(sc)][8].append(int(leaf_p))
                        _apply_p2mp_fs_usage(
                            P2MP_FS, n, src_p, int(t), int(sc),
                            fid, float(used_amt), phys_nodes, des, f[6],
                            n, src_p, des, leaf_p
                        )

                    # 提交4：写流路径记录（用于后续释放、可视化和冲突检查）
                    flow_path[fid][0].append(fid)
                    flow_path[fid][1].extend(used_links)
                    flow_path[fid][2].extend(phys_nodes)
                    flow_path[fid][3].append(f[6])

                    # 提交5：写leaf侧SC使用并同步leaf侧P2MP_FS使用
                    _apply_leaf_usage(
                        P2MP_SC, des, leaf_p,
                        int(leaf_sc_s), int(leaf_sc_e),
                        leaf_usage, float(sc_cap_raw),
                        fid, phys_nodes, des, f[6],
                        n, src_p, des, leaf_p,
                    )
                    leaf_type = int(node_P2MP[des][leaf_p][3])
                    for sc, used_amt in leaf_usage.items():
                        _apply_p2mp_fs_usage(
                            P2MP_FS, des, leaf_p, int(leaf_type), int(sc),
                            fid, float(used_amt), phys_nodes, des, f[6],
                            n, src_p, des, leaf_p
                        )

                    # 提交6：把路径上所有链路对应FS区间占用计数 +1
                    for l in set(used_links):
                        link_FS[int(l)][int(fs_s_glo):int(fs_e_glo) + 1] += 1

                    placed_any = True
                    placed_ids.add(fid)

                # 仅删除成功分配 SC 的流
                # 未成功的流保留在当前分组中，后续还可在其它端口再次尝试
                if placed_ids:
                    group["flows"] = [
                        fo for fo in group["flows"]
                        if int(fo["original"][0]) not in placed_ids
                    ]

                # 本轮至少放置一条流才把端口标记为已使用
                if placed_any:
                    node_P2MP[n][src_p][2] = 1

                # 若本轮端口未放下任何流且该端口是本次新启用，则回滚到进入前状态
                if not placed_any and not had_existing_flows:
                    node_P2MP[n][src_p][2] = 0
                    if original_base < 0:
                        node_P2MP[n][src_p][5] = -1
                    node_flow[n][src_p][0].clear()
                    _clear_p2mp_sc(P2MP_SC, n, src_p)

        # 收集未处理流，进入逐条恢复兜底（顺序分配）
        for group in group_map.values():
            for f_obj in group["flows"]:
                rest_flows.append(f_obj["original"])

        if rest_flows:
            node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, P2MP_FS, flow_path, _ = allocate_flows_sequential(
                flows_info=np.array(rest_flows, dtype=object),
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
                stop_on_fail=False,
            )

    return node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, flow_path
