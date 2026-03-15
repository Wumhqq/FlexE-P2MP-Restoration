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
def _build_restore_inputs(
    affected_flow: list,
    topo_dis: np.ndarray,
    new_flow_acc: np.ndarray,
    new_flow_path: np.ndarray,
    break_node: int,
):
    """
    为“故障后的恢复阶段”构造新的待恢复子流列表。

    设计目标：
    1. 对 affected_flow 中的原始流做去重；
    2. 过滤掉以 break_node 为端点、或在故障后拓扑中不可达的流；
    3. 为可恢复流重新计算恢复路径长度、SC 有效容量、所需 SC 数；
    4. 扩容 flow_acc / flow_path，为这些“恢复子流”预留记录位置；
    5. 返回：
       - affected_flows_info : 传给 DP 恢复器的子流表
       - new_flow_acc / new_flow_path : 扩容后的状态表
       - failed_orig : 预处理阶段就确定失败的“原始流编号”
       - sub_to_orig : 新子流编号 -> 原始流编号 的映射
    """

    # 待恢复子流列表（元素结构与 flow_acc 前 7 列一致）
    affected_flows_info = []

    # 预处理阶段直接判定失败的“原始流编号”
    failed_orig = []

    # 新增恢复子流的编号从现有 flow_acc 末尾开始追加
    next_id = int(new_flow_acc.shape[0])

    # 防止同一个原始流在 affected_flow 中重复出现时被重复处理
    seen_orig = set()

    # 记录“恢复子流编号 -> 原始流编号”的映射
    sub_to_orig = {}

    for f in affected_flow:
        orig_id = int(f[0])

        # 同一原始流只处理一次
        if orig_id in seen_orig:
            continue
        seen_orig.add(orig_id)

        src = int(f[1])
        dst = int(f[2])
        bw = float(f[3])

        # 若故障节点本身是端点，则该原始流不可恢复
        if src == break_node or dst == break_node:
            failed_orig.append(orig_id)
            continue

        try:
            # 在“已经屏蔽故障节点”的拓扑上求恢复路径
            # k_shortest_path 的节点编号按 1-based 传入
            path = k_shortest_path(topo_dis, src + 1, dst + 1, 1)
            phys_nodes = path[0][0]
        except Exception:
            failed_orig.append(orig_id)
            continue

        # 不可达 / 空路径 / 非法路径都判恢复失败
        if not phys_nodes or len(phys_nodes) < 2:
            failed_orig.append(orig_id)
            continue

        # 根据恢复后的物理路径累计距离
        # 注意 phys_nodes 是 1-based，要减 1 后再访问 topo_dis
        dist = 0.0
        for a, b in zip(phys_nodes[:-1], phys_nodes[1:]):
            dist += float(topo_dis[int(a) - 1][int(b) - 1])

        # 重新计算该恢复流的单 SC 有效容量与所需 SC 数
        sc_cap_raw, _ = modu_format_Al(dist, bw)
        sc_cap = sc_effective_cap(float(sc_cap_raw))
        sc_num = sc_num_from_bw_cap(float(bw), float(sc_cap))

        # 为恢复子流分配新的 flow_id
        sub_id = next_id
        next_id += 1

        # 与 flow_acc 前 7 列保持一致：
        # [sub_id, src, dst, bw, sc_cap, sc_num, orig_id]
        item = [int(sub_id), src, dst, float(bw), float(sc_cap), int(sc_num), int(orig_id)]
        affected_flows_info.append(item)
        sub_to_orig[int(sub_id)] = int(orig_id)

    # 如果没有任何可恢复流，直接返回当前表
    if not affected_flows_info:
        return affected_flows_info, new_flow_acc, new_flow_path, failed_orig, sub_to_orig

    # ------------------------------------------------------------
    # 扩容 flow_acc
    # ------------------------------------------------------------
    add_n = len(affected_flows_info)
    flow_acc_ext = np.zeros((add_n, new_flow_acc.shape[1]), dtype=object)
    for i in range(add_n):
        # 7~14 列通常保存最终分配到的 hub/leaf/SC/FS 信息
        # 在真正分配前统一初始化为 -1
        flow_acc_ext[i][7:15] = [-1] * 8
    new_flow_acc = np.vstack([new_flow_acc, flow_acc_ext])

    # ------------------------------------------------------------
    # 扩容 flow_path
    # ------------------------------------------------------------
    flow_path_ext = np.empty((add_n, new_flow_path.shape[1]), dtype=object)
    for i in range(add_n):
        for c in range(new_flow_path.shape[1]):
            flow_path_ext[i][c] = []
    new_flow_path = np.vstack([new_flow_path, flow_path_ext])

    # ------------------------------------------------------------
    # 写入新增恢复子流的基础字段
    # ------------------------------------------------------------
    for item in affected_flows_info:
        sid = int(item[0])
        new_flow_acc[sid][0:7] = item
        new_flow_path[sid][0] = []
        new_flow_path[sid][1] = []
        new_flow_path[sid][2] = []
        new_flow_path[sid][3] = []

    return affected_flows_info, new_flow_acc, new_flow_path, failed_orig, sub_to_orig


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
    """
    DP 恢复总入口。

    核心流程：
    1. 先构造“故障后拓扑” topo_dis_rest，使恢复路径不能经过 break_node；
    2. 将 affected_flow 转成恢复子流表，并提前筛掉显然失败的原始流；
    3. 调用 FlexE_P2MP_DP(...) 在当前已有资源状态上做 DP 恢复；
    4. 把 DP 阶段失败的“子流编号”映射回“原始流编号”；
    5. 返回更新后的资源表，以及最终失败的原始流编号列表。
    """

    # ------------------------------------------------------------
    # 1. 构造故障后拓扑：
    #    与 break_node 相连的边全部置为 inf，表示不能再经过该节点
    # ------------------------------------------------------------
    topo_dis_rest = np.array(topo_dis, copy=True)
    topo_dis_rest[break_node, :] = np.inf
    topo_dis_rest[:, break_node] = np.inf

    # ------------------------------------------------------------
    # 2. 构造恢复输入，并预先收集明显失败的原始流
    # ------------------------------------------------------------
    affected_flows_info, flow_acc, flow_path, failed_pre, sub_to_orig = _build_restore_inputs(
        affected_flow,
        topo_dis_rest,
        flow_acc,
        flow_path,
        break_node,
    )

    # 若没有可尝试恢复的流，直接返回预处理失败结果
    if not affected_flows_info:
        failed_pre = list(dict.fromkeys(failed_pre))
        return node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, P2MP_FS, flow_path, failed_pre

    flows_num = len(affected_flows_info)

    # ------------------------------------------------------------
    # 3. 调用 DP 恢复器
    #    注意：这里传入的是“故障后拓扑” topo_dis_rest
    # ------------------------------------------------------------
    node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, P2MP_FS, flow_path, failed_sub = FlexE_P2MP_DP(
        affected_flows_info,
        flows_num,
        topo_num,
        topo_dis_rest,
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

    # ------------------------------------------------------------
    # 4. 将失败的“恢复子流编号”映射回“原始流编号”
    # ------------------------------------------------------------
    failed_orig = list(failed_pre)
    for sid in failed_sub:
        try:
            failed_orig.append(int(sub_to_orig.get(int(sid), int(sid))))
        except Exception:
            continue

    # 去重并保持顺序
    failed_orig = list(dict.fromkeys(failed_orig))

    # ------------------------------------------------------------
    # 5. 返回最终结果
    # ------------------------------------------------------------
    return node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, P2MP_FS, flow_path, failed_orig


# 基于故障后资源状态的 DP 恢复入口（源节点维度 + 连续FS块 + 带宽背包）
# 算法流程：
# 1. 先在函数内部再次构造故障后拓扑 topo_dis_rest，确保恢复路径不会经过 break_node。
# 2. 按源节点对待恢复子流分组；在每个源节点内部，再按 (目的节点, 物理路径) 聚成 group，
#    使同目的、同物理路径的业务优先复用同一棵 P2MP 树。
# 3. 对每个 group，依次枚举该源节点的所有 P2MP 端口 src_p：
#    - 若端口已有 base_fs，则尝试复用该端口；
#    - 若端口尚未分配 base_fs，则在 group 涉及的链路上寻找一个连续空闲 FS 块。
# 4. 在当前 src_p 上，以端口剩余带宽为容量做一次 knapsack_DP，先选出一批“候选流”；
#    这一步只是按带宽粗筛，真正能否放下还要经过后续 SC/FS 可行性检查。
# 5. 对背包选中的每条流，调用 _plan_sc_allocation(...) 做真实规划：
#    - 判断 hub 侧连续 SC 是否可用；
#    - 判断对应 FS 在旧链路/新链路上是否允许共享或新增；
#    - 选择或创建 leaf 端口，并得到 leaf 侧 SC 段。
#    只有规划成功的流才会真正提交到 node_flow / flow_acc / P2MP_SC / P2MP_FS / link_FS。
# 6. 当前 src_p 这一轮成功放下的流会从 group["flows"] 中删除；若 group 还有剩余流，
#    它们会继续尝试后续 src_p。注意：每个 src_p 只做一轮背包，不会在同一端口上反复重跑。
# 7. 某个源节点下所有 group 处理完后，DP 阶段仍未成功的流会收集到 rest_flows，
#    最后交给 allocate_flows_sequential(...) 做逐条兜底恢复。
# 8. 返回更新后的资源表，以及 DP + 顺序兜底之后仍失败的恢复子流编号列表。
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
    """
    在当前已有资源状态上，对一组“恢复子流”进行 DP 恢复。

    输入说明：
    - flows_info 中的每个元素结构为：
      [flow_id, src, dst, bw, sc_cap, sc_num, orig_id]
    - topo_dis 应当是恢复阶段使用的拓扑；若 break_node 不为 None，
      本函数内部也会再次做一层保护性屏蔽。

    返回：
    (
        node_flow,
        node_P2MP,
        flow_acc,
        link_FS,
        P2MP_SC,
        P2MP_FS,
        flow_path,
        failed_sub,   # 失败的“子流编号”列表
    )
    """

    # ------------------------------------------------------------
    # 0. 再做一层故障拓扑保护，避免调用方误把原拓扑传进来
    # ------------------------------------------------------------
    topo_dis_rest = np.array(topo_dis, copy=True)
    # if break_node is not None:
    #     topo_dis_rest[break_node, :] = np.inf
    #     topo_dis_rest[:, break_node] = np.inf

    # P2MP 类型表：
    # [type_id, 可用SC数量, 连续FS块大小]
    type_P2MP = [[k, int(_TYPE_INFO[k][0]), int(_TYPE_INFO[k][1])] for k in sorted(_TYPE_INFO.keys())]

    # 源节点可用的总 P2MP 端口数
    _p2mp_total = int(node_P2MP.shape[1])

    # 记录最终失败的“恢复子流编号”
    failed = []

    # ------------------------------------------------------------
    # 1. 按源节点对待恢复流分组
    # ------------------------------------------------------------
    scr_flows = np.empty(topo_num, dtype=object)
    for n in range(topo_num):
        scr_flows[n] = []
        for flow in flows_info:
            if int(flow[1]) == n:
                scr_flows[n].append(flow[0:7])

    # ------------------------------------------------------------
    # 2. 工具函数：在某条路径涉及的所有链路上找连续空闲 FS 块
    # ------------------------------------------------------------
    def _find_free_fs_block_on_links(link_FS, used_links, block_size):
        if not used_links:
            combined_usage = link_FS.sum(axis=0) * 0
        else:
            combined_usage = link_FS[used_links].sum(axis=0)
        for i0 in range(len(combined_usage) - int(block_size) + 1):
            if np.all(combined_usage[i0:i0 + int(block_size)] == 0):
                return int(i0)
        return None

    # ------------------------------------------------------------
    # 3. 按源节点逐个恢复
    # ------------------------------------------------------------
    for n in range(topo_num):
        # 故障节点本身不再作为恢复源节点
        if break_node is not None and n == break_node:
            continue

        # group_map:
        # key = (目的节点, 物理路径)
        # value 中保存这一组流共用的路径信息和流列表
        group_map = {}

        # 无法通过 P2MP 恢复的流，最后走逐条恢复兜底
        rest_flows = []

        # --------------------------------------------------------
        # 4.1 基于最短物理路径对源节点下的流做初始分组
        # --------------------------------------------------------
        for f in scr_flows[n]:
            fid = int(f[0])
            des = int(f[2])
            bw = float(f[3])

            # 若源/目的节点是故障节点，则这条恢复子流直接失败
            if break_node is not None and (n == break_node or des == break_node):
                failed.append(fid)
                continue

            try:
                # 在“故障后拓扑”上计算恢复路径
                path = k_shortest_path(topo_dis_rest, n + 1, des + 1, 1)
                phys_nodes = path[0][0]
                dist = float(path[0][1])
            except Exception:
                failed.append(fid)
                continue

            if not phys_nodes or len(phys_nodes) < 2:
                failed.append(fid)
                continue

            used_links = get_links_from_path(phys_nodes, link_index)
            used_links = [int(x) - 1 for x in used_links]

            # 这里按当前流自己的 bw 重新计算 sc_cap，并转为有效容量
            sc_cap_raw, _ = modu_format_Al(dist, bw)
            sc_eff = sc_effective_cap(float(sc_cap_raw))
            sc_needed = sc_num_from_bw_cap(float(bw), float(sc_eff))

            key = (des, tuple(phys_nodes))
            if key not in group_map:
                group_map[key] = {
                    "flows": [],
                    "phys_nodes": phys_nodes,
                    "used_links": used_links,
                }

            group_map[key]["flows"].append({
                "original": f,                 # 原子流记录 [id, src, dst, bw, sc_cap, sc_num, orig_id]
                "sc_needed": int(sc_needed),   # 当前代码里仅保留作统计/调试
                "bw": float(bw),               # 背包仍按带宽维度选择
                "sc_cap_eff": float(sc_eff),
            })

        # --------------------------------------------------------
        # 4.2 对每个分组尝试做 DP 打包恢复
        # --------------------------------------------------------
        for key, group in group_map.items():
            if not group["flows"]:
                continue

            des = int(key[0])
            phys_nodes = group["phys_nodes"]
            used_links = group["used_links"]

            # 枚举该源节点的所有 P2MP 端口
            for src_p in range(_p2mp_total):
                if not group["flows"]:
                    break

                # 读取端口类型与其对应的连续 FS 块大小
                t = int(node_P2MP[n][src_p][3])
                spec = type_P2MP[t - 1]
                fs_size = int(spec[2])

                # 若该端口已有基址，则复用；否则尝试为该组寻找新的连续 FS 块
                base_fs = int(node_P2MP[n][src_p][5])
                if base_fs >= 0:
                    fs_start = base_fs
                else:
                    fs_start = _find_free_fs_block_on_links(link_FS, used_links, fs_size)
                    if fs_start is None:
                        continue

                # 当前端口剩余带宽容量
                cap_bw = int(max(0, int(node_P2MP[n][src_p][4])))
                if cap_bw <= 0:
                    continue

                # 当前实现：背包按“带宽”选候选流
                items_for_group = []
                for flow_obj in group["flows"]:
                    fid = int(flow_obj["original"][0])
                    bw_int = int(round(float(flow_obj["bw"])))
                    items_for_group.append([fid, bw_int])

                if not items_for_group:
                    continue

                # knapsack_DP 返回被选中的流 id 集合
                _, _, _, group_items = knapsack_DP(int(cap_bw), items_for_group)
                if not group_items:
                    continue

                selected_ids = set(int(x) for x in group_items)
                selected_flow_objs = [fo for fo in group["flows"] if int(fo["original"][0]) in selected_ids]
                if not selected_flow_objs:
                    continue

                # ------------------------------------------------
                # 端口准备：
                # - 原本已有流：增量写入
                # - 原本空端口：先把 base_fs 写入，并清空 SC 容器
                # ------------------------------------------------
                original_base = int(node_P2MP[n][src_p][5])
                had_existing_flows = len(node_flow[n][src_p][0]) > 0

                if original_base < 0:
                    node_P2MP[n][src_p][5] = int(fs_start)

                if not had_existing_flows:
                    node_flow[n][src_p][0].clear()
                    _clear_p2mp_sc(P2MP_SC, n, src_p)

                placed_any = False
                placed_ids = set()

                # ------------------------------------------------
                # 4.2.1 对背包选出的候选流逐条做真实 SC/FS 可行性分配
                # ------------------------------------------------
                for f_obj in selected_flow_objs:
                    f = f_obj["original"]
                    fid = int(f[0])
                    bw = float(f[3])
                    sc_eff = float(f_obj["sc_cap_eff"])

                    plan = _plan_sc_allocation(
                        n, src_p, t, int(fs_start), des,
                        phys_nodes, bw, sc_eff, node_flow[n][src_p][0], flow_acc, P2MP_SC,
                        node_P2MP, node_flow, link_FS, P2MP_FS, used_links, flow_path
                    )
                    if plan is None:
                        continue

                    sc_s, sc_e, usage, fs_s_glo, fs_e_glo, leaf_p, leaf_sc_s, leaf_sc_e, leaf_usage = plan

                    # --------------------------------------------
                    # 提交 1：双端端口承载关系
                    # --------------------------------------------
                    node_flow[n][src_p][0].append(f)
                    node_flow[des][leaf_p][0].append(f)
                    node_P2MP[n][src_p][4] = float(node_P2MP[n][src_p][4]) - float(bw)
                    node_P2MP[des][leaf_p][4] = float(node_P2MP[des][leaf_p][4]) - float(bw)

                    # --------------------------------------------
                    # 提交 2：flow_acc 记录最终映射结果
                    # --------------------------------------------
                    flow_acc[fid][7] = int(src_p)
                    flow_acc[fid][8] = int(leaf_p)
                    flow_acc[fid][9] = int(sc_s)
                    flow_acc[fid][10] = int(sc_e)
                    flow_acc[fid][11] = int(fs_s_glo)
                    flow_acc[fid][12] = int(fs_e_glo)
                    flow_acc[fid][13] = int(leaf_sc_s)
                    flow_acc[fid][14] = int(leaf_sc_e)

                    # --------------------------------------------
                    # 提交 3：写 hub 侧 SC 与对应的 P2MP_FS 占用
                    # --------------------------------------------
                    for sc, used_amt in usage.items():
                        P2MP_SC[n][src_p][int(sc)][1].append([fid, float(used_amt), f[6]])
                        P2MP_SC[n][src_p][int(sc)][0].append(float(sc_eff))
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

                    # --------------------------------------------
                    # 提交 4：写路径记录
                    # --------------------------------------------
                    flow_path[fid][0].append(fid)
                    flow_path[fid][1].extend(used_links)
                    flow_path[fid][2].extend(phys_nodes)
                    flow_path[fid][3].append(f[6])

                    # --------------------------------------------
                    # 提交 5：写 leaf 侧 SC 与对应 P2MP_FS 占用
                    # --------------------------------------------
                    _apply_leaf_usage(
                        P2MP_SC, des, leaf_p,
                        int(leaf_sc_s), int(leaf_sc_e),
                        leaf_usage, float(sc_eff),
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

                    # --------------------------------------------
                    # 提交 6：更新路径上链路的 FS 占用计数
                    # --------------------------------------------
                    for l in set(used_links):
                        link_FS[int(l)][int(fs_s_glo):int(fs_e_glo) + 1] += 1

                    placed_any = True
                    placed_ids.add(fid)

                # 从分组中删除本轮成功恢复的流
                if placed_ids:
                    group["flows"] = [
                        fo for fo in group["flows"]
                        if int(fo["original"][0]) not in placed_ids
                    ]

                # 本轮至少成功一条，端口置为已启用
                if placed_any:
                    node_P2MP[n][src_p][2] = 1

                # 若新启用端口最终一条都没放下，则回滚
                if not placed_any and not had_existing_flows:
                    node_P2MP[n][src_p][2] = 0
                    if original_base < 0:
                        node_P2MP[n][src_p][5] = -1
                    node_flow[n][src_p][0].clear()
                    _clear_p2mp_sc(P2MP_SC, n, src_p)

        # --------------------------------------------------------
        # 4.3 收集当前源节点下，DP 仍未处理成功的流
        # --------------------------------------------------------
        for group in group_map.values():
            for f_obj in group["flows"]:
                rest_flows.append(f_obj["original"])

        # --------------------------------------------------------
        # 4.4 兜底：调用顺序恢复器逐条尝试
        #     注意这里也必须使用 topo_dis_rest
        # --------------------------------------------------------
        if rest_flows:
            node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, P2MP_FS, flow_path, failed_rest = allocate_flows_sequential(
                flows_info=np.array(rest_flows, dtype=object),
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
            failed.extend([int(x) for x in failed_rest])

    # 去重并保持顺序
    failed = list(dict.fromkeys(failed))

    return node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, P2MP_FS, flow_path, failed
