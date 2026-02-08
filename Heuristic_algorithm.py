#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : ILP.py 
# @File    : Heuristic_algorithm.py
# @Author  : Wumh
# @Time    : 2025/12/30 22:25
import numpy as np
import copy
import math
import SC_FS_list
import topology as tp
from k_shortest_path import k_shortest_path
from other_fx import extract_old_flow_info, build_virtual_topology, manage_s1_reservation_by_copy, try_assign_one_hop_s1, \
    build_fs_meta_np_from_p2mp_sc


# 假设你有一个 resource_allocation 模块或在同文件中定义了分配函数
# from resource_allocation import allocate_resources_strategy1, allocate_resources_strategy2

def Heuristic_algorithm(affected_flow, new_flow_acc, new_node_flow, new_node_P2MP, break_node, Tbox_num, Tbox_P2MP,
                        new_P2MP_SC_1, new_link_FS, node_flow_DP, node_P2MP_DP, flow_acc_DP, link_FS_DP, P2MP_SC_DP,
                        flow_path_DP):
    flow_num = len(affected_flow)
    M = 100000  # 大数
    topo_num, topo_matrix, topo_dis, link_num, link_index = tp.topology(1)

    # 参数设置
    K_PHY_CANDIDATES = 3  # 每一对虚拟节点间，预存多少条物理路径候选
    K_LOGICAL_PATHS = 5  # 虚拟层面上，尝试多少条逻辑路径 (稍微增加以提高S1成功率)

    RECONFIG_PENALTY = 5000  # 重构惩罚
    HOP_PENALTY = 200  # 每一跳的惩罚

    # =======================================================
    # 1. 预计算物理路径候选池 (Physical Path Pool)
    # =======================================================
    print("正在构建物理路径候选池...")
    phy_pool = {}
    for u in range(topo_num):
        for v in range(topo_num):
            if u == v: continue
            paths_output = k_shortest_path(topo_dis, u + 1, v + 1, K_PHY_CANDIDATES)
            valid_paths = []
            for entry in paths_output:
                path_1based = entry[0]  # 原始 1-based 路径
                cost = entry[1]
                # 【修正】：在这里直接转换为 0-based
                path_0based = [node - 1 for node in path_1based]
                # 存入 valid_paths，结构保持为 [path, cost]
                valid_paths.append([path_0based, cost])
            if len(valid_paths) > 0:
                phy_pool[(u, v)] = valid_paths

    # =======================================================
    # ！2. 预处理：提取流信息并进行 Tier 分类
    # =======================================================
    print("正在执行流的紧凑预处理与分类...")
    tier1_flows = []
    tier2_flows = []
    flow_vtopo_map = {}  # 新增：用于存储每条流的虚拟拓扑数组
    flow_metadata_map = {}

    for flow in affected_flow:
        f_id = flow[0]
        src, dst, band = flow[1], flow[2], flow[3]

        # 2.1 提取历史信息 [cite: 60-71]
        old_info = extract_old_flow_info(f_id, src, dst, flow_acc_DP, P2MP_SC_DP)
        flow_metadata_map[f_id] = old_info

        # 2.2 紧判别：构建 S1 专属虚拟拓扑 [cite: 120-125, 263-265]
        # 在这里强制尝试 Strategy 1 约束
        v_adj, v_phy_map = build_virtual_topology(
            src, dst, topo_num, phy_pool, break_node, new_node_P2MP, band,
            force_strategy1=True,  # 紧约束模式
            orig_src_modu=old_info['orig_src_modu'],
            orig_dst_modu=old_info['orig_dst_modu'],
            RECONFIG_PENALTY=RECONFIG_PENALTY,
            HOP_PENALTY=HOP_PENALTY
        )

        # 检查在该拓扑下，源宿是否连通 (只要不是全是 inf)
        # 使用 Dijkstra 或简单检查 v_adj[src][dst] 是否小于 inf
        if np.any(v_adj[src, :] < np.inf) and np.any(v_adj[:, dst] < np.inf):
            # 具备 S1 资格，缓存其拓扑并入队 Tier 1
            flow_vtopo_map[f_id] = (v_adj, v_phy_map)
            tier1_flows.append(flow)
        else:
            # 物理距离或硬件格式完全不匹配，直接划入 Tier 2
            tier2_flows.append(flow)

    print(f"紧分类完成: Tier 1(潜在 S1): {len(tier1_flows)}, Tier 2(强制 S2): {len(tier2_flows)}")


    # =======================================================
    # 3. 排序 (Sort)
    # =======================================================
    # 两个队列内部都按带宽降序排列
    tier1_flows.sort(key=lambda x: x[3], reverse=True)
    tier2_flows.sort(key=lambda x: x[3], reverse=True)

    print(f"分类完成: Tier 1 (优先尝试 S1): {len(tier1_flows)} 条, Tier 2 (S2 重构): {len(tier2_flows)} 条")
    # =======================================================
    # 4. 资源预占位
    # =======================================================
    manage_s1_reservation_by_copy(
        tier1_flows,
        flow_metadata_map,
        P2MP_SC_DP,  # 初始网络的老表
        new_P2MP_SC_1,  # 当前网络的新表
        new_node_P2MP,  # 节点 P2MP 容量表
        action='reserve'  # 执行占位
    )

    # =========================
    # 5) Strategy 1：逐流恢复（允许 OEO）——全表 deepcopy 尝试版
    # =========================

    tier1_failed = []
    tier1_restored = []

    for flow in tier1_flows:
        f_id, src, dst, band = int(flow[0]), int(flow[1]), int(flow[2]), float(flow[3])
        v_adj, v_phy_map = flow_vtopo_map[f_id]

        logical_paths = k_shortest_path(v_adj, src + 1, dst + 1, K_LOGICAL_PATHS)

        restored = False
        for path_nodes_1based, _cost in logical_paths:

            # 兼容：你的 logical_path_is_valid 可能是 1-based 或 0-based
            try:
                if not logical_path_is_valid(path_nodes_1based, v_adj, v_phy_map):
                    continue
            except Exception:
                nodes0_tmp = [x - 1 for x in path_nodes_1based]
                if not logical_path_is_valid(nodes0_tmp, v_adj, v_phy_map):
                    continue

            nodes0 = [x - 1 for x in path_nodes_1based]  # v_phy_map key 通常用 0-based (a,b)

            # --------- 核心变化：每条逻辑路径尝试前，全表复制一份作为 working state ---------
            work_node_flow = copy.deepcopy(new_node_flow)
            work_node_P2MP = copy.deepcopy(new_node_P2MP)
            work_flow_acc = copy.deepcopy(new_flow_acc)
            work_link_FS = copy.deepcopy(new_link_FS)
            work_P2MP_SC_1 = copy.deepcopy(new_P2MP_SC_1)
            work_flow_path = copy.deepcopy(flow_path_DP)
            work_P2MP_FS_1 = copy.deepcopy(new_P2MP_FS_1)
            work_link_FS_meta = copy.deepcopy(new_link_FS_meta)

            hop_results = []
            ok_path = True

            for a, b in zip(nodes0[:-1], nodes0[1:]):

                cand_obj = v_phy_map[(a, b)]
                if isinstance(cand_obj, dict):
                    cand_list = [cand_obj]
                else:
                    cand_list = cand_obj  # list[dict]

                hop_ok = False
                for cand in cand_list:
                    phy_path_1based = cand["path"]

                    # try_assign_one_hop_s1：在 working state 上边尝试边写
                    ok_hop, hop_res = try_assign_one_hop_s1(
                        flow=flow,
                        a=a, b=b,
                        phy_candidate=cand,  # 推荐：直接传candidate（含path/dist/modu/cost）
                        phy_path_1based=phy_path_1based,  # 若你不想改签名，也可只用path

                        flow_metadata_map=flow_metadata_map,
                        link_index=link_index,

                        new_link_FS=work_link_FS,
                        new_node_P2MP=work_node_P2MP,
                        new_P2MP_SC_1=work_P2MP_SC_1,
                        new_node_flow=work_node_flow,
                        new_flow_acc=work_flow_acc,
                        new_P2MP_FS_1 = work_P2MP_FS_1,
                        new_link_FS_meta = work_link_FS_meta,

                        strict_s1 = True,)

                    if ok_hop:
                        if hop_res is None:
                            hop_res = {}
                        hop_res.setdefault("phy_path_1based", phy_path_1based)
                        hop_res.setdefault("phy_dist", cand.get("dist"))
                        hop_res.setdefault("phy_modu", cand.get("modu"))
                        hop_res.setdefault("phy_cost", cand.get("cost"))
                        hop_results.append(hop_res)

                        hop_ok = True
                        break

                if not hop_ok:
                    ok_path = False
                    break

            if ok_path:
                # 成功：commit 写回（在 working 的 flow_acc/flow_path 上写）
                commit_s1_flow(
                    flow=flow,
                    hop_results=hop_results,
                    flow_metadata_map=flow_metadata_map,
                    new_flow_acc=work_flow_acc,
                    new_flow_path=work_flow_path,
                )

                # 再把 working state 覆盖回 new_*（注意用[:]保持外部引用不变）
                new_node_flow[:] = work_node_flow
                new_node_P2MP[:] = work_node_P2MP
                new_flow_acc[:] = work_flow_acc
                new_link_FS[:] = work_link_FS
                new_P2MP_SC_1[:] = work_P2MP_SC_1
                new_flow_path[:] = work_flow_path
                new_P2MP_FS_1[:] = work_P2MP_FS_1
                new_link_FS_meta[:] = work_link_FS_meta

                tier1_restored.append(flow)
                restored = True
                break

            # ok_path == False：直接丢弃 working 副本，继续下一条逻辑路径

        if not restored:
            # S1彻底失败：撤销端点预占位资源，再进入 S2
            manage_s1_reservation_by_copy(
                tier1_flows,
                flow_metadata_map,
                P2MP_SC_DP,  # 初始网络的老表
                new_P2MP_SC_1,  # 当前网络的新表
                new_node_P2MP,  # 节点 P2MP 容量表
                action='rollback'  # 执行占位
            )
            tier1_failed.append(flow)

    tier2_flows.extend(tier1_failed)

    # =========================
    # 6) Strategy 2：逐流恢复（更自由/更激进）——同样全表 deepcopy 尝试版
    # =========================

    tier2_failed = []
    tier2_restored = []

    for flow in tier2_flows:
        f_id, src, dst, band = int(flow[0]), int(flow[1]), int(flow[2]), float(flow[3])
        v_adj, v_phy_map = flow_vtopo_map[f_id]

        logical_paths = k_shortest_path(v_adj, src + 1, dst + 1, K_LOGICAL_PATHS)

        restored = False
        for path_nodes_1based, _cost in logical_paths:

            try:
                if not logical_path_is_valid(path_nodes_1based, v_adj, v_phy_map):
                    continue
            except Exception:
                nodes0_tmp = [x - 1 for x in path_nodes_1based]
                if not logical_path_is_valid(nodes0_tmp, v_adj, v_phy_map):
                    continue

            nodes0 = [x - 1 for x in path_nodes_1based]

            # S2：每条逻辑路径尝试前，也全表 copy 一份
            work_node_flow = copy.deepcopy(new_node_flow)
            work_node_P2MP = copy.deepcopy(new_node_P2MP)
            work_flow_acc = copy.deepcopy(new_flow_acc)
            work_link_FS = copy.deepcopy(new_link_FS)
            work_P2MP_SC_1 = copy.deepcopy(new_P2MP_SC_1)
            work_flow_path = copy.deepcopy(new_flow_path)

            hop_results = []
            ok_path = True

            for a, b in zip(nodes0[:-1], nodes0[1:]):

                cand_obj = v_phy_map[(a, b)]
                if isinstance(cand_obj, dict):
                    cand_list = [cand_obj]
                else:
                    cand_list = cand_obj

                hop_ok = False
                for cand in cand_list:
                    phy_path_1based = cand["path"]

                    ok_hop, hop_res = try_assign_one_hop_s1(
                        flow=flow,
                        a=a, b=b,
                        phy_candidate=cand,
                        phy_path_1based=phy_path_1based,

                        flow_metadata_map=flow_metadata_map,
                        link_index=link_index,

                        new_link_FS=work_link_FS,
                        new_node_P2MP=work_node_P2MP,
                        new_P2MP_SC_1=work_P2MP_SC_1,
                        new_node_flow=work_node_flow,
                        new_flow_acc=work_flow_acc,

                        strict_s1 = False)

                    if ok_hop:
                        if hop_res is None:
                            hop_res = {}
                        hop_res.setdefault("phy_path_1based", phy_path_1based)
                        hop_res.setdefault("phy_dist", cand.get("dist"))
                        hop_res.setdefault("phy_modu", cand.get("modu"))
                        hop_res.setdefault("phy_cost", cand.get("cost"))
                        hop_results.append(hop_res)

                        hop_ok = True
                        break

                if not hop_ok:
                    ok_path = False
                    break

            if ok_path:
                commit_s1_flow(
                    flow=flow,
                    hop_results=hop_results,
                    flow_metadata_map=flow_metadata_map,
                    new_flow_acc=work_flow_acc,
                    new_flow_path=work_flow_path,
                )

                # 覆盖回主状态
                new_node_flow[:] = work_node_flow
                new_node_P2MP[:] = work_node_P2MP
                new_flow_acc[:] = work_flow_acc
                new_link_FS[:] = work_link_FS
                new_P2MP_SC_1[:] = work_P2MP_SC_1
                new_flow_path[:] = work_flow_path

                tier2_restored.append(flow)
                restored = True
                break

        if not restored:
            tier2_failed.append(flow)

    return new_node_flow, new_node_P2MP, new_flow_acc, new_link_FS, new_P2MP_SC_1, new_flow_path