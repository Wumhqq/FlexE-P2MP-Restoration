#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Patched version:
#  - Keep SC_cap as float (e.g., 12.5)
#  - Compute SC_num using integer TS allocation rule (TS_UNIT=5)
#  - Ensure leaf uses the SAME absolute FS range as hub for every flow
#  - If a matching leaf cannot be allocated, rollback the hub allocation (and any temporary leaf allocations)

import copy
import math
import numpy as np

from knapsack_DP import knapsack_DP
from k_shortest_path import k_shortest_path
from modu_format_Al import modu_format_Al
from SC_FS import sc_fs
from FS_SC import map_fs_to_sc
from Path_Link import get_links_from_path

TS_UNIT = 5  # TS 粒度（Gbps），与 ILP 中 *5 的定义保持一致


def sc_effective_cap(sc_cap: float) -> int:
    """把 SC 容量映射成可用的整数 TS 容量（不允许 2.5 TS 这种分配）。
    例如：12.5Gbps => floor(12.5/5)=2 个 TS => 10Gbps。
    """
    return int(math.floor(float(sc_cap) / TS_UNIT) * TS_UNIT)


def sc_num_from_bw_cap(bw: float, sc_cap: float) -> int:
    eff = sc_effective_cap(sc_cap)
    if eff <= 0:
        raise ValueError(f"Invalid SC_cap={sc_cap}")
    return int(math.ceil(float(bw) / eff))


def _clear_p2mp_sc(P2MP_SC: np.ndarray, node: int, p: int) -> None:
    """清空一个节点 node 的某个 P2MP(p) 上的 16*5 SC 记录。"""
    for s in range(16):
        for c in range(5):
            P2MP_SC[node][p][s][c] = []


def FlexE_P2MP_DP(flows_info, flows_num, topo_num, topo_dis, link_num, link_index, Tbox_num, Tbox_P2MP):
    # P2MP 资源类型（沿用你原代码的设定）
    # [type_id, max_SC_num_for_knapsack, FS_block_size]
    type_P2MP = [[1, 1, 1], [2, 4, 2], [3, 16, 6]]

    # 将 P2MP 资源按索引粗分为 3 类（3/2/1 型号），用整数阈值避免浮点/边界重叠
    _p2mp_total = Tbox_num * Tbox_P2MP
    _t1 = _p2mp_total // 3
    _t2 = 2 * _p2mp_total // 3

    # 记录链路频谱占用
    link_FS = np.zeros((link_num, 358), dtype=int)

    # ---- 预处理 flows：按 src/dst 排序，重编号 flow_id ----
    temp = np.array(flows_info, dtype=object)
    sorted_indices = np.lexsort((temp[:, 2], temp[:, 1]))
    flows_info = temp[sorted_indices]
    for i, f in enumerate(flows_info):
        f[0] = i

    # 将流按源节点分类
    scr_flows = np.empty(topo_num, dtype=object)
    for n in range(topo_num):
        scr_flows[n] = []
        for flow in flows_info:
            if int(flow[1]) == n:
                scr_flows[n].append(flow[0:7])

    # ---- 状态表 ----
    # node_P2MP: [Tbox_id, P2MP_id, used_flag(0/1/2), type(1/2/3), remaining_cap, FS_start]
    node_P2MP = np.zeros((topo_num, _p2mp_total, 6), dtype=int)
    # node_flow[u][p][0] = list of flows on this P2MP
    node_flow = np.empty((topo_num, _p2mp_total, 1), dtype=object)
    # flow_acc: object 防止 float 截断
    # [0..6 原始信息, 7 hub_p2mp, 8 leaf_p2mp, 9 sc_s, 10 sc_e, 11 fs_s_abs, 12 fs_e_abs]
    flow_acc = np.zeros((flows_num, 13), dtype=object)
    # P2MP_SC[u,p,sc,:] 每个 entry 存 list
    P2MP_SC = np.empty((topo_num, _p2mp_total, 16, 5), dtype=object)
    # flow_path[f]：记录链路集合等
    flow_path = np.empty((flows_num, 4), dtype=object)

    for u in range(topo_num):
        for p in range(_p2mp_total):
            node_P2MP[u][p][0] = math.floor(p / Tbox_P2MP)
            node_P2MP[u][p][1] = p
            node_P2MP[u][p][4] = 400
            node_P2MP[u][p][5] = -1

            # init P2MP_SC lists
            for s in range(16):
                for c in range(5):
                    P2MP_SC[u][p][s][c] = []

            if p < _t1:
                node_P2MP[u][p][3] = 3
            elif p < _t2:
                node_P2MP[u][p][3] = 2
            else:
                node_P2MP[u][p][3] = 1

            node_flow[u][p][0] = []

    for i in range(flows_num):
        flow_acc[i][0:7] = flows_info[i][0:7]
        flow_path[i][0] = []
        flow_path[i][1] = []
        flow_path[i][2] = []
        flow_path[i][3] = []

    # ===================== hub 分配 + leaf 对齐分配（失败则回滚 hub） =====================
    for n in range(topo_num):
        h = 0
        B_flows_info = copy.deepcopy(scr_flows[n])

        while len(B_flows_info) > 0 and h < _p2mp_total:
            # --- 先按 400 带宽装一批 flow 到当前 hub ---
            _, _, _, final_items = knapsack_DP(400, B_flows_info)
            final_items_info = []
            for fid in final_items:
                final_items_info.append(flows_info[int(fid)][0:7])
                for idx, item in enumerate(B_flows_info):
                    if int(item[0]) == int(fid):
                        B_flows_info.pop(idx)
                        break

            # --- 以目的节点聚合，计算每个目的节点需要的 SC_num（按 TS 口径） ---
            des_band = np.empty((topo_num, 2), dtype=object)
            des_SC = np.empty((topo_num, 4), dtype=object)  # [des, sc_num, total_bw, sc_cap]

            for i in range(topo_num):
                des_band[i][0] = []
                des_band[i][1] = []
                des_SC[i] = [i, 0, 0, 0.0]

            for f in final_items_info:
                des = int(f[2])
                des_band[des][0].append(f)
                des_SC[des][2] = float(des_SC[des][2]) + float(f[3])

            for des in range(topo_num):
                des_SC[des][0] = des
                path = k_shortest_path(topo_dis, n + 1, des + 1, 1)
                des_band[des][1].append(path[0][0])
                SC_cap, _ = modu_format_Al(path[0][1], des_SC[des][2])
                des_SC[des][1] = sc_num_from_bw_cap(des_SC[des][2], SC_cap)
                des_SC[des][3] = float(SC_cap)

            copy_Sdes_SC = np.array(copy.deepcopy(des_SC), dtype=object)
            copy_Sdes_band = copy.deepcopy(des_band)

            # --- 初始化当前 hub 状态 ---
            node_P2MP[n][h][2] = 1
            node_P2MP[n][h][5] = -1
            node_flow[n][h][0].clear()
            _clear_p2mp_sc(P2MP_SC, n, h)

            P2MP_tree = []

            # 先在 hub 上按 "SC 数" 装目的节点（knapsack）
            _, _, _, final_items_P2MP = knapsack_DP(type_P2MP[int(node_P2MP[n][h][3]) - 1][1], copy_Sdes_SC)

            # 未被选中的目的节点：把流回炉（否则流会丢）
            selected_des = set(int(x) for x in final_items_P2MP)
            for des_id in range(topo_num):
                if des_id not in selected_des and len(copy_Sdes_band[des_id][0]) > 0:
                    B_flows_info.extend(copy_Sdes_band[des_id][0])

            # --- 在 hub 内进行 SC 分配（按目的节点顺序占 SC）---
            FS_end = -1
            for des in final_items_P2MP:
                des = int(des)
                FS_start_rel = FS_end + 1
                SC_start = map_fs_to_sc(FS_start_rel, int(node_P2MP[n][h][3]))
                if SC_start is None:
                    # 该目的节点无法映射到 SC：流回炉
                    for f in copy_Sdes_band[des][0]:
                        B_flows_info.append(f)
                    continue

                SC_cap = float(copy_Sdes_SC[des][3])
                SC_spare = SC_cap
                SC_spare_0 = SC_spare

                for f in copy_Sdes_band[des][0]:
                    bw = float(f[3])
                    if bw <= SC_spare:
                        SC_end = SC_start
                        SC_spare -= bw
                    else:
                        SC_end = int(math.ceil((bw - SC_spare) / SC_cap) + SC_start)
                        SC_spare = SC_cap * (SC_end - SC_start) - (bw - SC_spare)

                    if SC_end <= 15:
                        FS_start_rel = sc_fs(int(node_P2MP[n][h][3]), SC_start, 1)
                        FS_end = sc_fs(int(node_P2MP[n][h][3]), SC_end, 2)

                        node_flow[n][h][0].append(f)
                        fid = int(f[0])

                        flow_acc[fid][7] = h
                        flow_acc[fid][9] = SC_start
                        flow_acc[fid][10] = SC_end
                        flow_acc[fid][11] = FS_start_rel
                        flow_acc[fid][12] = FS_end

                        # 记录使用 SC 的流和大小
                        if SC_start == SC_end:
                            P2MP_SC[n][h][SC_start][1].append([fid, bw, f[6]])
                            P2MP_SC[n][h][SC_start][0].append(SC_cap)
                            P2MP_SC[n][h][SC_start][2].append(copy_Sdes_band[des][1][0])
                            P2MP_SC[n][h][SC_start][4].append(int(f[2]))
                        else:
                            P2MP_SC[n][h][SC_start][1].append([fid, SC_spare_0, f[6]])
                            P2MP_SC[n][h][SC_start][0].append(SC_cap)
                            P2MP_SC[n][h][SC_start][2].append(copy_Sdes_band[des][1][0])
                            P2MP_SC[n][h][SC_start][4].append(int(f[2]))
                            for s in range(SC_start + 1, SC_end):
                                P2MP_SC[n][h][s][1].append([fid, SC_cap, f[6]])
                                P2MP_SC[n][h][s][0].append(SC_cap)
                                P2MP_SC[n][h][s][2].append(copy_Sdes_band[des][1][0])
                                P2MP_SC[n][h][s][4].append(int(f[2]))
                            P2MP_SC[n][h][SC_end][1].append([fid, SC_cap - SC_spare, f[6]])
                            P2MP_SC[n][h][SC_end][0].append(SC_cap)
                            P2MP_SC[n][h][SC_end][2].append(copy_Sdes_band[des][1][0])
                            P2MP_SC[n][h][SC_end][4].append(int(f[2]))

                        # 记录路径/链路
                        used_links = get_links_from_path(copy_Sdes_band[des][1][0], link_index)
                        used_links = [t - 1 for t in used_links]  # 0-based
                        flow_path[fid][0].append(fid)
                        flow_path[fid][1].extend(used_links)
                        flow_path[fid][2].extend(copy_Sdes_band[des][1][0])
                        flow_path[fid][3].append(f[6])
                        P2MP_tree.extend(used_links)

                        SC_spare_0 = SC_spare
                        SC_start = SC_end
                        if SC_spare == 0:
                            SC_start = SC_end + 1
                            SC_spare = SC_cap
                            SC_spare_0 = SC_spare
                    else:
                        # hub 上 SC 不够：该流回炉
                        B_flows_info.append(f)

            # --- 为 hub 选择全局 FS_start（中心频率）---
            P2MP_tree = list(set(P2MP_tree))
            combined_usage = link_FS[P2MP_tree].sum(axis=0) if len(P2MP_tree) > 0 else link_FS.sum(axis=0) * 0
            block_size = type_P2MP[int(node_P2MP[n][h][3]) - 1][2]

            free_block_indices = None
            for i0 in range(len(combined_usage) - block_size + 1):
                if np.all(combined_usage[i0:i0 + block_size] == 0):
                    free_block_indices = i0
                    break

            if free_block_indices is None:
                # 没有连续块：回滚 hub
                for _f in node_flow[n][h][0]:
                    B_flows_info.append(_f)
                    fid = int(_f[0])
                    flow_acc[fid][7:13] = [0, 0, 0, 0, 0, 0]
                    flow_path[fid][0].clear()
                    flow_path[fid][1].clear()
                    flow_path[fid][2].clear()
                    flow_path[fid][3].clear()
                node_flow[n][h][0].clear()
                node_P2MP[n][h][2] = 0
                node_P2MP[n][h][5] = -1
                _clear_p2mp_sc(P2MP_SC, n, h)
                h += 1
                continue

            hub_fs0 = int(free_block_indices)
            node_P2MP[n][h][5] = hub_fs0

            # 把 flow_acc 的 FS 变成绝对 FS（但先不更新 link_FS，等 leaf 成功后再 commit）
            for f in node_flow[n][h][0]:
                fid = int(f[0])
                flow_acc[fid][11] = hub_fs0 + int(flow_acc[fid][11])
                flow_acc[fid][12] = hub_fs0 + int(flow_acc[fid][12])

            # ================= leaf 分配（强约束：与 hub 同 FS）=================
            flows_on_hub = node_flow[n][h][0]
            hub_type = int(node_P2MP[n][h][3])

            # 每个目的节点占用的 SC 区间
            des_to_sc_range = {}
            for f in flows_on_hub:
                fid = int(f[0])
                des = int(f[2])
                sc_s = int(flow_acc[fid][9])
                sc_e = int(flow_acc[fid][10])
                if des not in des_to_sc_range:
                    des_to_sc_range[des] = [sc_s, sc_e]
                else:
                    des_to_sc_range[des][0] = min(des_to_sc_range[des][0], sc_s)
                    des_to_sc_range[des][1] = max(des_to_sc_range[des][1], sc_e)

            allocated_leaf = []  # (des, p)
            leaf_ok = True

            for des, (min_sc, max_sc) in des_to_sc_range.items():
                # 约束：leaf 与 hub 同 type，且 leaf 的 FS_start(中心频率) 与 hub 相同
                chosen_p = None
                for p in range(_p2mp_total):
                    if node_P2MP[des][p][2] != 0:
                        continue
                    if int(node_P2MP[des][p][3]) != hub_type:
                        continue
                    chosen_p = p
                    break

                if chosen_p is None:
                    leaf_ok = False
                    break

                # 分配该 leaf
                node_P2MP[des][chosen_p][2] = 2
                node_P2MP[des][chosen_p][5] = hub_fs0
                node_flow[des][chosen_p][0].clear()
                _clear_p2mp_sc(P2MP_SC, des, chosen_p)

                # 复制 hub 上相同 SC 索引区间（不做平移），保证 FS 一致
                for sc in range(min_sc, max_sc + 1):
                    for c in range(5):
                        P2MP_SC[des][chosen_p][sc][c] = copy.deepcopy(P2MP_SC[n][h][sc][c])

                # 挂接流、并检查每条流的 FS 是否完全一致
                for f in flows_on_hub:
                    if int(f[2]) != int(des):
                        continue
                    fid = int(f[0])
                    node_flow[des][chosen_p][0].append(f)
                    flow_acc[fid][8] = chosen_p

                    # 校验 leaf 侧推导的 FS 与 hub 侧绝对 FS 一致
                    leaf_fs_s = hub_fs0 + sc_fs(hub_type, int(flow_acc[fid][9]), 1)
                    leaf_fs_e = hub_fs0 + sc_fs(hub_type, int(flow_acc[fid][10]), 2)
                    if int(leaf_fs_s) != int(flow_acc[fid][11]) or int(leaf_fs_e) != int(flow_acc[fid][12]):
                        leaf_ok = False
                        break

                allocated_leaf.append((int(des), int(chosen_p)))
                if not leaf_ok:
                    break

            if not leaf_ok:
                # 回滚 leaf
                for des, p in allocated_leaf:
                    node_flow[des][p][0].clear()
                    node_P2MP[des][p][2] = 0
                    node_P2MP[des][p][5] = -1
                    _clear_p2mp_sc(P2MP_SC, des, p)

                # 回滚 hub（把流放回去，清状态）
                for _f in node_flow[n][h][0]:
                    B_flows_info.append(_f)
                    fid = int(_f[0])
                    flow_acc[fid][7:13] = [0, 0, 0, 0, 0, 0]
                    flow_path[fid][0].clear()
                    flow_path[fid][1].clear()
                    flow_path[fid][2].clear()
                    flow_path[fid][3].clear()

                node_flow[n][h][0].clear()
                node_P2MP[n][h][2] = 0
                node_P2MP[n][h][5] = -1
                _clear_p2mp_sc(P2MP_SC, n, h)

                h += 1
                continue

            # ================= commit：更新 link_FS（此时 leaf 已成功）=================
            for f in flows_on_hub:
                fid = int(f[0])
                fs_s = int(flow_acc[fid][11])
                fs_e = int(flow_acc[fid][12])
                used_links = set(flow_path[fid][1])
                for l in used_links:
                    link_FS[int(l)][fs_s:fs_e + 1] += 1

            h += 1

    return node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, flow_path
