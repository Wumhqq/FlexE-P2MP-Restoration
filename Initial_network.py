#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : code 
# @File    : FlexE_P2MP_DP.py
# @Author  : Wumh
# @Time    : 2024/7/16 17:07
import numpy as np
from knapsack_DP import knapsack_DP
import copy
import math
from k_shortest_path import k_shortest_path
from modu_format_Al import modu_format_Al
from topology import topology
from SC_FS import sc_fs
from FS_SC import map_fs_to_sc
from Path_Link import get_links_from_path
def FlexE_P2MP_DP(flows_info, flows_num, topo_num, topo_dis, link_num, link_index, Tbox_num, Tbox_P2MP):
    # 设置参数
    type_P2MP = [[1, 1, 1], [2, 4, 2], [3, 16, 6]]
    # 创建记录FS的信息表
    link_FS = np.zeros((link_num, 358), dtype=int)
    # SC和FS的对应表

    temp = np.array(flows_info)
    sorted_indices = np.lexsort((temp[:, 2], temp[:, 1]))
    flows_info = temp[sorted_indices]
    i = 0
    for f in flows_info:
        f[0] = i
        i = i + 1
    # 将流量按照源节点分类
    scr_flows = np.empty(topo_num, dtype=object)
    for n in range(topo_num):
        scr_flows[n] = []
        for flow in flows_info:
            if flow[1] == n:
                scr_flows[n].append(flow[0:7])

    # 创建记录的信息表
    # 1.node_P2MP: 0.T-box序号 1.P2MP序号 2.P2MP是否使用 3.型号 4.接收端剩余容量 5.起始频率
    # 2.node_flow: 0.P2MP容纳的流序号: 0.flow序号 1.源节点 2.目的节点 3.带宽 4.调制格式 5.SC
    # 3.flow_acc: 0.flow序号 1.源节点 2.目的节点 3.带宽 4.调制格式 5.SC 6.源流序号 7.hub_P2MP 8.leaf_P2MP 9.SC起始 10.SC末尾 11.FS起始 12.FS末尾
    node_P2MP = np.zeros((topo_num, Tbox_num * Tbox_P2MP, 6), dtype=int)
    node_flow = np.empty((topo_num, Tbox_num * Tbox_P2MP, 1), dtype=object)
    flow_acc = np.zeros((flows_num, 13), dtype=int)
    P2MP_SC = np.empty((topo_num, Tbox_num * Tbox_P2MP, 16, 5), dtype=object)
    flow_path = np.empty((flows_num, 4), dtype=object)
    for i in range(topo_num):
        for j in range(Tbox_num * Tbox_P2MP):
            node_P2MP[i][j][0] = math.floor(j / Tbox_P2MP)
            node_P2MP[i][j][1] = j
            node_P2MP[i][j][4] = 400
            node_P2MP[i][j][5] = -1
            for s in range(16):
                P2MP_SC[i][j][s][0] = []
                P2MP_SC[i][j][s][1] = []
                P2MP_SC[i][j][s][2] = []
                P2MP_SC[i][j][s][3] = []
                P2MP_SC[i][j][s][4] = []
            if j <= Tbox_num / 3:
                node_P2MP[i][j][3] = 3
            elif Tbox_num / 3 <= j <= Tbox_num / 3 * 2:
                node_P2MP[i][j][3] = 2
            else:
                node_P2MP[i][j][3] = 1
            node_flow[i][j][0] = []
    for i in range(flows_num):
        flow_acc[i][0:7] = flows_info[i][0:7]
        flow_path[i][0] = []
        flow_path[i][1] = []
        flow_path[i][2] = []
        flow_path[i][3] = []


    # 分配hub资源
    for n in range(topo_num):
        h = 0
        B_flows_info = copy.deepcopy(scr_flows[n])
        while(len(B_flows_info) > 0):
            value, bag, final_value, final_items = knapsack_DP(400, B_flows_info)
            final_items_info = []
            for i in final_items:
                final_items_info.append(flows_info[i][0:7])
                for index, j in enumerate(B_flows_info):
                    if j[0] == i:
                        B_flows_info.pop(index)
                        break

            # 以不同的目的地为基准计算使用的SC
            # des_band: 0.相同目的地流的整体信息flows_info 1.源目的节点所走的路径
            # des_SC: 0.目的地序号 1.SC个数 2.带宽总和 3.SC容量
            des_band = np.empty((topo_num, 2), dtype=object)
            des_SC = np.zeros((topo_num, 4), dtype=int)

            for i in range(topo_num):
                des_band[i][0] = []
                des_band[i][1] = []

            for i in final_items_info:
                des_band[i[2]][0].append(i)
                des_SC[i[2]][2] = des_SC[i[2]][2] + i[3]

            for i in range(topo_num):
                des_SC[i][0] = i
                path = k_shortest_path(topo_dis, n + 1, i + 1, 1)
                des_band[i][1].append(path[0][0])
                SC_cap, SC_num = modu_format_Al(path[0][1], des_SC[i][2])
                des_SC[i][1] = SC_num
                if SC_cap == 12.5:
                    des_SC[i][3] = 10
                else:
                    des_SC[i][3] = SC_cap

            # 需要注意，不应该在最开始确定使用的P2MP类型，应该先装一个然后在装一个
            # 1.node_P2MP: 0.T-box序号 1.P2MP序号 2.P2MP是否使用 3.型号 4.接收端剩余容量 5.起始频率
            # 2.node_flow: 0.P2MP容纳的流序号: 0.flow序号 1.源节点 2.目的节点 3.带宽 4.调制格式 5.SC
            # 3.flow_acc: 0.flow序号 1.源节点 2.目的节点 3.带宽 4.调制格式 5.SC 6.源流序号 7.hub_P2MP 8.leaf_P2MP 9.SC起始 10.SC末尾 11.FS起始 12.FS末尾
            copy_Sdes_SC = copy.deepcopy(des_SC)
            copy_Sdes_SC = np.array(copy_Sdes_SC)
            copy_Sdes_band = copy.deepcopy(des_band)

            node_P2MP[n][h][2] = 1
            P2MP_tree = []
            # P2MP级别的SC分配和FS分配
            value_P2MP, bag_P2MP, final_value_P2MP, final_items_P2MP = knapsack_DP(type_P2MP[node_P2MP[n][h][3] - 1][1], copy_Sdes_SC)
            # ====== FIX: 未被该 hub-P2MP 接纳的目的地，其流必须回炉重算，否则会“消失” ======
            selected_des = set(int(x) for x in final_items_P2MP)
            for des_id in range(topo_num):
                if des_id not in selected_des and len(copy_Sdes_band[des_id][0]) > 0:
                    # 这些流此前已从 B_flows_info pop 掉；这里必须放回去
                    B_flows_info.extend(copy_Sdes_band[des_id][0])
            # ==========================================================================
            FS_end = -1
            for i in final_items_P2MP:
                FS_start = FS_end + 1
                SC_start = map_fs_to_sc(FS_start, node_P2MP[n][h][3])
                if SC_start == None:
                    for f in copy_Sdes_band[i][0]:
                        B_flows_info.append(f)
                    continue
                SC_spare = copy_Sdes_SC[i][3]
                SC_spare_0 = SC_spare
                for f in copy_Sdes_band[i][0]:
                    if f[3] <= SC_spare:
                        SC_end = SC_start
                        SC_spare = SC_spare - f[3]
                    else:
                        SC_end = math.ceil((f[3] - SC_spare) / copy_Sdes_SC[i][3]) + SC_start
                        SC_spare = copy_Sdes_SC[i][3] * (SC_end - SC_start) - (f[3] - SC_spare)
                    if SC_end <= 15:
                        FS_start = sc_fs(node_P2MP[n][h][3], SC_start, 1)
                        FS_end = sc_fs(node_P2MP[n][h][3], SC_end, 2)
                        node_flow[n][h][0].append(f)
                        flow_acc[f[0]][7] = h
                        flow_acc[f[0]][9] = SC_start
                        flow_acc[f[0]][10] = SC_end
                        flow_acc[f[0]][11] = FS_start
                        flow_acc[f[0]][12] = FS_end

                        # 记录使用SC的流和使用大小
                        if SC_start == SC_end:
                            P2MP_SC[n][h][SC_start][1].append([f[0], f[3], f[6]])
                            P2MP_SC[n][h][SC_start][0].append(copy_Sdes_SC[i][3])
                            P2MP_SC[n][h][SC_start][2].append(copy_Sdes_band[i][1][0])
                            P2MP_SC[n][h][SC_start][4].append(f[2])
                        else:
                            P2MP_SC[n][h][SC_start][1].append([f[0], SC_spare_0, f[6]])
                            P2MP_SC[n][h][SC_start][0].append(copy_Sdes_SC[i][3])
                            P2MP_SC[n][h][SC_start][2].append(copy_Sdes_band[i][1][0])
                            P2MP_SC[n][h][SC_start][4].append(f[2])
                            for s in range(SC_start + 1, SC_end):
                                P2MP_SC[n][h][s][1].append([f[0], copy_Sdes_SC[i][3], f[6]])
                                P2MP_SC[n][h][s][0].append(copy_Sdes_SC[i][3])
                                P2MP_SC[n][h][s][2].append(copy_Sdes_band[i][1][0])
                                P2MP_SC[n][h][s][4].append(f[2])
                            P2MP_SC[n][h][SC_end][1].append([f[0], copy_Sdes_SC[i][3] - SC_spare, f[6]])
                            P2MP_SC[n][h][SC_end][0].append(copy_Sdes_SC[i][3])
                            P2MP_SC[n][h][SC_end][2].append(copy_Sdes_band[i][1][0])
                            P2MP_SC[n][h][SC_end][4].append(f[2])

                        used_links = get_links_from_path(copy_Sdes_band[i][1][0], link_index)
                        used_links = [t - 1 for t in used_links]
                        flow_path[f[0]][0].append(f[0])
                        flow_path[f[0]][1].extend(used_links)
                        flow_path[f[0]][2].extend(copy_Sdes_band[i][1][0])
                        flow_path[f[0]][3].append(f[6])
                        P2MP_tree.extend(used_links)
                        SC_spare_0 = SC_spare #记录的是上一条流的剩余SC_spare
                        SC_start = SC_end
                        if SC_spare == 0:
                            SC_start = SC_end + 1
                            SC_spare = copy_Sdes_SC[i][3]
                            SC_spare_0 = SC_spare
                    else:
                        B_flows_info.append(f)
            # 判断使用的中心频率
            P2MP_tree = list(set(P2MP_tree))
            combined_usage = link_FS[P2MP_tree].sum(axis=0)
            # 寻找连续 block_size 个 0 的起始位置
            block_size = type_P2MP[node_P2MP[n][h][3] - 1][2]
            for i in range(len(combined_usage) - block_size + 1):
                if np.all(combined_usage[i:i + block_size] == 0):
                    free_block_indices = i
                    break
            node_P2MP[n][h][5] = free_block_indices

            # 更新使用的FS信息 + flew_acc中FS的使用信息
            for f in node_flow[n][h][0]:
                flow_acc[f[0]][11] = node_P2MP[n][h][5] + flow_acc[f[0]][11]
                flow_acc[f[0]][12] = node_P2MP[n][h][5] + flow_acc[f[0]][12]
                path = k_shortest_path(topo_dis, n + 1, f[2] + 1, 1) # 问题：可能会存在这边的path和之前的des_band的path不一致的情况
                used_links = get_links_from_path(path[0][0], link_index)
                for l in used_links:
                    link_FS[l - 1][flow_acc[f[0]][11] : flow_acc[f[0]][12] + 1] = \
                        link_FS[l - 1][flow_acc[f[0]][11] : flow_acc[f[0]][12] + 1] + 1

            h = h + 1


    # leaf-P2MP分配
    for n in range(topo_num):
        for h in range(Tbox_num * Tbox_P2MP):
            if node_P2MP[n][h][2] == 1:
                col = P2MP_SC[n, h, :, -1]
                col_filtered = [x for x in col if not (isinstance(x, (list, np.ndarray)) and len(x) == 0)]
                unique_vals = np.unique(col_filtered)
                result = []
                for val in unique_vals:
                    rows = np.where([x == [val] for x in col])[0]  # 取这个值出现的所有行号
                    first, last = rows[0], rows[-1]
                    FS_start = sc_fs(node_P2MP[n][h][3], first, 1)
                    result.append([val, first, last, node_P2MP[n][h][5] + FS_start])
                for r in result:
                    des = r[0]
                    for p in range(Tbox_num * Tbox_P2MP):
                        p_type = node_P2MP[des][p][3] - 1
                        FS_end = sc_fs(p_type, r[2] - r[1], 2)
                        if node_P2MP[des][p][2] == 0 and FS_end <= type_P2MP[p_type - 1][2]:
                            P2MP_SC[des, p, :r[2]-r[1]+1, :] = P2MP_SC[n, h, r[1]:r[2]+1, :]
                            for f in node_flow[n][h][0]:
                                if f[2] == des:
                                    node_flow[des][p][0].append(f)
                                    flow_acc[f[0]][8] = p
                            node_P2MP[des][p][2] = 2
                            node_P2MP[des][p][5] = r[3]
                            break

    return node_flow, node_P2MP, flow_acc, link_FS, P2MP_SC, flow_path

