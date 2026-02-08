#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : code 
# @File    : FlexE_P2MP_BM.py
# @Author  : Wumh
# @Time    : 2024/9/12 22:30
import numpy as np
import copy
import math
from k_shortest_path import k_shortest_path
from modu_format_Al import modu_format_Al


def FlexE_P2MP_BM(flows_info, flows_num, topo_num, topo_dis, Tbox_num, Tbox_P2MP):
    # 设置参数
    type_P2MP = [[1, 1], [2, 4], [3, 16]]

    # 创建记录的信息表
    # 1.node_P2MP: 0.T-box序号 1.P2MP序号 2.P2MP是否使用 3.型号 4.发送端使用大小 5.发送端剩余容量 6.发送端使用的SC个数
    #              7.发送端SC容量 8.接收端使用大小 9.接收端剩余容量 10.接收端使用的SC个数 11.接收端SC容量
    # 2.node_flow: 0.P2MP容纳的流序号: 0.flow序号 1.源节点 2.目的节点 3.带宽 4.调制格式 5.SC
    # 3.flow_acc: 0.flow序号 1.源节点 2.目的节点 3.带宽 4.调制格式 5.SC 6.hub_P2MP 7.leaf_P2MP
    node_P2MP = np.zeros((topo_num, Tbox_num * 2, 12), dtype=int)
    node_flow = np.empty((topo_num, Tbox_num * 2, 1), dtype=object)
    flow_acc = np.zeros((flows_num, 8), dtype=int)
    node_flow_sc = np.zeros((topo_num, Tbox_num * Tbox_P2MP, topo_num, 4), dtype=int)

    for n in range(topo_num):
        for p in range(Tbox_num * Tbox_P2MP):
            for i in range(topo_num):
                node_flow_sc[n][p][i][0] = i
                path = k_shortest_path(topo_dis, n + 1, i + 1, 1)
                SC_cap, SC_num = modu_format_Al(path[0][1], 0)
                node_flow_sc[n][p][i][1] = SC_cap

    for i in range(topo_num):
        for j in range(Tbox_num * 2):
            node_P2MP[i][j][0] = math.floor(j / 2)
            node_P2MP[i][j][1] = j
            node_P2MP[i][j][5] = 400
            node_P2MP[i][j][7] = 16
            node_P2MP[i][j][9] = 400
            node_P2MP[i][j][11] = 16
            node_flow[i][j][0] = []

    for i in range(flows_num):
        flow_acc[i][0:6] = flows_info[i][0:6]
        flow_acc[i][6] = -1
        flow_acc[i][7] = -1

    for f in flows_info:
        flow_index = f[0]
        n = flows_info[flow_index][1]
        des = flows_info[flow_index][2]
        band = flows_info[flow_index][3]
        modu = flows_info[flow_index][4]
        sc = flows_info[flow_index][5]
        for p in range(Tbox_num * Tbox_P2MP):
            if node_P2MP[n][p][2] == 0 and node_P2MP[n][p][5] >= band and node_P2MP[n][p][7] >= sc:
                node_flow[n][p][0].append(flows_info[flow_index])
                flow_acc[flow_index][6] = p
                node_P2MP[n][p][2] = 1
                node_P2MP[n][p][4] = band
                node_P2MP[n][p][5] = node_P2MP[n][p][5] - band
                node_P2MP[n][p][6] = sc
                node_flow_sc[n][p][des][2] = band
                node_flow_sc[n][p][des][3] = sc
                if p % 2 == 0:
                    node_P2MP[n][p + 1][5] = node_P2MP[n][p][5]
                else:
                    node_P2MP[n][p - 1][5] = node_P2MP[n][p][5]
                break

            elif node_P2MP[n][p][2] == 1:
                temp_1 = copy.deepcopy(node_flow_sc[n][p])
                temp_1[des][2] = temp_1[des][2] + band
                temp_1[des][3] = math.ceil(temp_1[des][2] / 25)
                if temp_1[des][1] == 12:
                    temp_1[des][3] = math.ceil(temp_1[des][2] / 12.5)
                sc_sum = np.sum(temp_1[:, 3])

                if node_P2MP[n][p][5] >= band and node_P2MP[n][p][7] >= sc_sum:
                    node_flow[n][p][0].append(flows_info[flow_index])
                    flow_acc[flow_index][6] = p
                    node_P2MP[n][p][4] = node_P2MP[n][p][4] + band
                    node_P2MP[n][p][5] = node_P2MP[n][p][5] - band
                    node_P2MP[n][p][6] = sc_sum
                    node_flow_sc[n][p] = copy.deepcopy(temp_1)
                    if p % 2 == 0:
                        node_P2MP[n][p + 1][5] = node_P2MP[n][p][5]
                    else:
                        node_P2MP[n][p - 1][5] = node_P2MP[n][p][5]
                    break

    # leaf-P2MP分配
    for i in range(topo_num):
        for p in range(Tbox_num * Tbox_P2MP):
            if node_P2MP[i][p][2] == 1:
                for j in range(topo_num):
                    band_sum = node_flow_sc[i][p][j][2]
                    sc_sum = node_flow_sc[i][p][j][3]
                    if band_sum > 0:
                        for pp in range(int((Tbox_num * Tbox_P2MP) / 2), Tbox_num * Tbox_P2MP):
                            if node_P2MP[j][pp][2] == 0 and node_P2MP[j][pp][9] >= band_sum:
                                node_P2MP[j][pp][2] = 2
                                node_P2MP[j][pp][8] = band_sum
                                node_P2MP[j][pp][9] = node_P2MP[j][pp][9] - band_sum
                                node_P2MP[j][pp][10] = sc_sum
                                if pp % 2 == 0:
                                    node_P2MP[j][pp + 1][9] = node_P2MP[j][pp][9]
                                else:
                                    node_P2MP[j][pp - 1][9] = node_P2MP[j][pp][9]
                                for f in node_flow[i][p][0]:
                                    if f[2] == j:  # 保证接收端的T-BOX要处理的流的大小不超过400Gbps
                                        node_flow[j][pp][0].append(f)
                                        flow_acc[f[0]][7] = pp
                                break

    # 更新node_P2MP_2中未被确定的类型
    for n in range(topo_num):
        for p in range(Tbox_num * Tbox_P2MP):
            if node_P2MP[n][p][2] == 1:
                if node_P2MP[n][p][6] > 16:
                    print("P2MP: 流在 node %d hub %d 的SC总数超过16" % (n, p))
                elif 4 < node_P2MP[n][p][6] <= 16:
                    node_P2MP[n][p][3] = 3
                elif 1 < node_P2MP[n][p][6] <= 4:
                    node_P2MP[n][p][3] = 2
                else:
                    node_P2MP[n][p][3] = 1

            elif node_P2MP[n][p][2] == 2:
                if node_P2MP[n][p][10] > 16:
                    print("P2MP: 流在 node %d hub %d 的SC总数超过16" % (n, p))
                elif 4 < node_P2MP[n][p][10] <= 16:
                    node_P2MP[n][p][3] = 3
                elif 1 < node_P2MP[n][p][10] <= 4:
                    node_P2MP[n][p][3] = 2
                else:
                    node_P2MP[n][p][3] = 1

    return node_flow, node_P2MP, flow_acc