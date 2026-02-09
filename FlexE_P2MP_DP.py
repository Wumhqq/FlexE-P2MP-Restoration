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

def FlexE_P2MP_DP(flows_info, flows_num, topo_num, topo_dis, Tbox_num, Tbox_P2MP):
    # 设置参数
    type_P2MP = [[1, 1], [2, 4], [3, 16]]

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
                scr_flows[n].append(flow)

    # 创建记录的信息表
    # 1.node_P2MP: 0.T-box序号 1.P2MP序号 2.P2MP是否使用 3.型号 4.接收端剩余容量
    # 2.node_flow: 0.P2MP容纳的流序号: 0.flow序号 1.源节点 2.目的节点 3.带宽 4.调制格式 5.SC
    # 3.flow_acc: 0.flow序号 1.源节点 2.目的节点 3.带宽 4.调制格式 5.SC 6.hub_P2MP 7.leaf_P2MP
    node_P2MP = np.zeros((topo_num, Tbox_num * 2, 5), dtype=int)
    node_flow = np.empty((topo_num, Tbox_num * 2, 1), dtype=object)
    flow_acc = np.zeros((flows_num, 8), dtype=int)
    for i in range(topo_num):
        for j in range(Tbox_num * 2):
            node_P2MP[i][j][0] = math.floor(j / 2)
            node_P2MP[i][j][1] = j
            node_P2MP[i][j][4] = 400
            node_flow[i][j][0] = []
    for i in range(flows_num):
        flow_acc[i][0:6] = flows_info[i][:]

    # 分配hub资源
    for n in range(topo_num):
        Tbox_index = 0
        B_flows_info = copy.deepcopy(scr_flows[n])
        while(len(B_flows_info) > 0):
            value, bag, final_value, final_items = knapsack_DP(400, B_flows_info)
            final_items_info = []
            for i in final_items:
                final_items_info.append(flows_info[i][:])
                for index, j in enumerate(B_flows_info):
                    if j[0] == i:
                        B_flows_info.pop(index)
                        break

            # 以不同的目的地为基准计算使用的SC
            des_band = np.empty((topo_num, 1), dtype=object)
            for i in range(topo_num):
                des_band[i][0] = []
            des_SC = np.zeros((topo_num, 3), dtype=int)
            for i in final_items_info:
                des_band[i[2]][0].append(i)
                des_SC[i[2]][2] = des_SC[i[2]][2] + i[3]
            for i in range(topo_num):
                des_SC[i][0] = i
                path = k_shortest_path(topo_dis, n + 1, i + 1, 1)
                SC_cap, SC_num = modu_format_Al(path[0][1], des_SC[i][2])
            # 根据总SC确定P2MP的使用并更新node_P2MP
            # 如果总SC超过16，我们需要多个P2MP或拆分流
            sum = np.sum(des_SC[:, 1])
            # 注意：不要一开始就确定P2MP类型；先填满一个P2MP，再填下一个
            copy_Sdes_SC = copy.deepcopy(des_SC)
            copy_Sdes_SC = np.array(copy_Sdes_SC)
            copy_Sdes_band = copy.deepcopy(des_band)
            for h in range(Tbox_P2MP):
                p = Tbox_index * 2 + h
                # 情况1：所需总SC > 16。需要优化填充一个P2MP（容量16 SC）
                if sum > 16:
                    node_P2MP[n][p][2] = 1 # 标记P2MP为已使用
                    node_P2MP[n][p][3] = 3 # 分配最大的P2MP类型（16 SC）
                    s = type_P2MP[2][1]
                    temp = np.zeros((topo_num, 4), dtype=int)
                    # 计算每个目的地的SC需求
                    for i in range(topo_num):
                        temp[i][0] = i
                        path = k_shortest_path(topo_dis, n + 1, i + 1, 1)
                        SC_cap, SC_num = modu_format_Al(path[0][1], 0)
                        temp[i][1] = SC_cap

                    
                    # 使用背包DP选择最适合放入16 SC的流
                    value_P2MP, bag_P2MP, final_value_P2MP, final_items_P2MP = knapsack_DP(16, copy_Sdes_SC)
                    for i in final_items_P2MP:
                        # 将选定的流添加到当前P2MP
                        node_flow[n][p][0].extend(copy_Sdes_band[i][0])
                        copy_Sdes_band[i][0] = [] # 清除已处理的流
                        # 在temp中存储原始SC信息，以便通过后续步骤回滚/重新计算
                        temp[i][2] = copy_Sdes_SC[i][2]
                        temp[i][3] = copy_Sdes_SC[i][1]
                        # 在copy_Sdes_SC中标记为已处理
                        copy_Sdes_SC[i][1] = 0
                        copy_Sdes_SC[i][2] = 0
                    # 如果空间允许，尝试添加更多流（贪婪优化）
                    copy2_Sdes_band = copy.deepcopy(copy_Sdes_band)
                    for i in range(topo_num):
                        for f in copy2_Sdes_band[i][0]:
                            copy_temp = copy.deepcopy(temp)
                            copy_temp[i][2] = copy_temp[i][2] + f[3]
                            # 重新计算所需SC
                            copy_temp[i][3] = math.ceil(copy_temp[i][2] / 25)
                            if copy_temp[i][1] == 12:
                                copy_temp[i][3] = math.ceil(copy_temp[i][2] / 12.5)
                            
                            # 检查添加此流是否超过16 SC限制
                            if np.sum(copy_temp[:, 3]) <= 16:
                                node_flow[n][p][0].append(f)
                                for index, jj in enumerate(copy_Sdes_band[i][0]):
                                    if jj[0] == f[0]:
                                        copy_Sdes_band[i][0].pop(index)
                                        break
                                temp[i][2] = copy_temp[i][2]
                                temp[i][3] = copy_temp[i][3]
                                # 更新剩余SC需求
                                copy_Sdes_SC[i][2] = copy_Sdes_SC[i][2] - f[3]
                                copy_Sdes_SC[i][1] = math.ceil(copy_Sdes_SC[i][2] / 25)
                                if temp[i][1] == 12:
                                    copy_Sdes_SC[i][1] = math.ceil(copy_Sdes_SC[i][2] / 12.5)
                    sum = np.sum(copy_Sdes_SC[:, 1])

                # 情况2：剩余SC <= 16。可以放入一个P2MP。
                elif 0 < sum <= 16:
                    node_P2MP[n][p][2] = 1
                    if np.sum(copy_Sdes_SC[:, 1]) > 16:
                        print("error_hub: if分类出现问题")
                    # 根据所需SC确定P2MP类型
                    elif 4 < np.sum(copy_Sdes_SC[:, 1]) <= 16:
                        node_P2MP[n][p][3] = 3 # 类型3：16 SC
                    elif 1 < np.sum(copy_Sdes_SC[:, 1]) <= 4:
                        node_P2MP[n][p][3] = 2 # 类型2：4 SC
                    else:
                        node_P2MP[n][p][3] = 1 # 类型1：1 SC（单波长/低容量）
                        
                    s = type_P2MP[node_P2MP[n][p][3] - 1][1]
                    # 将剩余流分配给此P2MP
                    for i in copy_Sdes_SC:
                        if i[1] != 0 and i[1] <= s:
                            node_flow[n][p][0].extend(copy_Sdes_band[i[0]][0])
                            copy_Sdes_band[i[0]][0] = []
                            s = s - i[1] # 扣除已用容量
                            i[1] = 0
                        elif i[1] > s:
                            print("error_hub: T-box中2-thP2MP的SC分配出现问题")
                    sum = 0
                else:
                    continue
            # 恢复多删的流
            for nn in range(topo_num):
                for i in copy_Sdes_band[nn][0]:
                    if len(i) != 0:
                        B_flows_info.append(i)

            Tbox_index = Tbox_index + 1

    # 更新flow_acc
    for f in flow_acc:
        for n in range(topo_num):
            for p in range(Tbox_num * Tbox_P2MP):
                for i in node_flow[n][p][0]:
                    if len(i) > 0:
                        if i[0] == f[0]:
                            f[6] = p

    # leaf-P2MP分配（接收侧）
    for i in range(topo_num):
        for p in range(Tbox_num * Tbox_P2MP):
            # 如果源节点(i)使用了P2MP
            if node_P2MP[i][p][2] == 1:
                # 查找此P2MP中流的目的节点(j)
                for j in range(topo_num):
                    band_sum = 0
                    k = 0
                    sum = 0
                    # 计算目的地j的总带宽
                    for f in node_flow[i][p][0]:
                        if f[2] == j:
                            sum = sum + f[3]

                    # 在目的地j分配接收P2MP
                    # 遍历可用的P2MP插槽（使用Tbox的后半部分）
                    for pp in range(int((Tbox_num * Tbox_P2MP) / 2), Tbox_num * Tbox_P2MP):
                        # 检查P2MP是否未使用且有足够容量（剩余<= 400G）
                        if node_P2MP[j][pp][2] == 0 and sum <= node_P2MP[j][pp][4]:
                            for f in node_flow[i][p][0]:
                                if f[2] == j: # 确保接收端T-BOX流不超过400Gbps
                                    node_P2MP[j][pp][2] = 2 # 标记为接收端已使用
                                    node_P2MP[j][pp][4] = node_P2MP[j][pp][4] - f[3] # 更新剩余容量
                                    # 同步配对P2MP的剩余容量
                                    if pp % 2 == 0:
                                        node_P2MP[j][pp + 1][4] = node_P2MP[j][pp][4]
                                    else:
                                        node_P2MP[j][pp - 1][4] = node_P2MP[j][pp][4]
                                    
                                    node_flow[j][pp][0].append(f)
                                    flow_acc[f[0]][7] = pp # 记录Leaf P2MP ID
                                    band_sum = band_sum + f[3]
                                    k = f[4] # 调制格式
                            
                            # 根据接收带宽确定Leaf P2MP类型
                            if band_sum > 0:
                                kk = math.ceil(band_sum/25)
                                if k == 12:
                                    kk = math.ceil(band_sum/12.5)
                                if kk > 16:
                                    print("error_leaf: 单流SC超过16")
                                elif 4 < kk <= 16:
                                    node_P2MP[j][pp][3] = 3
                                elif 1 < kk <= 4:
                                    node_P2MP[j][pp][3] = 2
                                else:
                                    node_P2MP[j][pp][3] = 1
                            break

    return node_flow, node_P2MP, flow_acc