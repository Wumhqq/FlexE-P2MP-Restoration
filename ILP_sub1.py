#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : ILP.py 
# @File    : ILP_sub1.py
# @Author  : Wumh
# @Time    : 2025/12/30 23:05
import gurobipy as gp  # 导入Gurobi的Python接口，并缩写为gp
import numpy as np
import SC_FS_list
import topology as tp
from gurobipy import GRB  # 导入Gurobi中的常量

def Sub_problem(affected_flow, new_flow_acc, new_node_flow, new_node_P2MP, break_node, Tbox_num, Tbox_P2MP,
                    new_P2MP_SC_1, new_link_FS, node_flow_DP, node_P2MP_DP, flow_acc_DP, link_FS_DP, P2MP_SC_DP, flow_path_DP,
                Reconf_f, scr_u, des_u, Modu_uv, PHY_uv):
    # 参数
    flow_num = len(affected_flow)
    M = 10000
    P2MP_num = Tbox_num * Tbox_P2MP
    max_sc = 16
    max_sc_fs = 6
    max_fs = 352
    # sc_fs_start = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5]]
    # sc_fs_end = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #              [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #              [0 ,1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]]
    topo_num, topo_matrix, topo_dis, link_num, link_index = tp.topology(1)
    type_P2MP = [[1, 1, 1], [2, 4, 2], [3, 16, 6]]

    # 因为不能传inf，所以需要把topo_dis的inf转换成0
    topo_dis[topo_dis == np.inf] = 0
    # 把SC的传输路径转换成邻接矩阵
    SC_path = np.zeros((topo_num, P2MP_num, max_sc, topo_num, topo_num), dtype=int)
    for u in range(topo_num):
        for p in range(P2MP_num):
            for s in range(max_sc):
                if len(new_P2MP_SC_1[u][p][s][0]) != 0:
                    path = new_P2MP_SC_1[u][p][s][2][0]
                    path = [x - 1 for x in path]
                    for i in range(len(path) - 1):
                        v_1, v_2 = path[i], path[i + 1]
                        SC_path[u][p][s][v_1][v_2] = 1
    FS_path = np.zeros((topo_num, P2MP_num, max_sc_fs, topo_num, topo_num), dtype=int)
    for u in range(topo_num):
        for p in range(P2MP_num):
            p_type = new_node_P2MP[u][p][3]
            sc_fs_list = SC_FS_list.sc_fs_list(p_type)
            for s in range(max_sc):
                for w in range(max_sc_fs):
                    if sc_fs_list[s][w] == 1 and len(new_P2MP_SC_1[u][p][s][0]) != 0:
                        FS_path[u, p, w, :, :] = SC_path[u, p, s, :, :]

    # 处理流的路径和调制格式的格式
    pre_flow_path = np.zeros((flow_num, topo_num, topo_num))
    pre_flow_modu = np.zeros((flow_num, topo_num, topo_num))
    # 0.原流序号 1.源节点 2.目的节点 3.源节点使用的FlexE 4.目的节点使用的FlexE 5.源节点使用的SC起始 6.源节点使用的SC结束 7.目的节点使用的SC起始 8.目的节点使用的SC结束
    pre_flow_info = np.zeros((flow_num, 8))
    i = 0
    for f in affected_flow:
        index = f[0]
        for ff in flow_path_DP:
            if ff[3][0] == index:
                path_len = 0
                for n in range(len(ff[2]) - 1):
                    path_len = path_len + topo_dis[ff[2][n] - 1][ff[2][n + 1] - 1]
                pre_flow_path[i][ff[2][0] - 1][ff[2][-1] - 1] = path_len
                if path_len <= 500:
                    pre_flow_modu[i][ff[2][0] - 1][ff[2][-1] - 1] = 1
                else:
                    pre_flow_modu[i][ff[2][0] - 1][ff[2][-1] - 1] = 2
        # 在初始网络场景中，使用的是中断流进行分配，现在要在整条流的角度上整合信息
        for ff in flow_acc_DP:
            if ff[6] == index and ff[1] == f[1]:
                pre_flow_info[i][0] = index
                pre_flow_info[i][1] = f[1]
                pre_flow_info[i][3] = ff[7]
            if ff[6] == index and ff[2] == f[2]:
                pre_flow_info[i][2] = f[2]
                pre_flow_info[i][4] = ff[8]
        i = i + 1

    # 更新以邻接矩阵为索引的流使用的FS，以及链路上使用FS的P2MP
    pre_FS_scr = np.zeros((topo_num, topo_num, max_fs, topo_num, P2MP_num))
    pre_FS_des = np.zeros((topo_num, topo_num, max_fs, topo_num, P2MP_num))
    for f in range(len(new_flow_acc)):
        if new_flow_acc[f, 7] != -1:
            path = flow_path_DP[f, 2]
            path = [x - 1 for x in path]
            for w in range(new_flow_acc[f, 11], new_flow_acc[f, 12] + 1):
                for p in range(len(path) - 1):
                    pre_FS_scr[path[p], path[p + 1], w, new_flow_acc[f, 1], new_flow_acc[f, 7]] = 1
                    pre_FS_des[path[p], path[p + 1], w, new_flow_acc[f, 2], new_flow_acc[f, 8]] = 1


    # 创建模型
    model = gp.Model("Subproblem_1")
    # 恢复第一策略流量，需要更新 1.FlexE组 2.P2MP的SC 3.频谱资源
    # FlexE组：分配FlexE Group资源，要考虑第一策略
    # FlexE_f = model.addVars(flow_num, topo_num, P2MP_num, lb=0, ub=1, vtype=GRB.BINARY)
    FlexE_scr = model.addVars(flow_num, topo_num, P2MP_num, lb=0, ub=1, vtype=GRB.BINARY)
    FlexE_des = model.addVars(flow_num, topo_num, P2MP_num, lb=0, ub=1, vtype=GRB.BINARY)
    SC_scr_1 = model.addVars(flow_num, topo_num, P2MP_num, max_sc, lb=0, ub=1, vtype=GRB.BINARY)  # 是否使用
    SC_des_1 = model.addVars(flow_num, topo_num, P2MP_num, max_sc, lb=0, ub=1, vtype=GRB.BINARY)
    SC_scr_2 = model.addVars(flow_num, topo_num, P2MP_num, max_sc, lb=0, vtype=GRB.INTEGER)  # 使用的大小
    SC_des_2 = model.addVars(flow_num, topo_num, P2MP_num, max_sc, lb=0, vtype=GRB.INTEGER)
    z2 = model.addVars(topo_num, P2MP_num, max_sc, lb=0, ub=1, vtype=GRB.BINARY)  # 辅助变量，指示该SC只可当作发送/接收SC
    SC_slow = model.addVars(topo_num, P2MP_num, max_sc, vtype=GRB.BINARY,
                            name="slow")  # 判断该SC是否使用低阶调制格式 Modu = 2 --> SC_slow = 1
    SC_cap = model.addVars(topo_num, P2MP_num, max_sc, lb=0, vtype=GRB.CONTINUOUS)
    equal_PHY = model.addVars(flow_num, flow_num, topo_num, topo_num, topo_num, lb=0, ub=1, vtype=GRB.BINARY)
    SC_scr_start = model.addVars(flow_num, topo_num, P2MP_num, vtype=GRB.INTEGER, lb=0, ub=max_sc - 1)
    SC_scr_end = model.addVars(flow_num, topo_num, P2MP_num, vtype=GRB.INTEGER, lb=0, ub=max_sc - 1)
    SC_des_start = model.addVars(flow_num, topo_num, P2MP_num, vtype=GRB.INTEGER, lb=0, ub=max_sc - 1)
    SC_des_end = model.addVars(flow_num, topo_num, P2MP_num, vtype=GRB.INTEGER, lb=0, ub=max_sc - 1)
    FS_scr_1 = model.addVars(flow_num, topo_num, P2MP_num, max_sc_fs, lb=0, ub=1, vtype=GRB.BINARY)  # 是否使用
    FS_des_1 = model.addVars(flow_num, topo_num, P2MP_num, max_sc_fs, lb=0, ub=1, vtype=GRB.BINARY)  # 是否使用
    FS_scr_start = model.addVars(flow_num, topo_num, P2MP_num, vtype=GRB.INTEGER, lb=0, ub=max_sc_fs - 1)
    FS_scr_end = model.addVars(flow_num, topo_num, P2MP_num, vtype=GRB.INTEGER, lb=0, ub=max_sc_fs - 1)
    FS_des_start = model.addVars(flow_num, topo_num, P2MP_num, vtype=GRB.INTEGER, lb=0, ub=max_sc_fs - 1)
    FS_des_end = model.addVars(flow_num, topo_num, P2MP_num, vtype=GRB.INTEGER, lb=0, ub=max_sc_fs - 1)
    FS_P2MP_start = model.addVars(topo_num, P2MP_num, lb=0, ub=max_fs - 1, vtype=GRB.INTEGER)
    FS_uv_start = model.addVars(flow_num, topo_num, topo_num, topo_num, vtype=GRB.INTEGER, lb=0, ub=max_fs - 1)  # 是否使用
    FS_uv_end = model.addVars(flow_num, topo_num, topo_num, topo_num, vtype=GRB.INTEGER, lb=0, ub=max_fs - 1)
    FS_uv_w = model.addVars(flow_num, topo_num, topo_num, topo_num, max_fs, vtype=GRB.BINARY)
    FS_edge_scr = model.addVars(topo_num, topo_num, max_fs, topo_num, P2MP_num,
                                vtype=GRB.BINARY)  # 某条链路上的某个fs被源节点上某个P2MP使用
    FS_edge_des = model.addVars(topo_num, topo_num, max_fs, topo_num, P2MP_num,
                                vtype=GRB.BINARY)  # 某条链路上的某个fs被目的节点上某个P2MP使用


    for f in range(flow_num):
        for u in range(topo_num):
            # 1. 保证在该scr或者des上需要分配TS
            model.addConstr(gp.quicksum(FlexE_scr[f, u, p] for p in range(P2MP_num)) == sum(
                scr_u[f, i, u] for i in range(topo_num)))
            model.addConstr(gp.quicksum(FlexE_des[f, u, p] for p in range(P2MP_num)) == sum(
                des_u[f, i, u] for i in range(topo_num)))
            if u == affected_flow[f][1]:
                p = pre_flow_info[f][3]
                model.addConstr(FlexE_scr[f, u, p] <= 1 + Reconf_f[f])
                model.addConstr(FlexE_scr[f, u, p] >= 1 - Reconf_f[f])
            if u == affected_flow[f][2]:
                p = pre_flow_info[f][4]
                model.addConstr(FlexE_des[f, u, p] <= 1 + Reconf_f[f])
                model.addConstr(FlexE_des[f, u, p] >= 1 - Reconf_f[f])
    for u in range(topo_num):
        for p in range(P2MP_num):
            # 2. 保证在该scr或者des上的FlexE Group上分配的流小于FlexE Group的容量
            model.addConstr(gp.quicksum((FlexE_scr[f, u, p] * affected_flow[f][3]) for f in range(flow_num)) +
                            gp.quicksum((FlexE_des[f, u, p] * affected_flow[f][3]) for f in range(flow_num)) <=
                            new_node_P2MP[u][p][4])

    # 分配SC资源，要考虑第一策略
    for f in range(flow_num):
        for u in range(topo_num):
            for p in range(P2MP_num):
                # 1. 保证使用该P2MP的流使用该P2MP的SC
                model.addConstr(FlexE_scr[f, u, p] <= gp.quicksum(SC_scr_1[f, u, p, s] for s in range(max_sc)))
                model.addConstr(gp.quicksum(SC_scr_1[f, u, p, s] for s in range(max_sc)) <= FlexE_scr[f, u, p] * M)
                model.addConstr(FlexE_des[f, u, p] <= gp.quicksum(SC_des_1[f, u, p, s] for s in range(max_sc)))
                model.addConstr(gp.quicksum(SC_des_1[f, u, p, s] for s in range(max_sc)) <= FlexE_des[f, u, p] * M)
                # 第一策略使用的是确定的SC和确定的SC大小
                if u == pre_flow_info[f][1] and p == pre_flow_info[f][3]:
                    for s in range(max_sc):
                        if len(P2MP_SC_DP[u][p][s][1]) != 0:
                            for ff in P2MP_SC_DP[u][p][s][1]:
                                if ff[2] == affected_flow[f][0]:
                                    model.addConstr(SC_scr_1[f, u, p, s] <= 1 + Reconf_f[f])
                                    model.addConstr(SC_scr_1[f, u, p, s] >= 1 - Reconf_f[f])
                                    model.addConstr(SC_scr_2[f, u, p, s] <= (ff[1] / 5) + Reconf_f[f] * M)
                                    model.addConstr(SC_scr_2[f, u, p, s] >= (ff[1] / 5) - Reconf_f[f] * M)

                if u == pre_flow_info[f][2] and p == pre_flow_info[f][4]:
                    for s in range(max_sc):
                        if len(P2MP_SC_DP[u][p][s][1]) != 0:
                            for ff in P2MP_SC_DP[u][p][s][1]:
                                if ff[2] == affected_flow[f][0]:
                                    model.addConstr(SC_des_1[f, u, p, s] <= 1 + Reconf_f[f])
                                    model.addConstr(SC_des_1[f, u, p, s] >= 1 - Reconf_f[f])
                                    model.addConstr(SC_des_2[f, u, p, s] <= (ff[1] / 5) + Reconf_f[f] * M)
                                    model.addConstr(SC_des_2[f, u, p, s] >= (ff[1] / 5) - Reconf_f[f] * M)
                for s in range(max_sc):
                    # 2. 保证使用该SC的流使用该SC的大小>=0
                    model.addConstr(SC_scr_1[f, u, p, s] <= SC_scr_2[f, u, p, s])
                    model.addConstr(SC_scr_2[f, u, p, s] <= SC_scr_1[f, u, p, s] * M)
                    model.addConstr(SC_des_1[f, u, p, s] <= SC_des_2[f, u, p, s])
                    model.addConstr(SC_des_2[f, u, p, s] <= SC_des_1[f, u, p, s] * M)
            # 3. 保证使用SC的大小满足流的需求，按照TS的个数来分配
            model.addConstr(gp.quicksum(SC_scr_2[f, u, p, s] for s in range(max_sc) for p in range(P2MP_num))
                            == affected_flow[f][3] / 5 * sum(scr_u[f, i, u] for i in range(topo_num)))
            model.addConstr(gp.quicksum(SC_des_2[f, u, p, s] for s in range(max_sc) for p in range(P2MP_num))
                            == affected_flow[f][3] / 5 * sum(des_u[f, i, u] for i in range(topo_num)))

    # 4. 保证使用该SC的流不超过该SC的容量
    # 4.1 Modu=2  ⇒ slow = 1
    for f in range(flow_num):
        for u in range(topo_num):
            for p in range(P2MP_num):
                for s in range(max_sc):
                    model.addConstr(SC_slow[u, p, s] >= SC_scr_1[f, u, p, s] + sum(Modu_uv[f, u, v] for v in range(topo_num)) - 2)
                    model.addConstr(SC_slow[u, p, s] >= SC_des_1[f, u, p, s] + sum(Modu_uv[f, v, u] for v in range(topo_num)) - 2)
    # 4.2 确定SC的容量
    for u in range(topo_num):
        for p in range(P2MP_num):
            for s in range(max_sc):
                if len(new_P2MP_SC_1[u][p][s][0]) != 0:
                    model.addConstr(SC_cap[u, p, s] == new_P2MP_SC_1[u][p][s][3][0])
                else:
                    model.addConstr(SC_cap[u, p, s] == 25 - 12.5 * SC_slow[u, p, s])
    # 4.3 满足容量限制，包括scr和des
    for u in range(topo_num):
        for p in range(P2MP_num):
            for s in range(max_sc):
                model.addConstr(gp.quicksum(SC_scr_2[f, u, p, s] for f in range(flow_num)) * 5 +
                                gp.quicksum(SC_des_2[f, u, p, s] for f in range(flow_num)) * 5
                                <= SC_cap[u, p, s])
                # 4.4 不同于选择FlexE Group和P2MP，SC的选择应该是互斥的
                model.addConstr(gp.quicksum(SC_scr_1[f, u, p, s] for f in range(flow_num)) <= M * z2[u, p, s])
                model.addConstr(gp.quicksum(SC_des_1[f, u, p, s] for f in range(flow_num)) <= M * (1 - z2[u, p, s]))

    # 5 选择该SC的流的路径是一样的
    # 5.1 判断两条流是否路径一致
    for i in range(topo_num):
        for u in range(topo_num):
            for v in range(topo_num):
                for f_1 in range(flow_num):
                    for f_2 in range(flow_num):
                        model.addConstr(equal_PHY[f_1, f_2, i, u, v] >= PHY_uv[f_1, i, u, v] - PHY_uv[f_2, i, u, v])
                        model.addConstr(equal_PHY[f_1, f_2, i, u, v] >= PHY_uv[f_2, i, u, v] - PHY_uv[f_1, i, u, v])
    # 5.2 如果此SC已经分配了流，那么路径既是已经分配的流的目的地，否则满足路径一致原则
    for u in range(topo_num):
        for p in range(P2MP_num):
            for s in range(max_sc):
                if len(new_P2MP_SC_1[u][p][s][0]) != 0:
                    for v_1 in range(topo_num):
                        for v_2 in range(topo_num):
                            for f in range(flow_num):
                                model.addConstr(PHY_uv[f, u, v_1, v_2] - SC_path[u][p][s][v_1][v_2]
                                                <= 1 - SC_scr_1[f, u, p, s])
                                model.addConstr(SC_path[u][p][s][v_1][v_2] - PHY_uv[f, u, v_1, v_2]
                                                <= 1 - SC_scr_1[f, u, p, s])
                                for i in range(topo_num):
                                    # 差值 ∈{-1,0,1}，右边取 2 - des_u - SC_des_1，只有在二者都为 1 时才变成 0
                                    model.addConstr(
                                        PHY_uv[f, i, v_1, v_2] - SC_path[u][p][s][v_1][v_2]
                                        <= 2 - des_u[f, i, u] - SC_des_1[f, u, p, s]
                                    )
                                    model.addConstr(
                                        SC_path[u][p][s][v_1][v_2] - PHY_uv[f, i, v_1, v_2]
                                        <= 2 - des_u[f, i, u] - SC_des_1[f, u, p, s]
                                    )
                else:
                    for f_1 in range(flow_num):
                        for f_2 in range(flow_num):
                            model.addConstr(gp.quicksum(equal_PHY[f_1, f_2, u, v_1, v_2]
                                                        for v_1 in range(topo_num) for v_2 in range(topo_num))
                                            <= (2 - SC_scr_1[f_1, u, p, s] - SC_scr_1[f_2, u, p, s]) * M)
                            for i in range(topo_num):
                                model.addConstr(
                                    gp.quicksum(equal_PHY[f_1, f_2, i, v_1, v_2]
                                        for v_1 in range(topo_num) for v_2 in range(topo_num))
                                    <= (4 - des_u[f_1, i, u] - des_u[f_2, i, u]
                                        - SC_des_1[f_1, u, p, s] - SC_des_1[f_2, u, p, s]) * M)

    # 6 保证选择连续的SC
    for f in range(flow_num):
        for u in range(topo_num):
            for p in range(P2MP_num):
                # 3) 选中的1的个数必须恰好等于区间长度 ⇒ 区间内全是 1，区间外全是 0
                model.addConstr(gp.quicksum(SC_scr_1[f, u, p, s] for s in range(max_sc))
                                == SC_scr_end[f, u, p] - SC_scr_start[f, u, p] + FlexE_scr[f, u, p])
                model.addConstr(gp.quicksum(SC_des_1[f, u, p, s] for s in range(max_sc))
                                 == SC_des_end[f, u, p] - SC_des_start[f, u, p] + FlexE_des[f, u, p])
                for s in range(max_sc):
                    model.addConstr(SC_scr_1[f, u, p, s] <= (s - SC_scr_start[f, u, p]) / M + 1)  # 左侧裁剪)
                    model.addConstr(SC_scr_1[f, u, p, s] <= (SC_scr_end[f, u, p] - s) / M + 1)  # 右侧裁剪)
                    model.addConstr(SC_des_1[f, u, p, s] <= (s - SC_des_start[f, u, p]) / M + 1) # 左侧裁剪)
                    model.addConstr(SC_des_1[f, u, p, s] <= (SC_des_end[f, u, p] - s) / M + 1) # 右侧裁剪)

    # 7 保证使用的sc_end满足不同类型的P2MP的最大限额
    for f in range(flow_num):
        for u in range(topo_num):
            for p in range(P2MP_num):
                model.addConstr(SC_scr_end[f, u, p] <= type_P2MP[new_node_P2MP[u][p][3] - 1][1] - 1)
                model.addConstr(SC_des_end[f, u, p] <= type_P2MP[new_node_P2MP[u][p][3] - 1][1] - 1)

    # 确定SC所在的FS
    for f in range(flow_num):
        for u in range(topo_num):
            for p in range(P2MP_num):
                p_type = new_node_P2MP[u][p][3]
                sc_fs_list = SC_FS_list.sc_fs_list(p_type)
                for w in range(max_sc_fs):
                    model.addConstr(FS_scr_1[f, u, p, w] <= gp.quicksum(
                        sc_fs_list[s][w] * SC_scr_1[f, u, p, s] for s in range(max_sc)))
                    model.addConstr(FS_des_1[f, u, p, w] <= gp.quicksum(
                        sc_fs_list[s][w] * SC_des_1[f, u, p, s] for s in range(max_sc)))
                    for s in range(max_sc):
                        if sc_fs_list[s][w] == 1:
                            model.addConstr(FS_scr_1[f, u, p, w] >= SC_scr_1[f, u, p, s])
                            model.addConstr(FS_des_1[f, u, p, w] >= SC_des_1[f, u, p, s])

    # 更新FS的起始和终止序号
    for f in range(flow_num):
        for u in range(topo_num):
            for p in range(P2MP_num):
                # 3) 选中的1的个数必须恰好等于区间长度 ⇒ 区间内全是 1，区间外全是 0
                model.addConstr(gp.quicksum(FS_scr_1[f, u, p, w] for w in range(max_sc_fs))
                                == FS_scr_end[f, u, p] - FS_scr_start[f, u, p] + FlexE_scr[f, u, p])
                model.addConstr(gp.quicksum(FS_des_1[f, u, p, w] for w in range(max_sc_fs))
                                 == FS_des_end[f, u, p] - FS_des_start[f, u, p] + FlexE_des[f, u, p])
                for w in range(max_sc_fs):
                    model.addConstr(FS_scr_1[f, u, p, w] <= (w - FS_scr_start[f, u, p]) / M + 1)  # 左侧裁剪)
                    model.addConstr(FS_scr_1[f, u, p, w] <= (FS_scr_end[f, u, p] - w) / M + 1)  # 右侧裁剪)
                    model.addConstr(FS_des_1[f, u, p, w] <= (w - FS_des_start[f, u, p]) / M + 1) # 左侧裁剪)
                    model.addConstr(FS_des_1[f, u, p, w] <= (FS_des_end[f, u, p] - w) / M + 1) # 右侧裁剪)

    # 确保共享同一FS的流的路径应该是一致的
    # 和SC的规则相同，共享同一个FS的流的目的地应该是一致的。
    for u in range(topo_num):
        for p in range(P2MP_num):
            for w in range(max_sc_fs):
                if np.sum(FS_path[u, p, w]) != 0:
                    for v_1 in range(topo_num):
                        for v_2 in range(topo_num):
                            for f in range(flow_num):
                                model.addConstr(PHY_uv[f, u, v_1, v_2] - FS_path[u][p][w][v_1][v_2]
                                                <= 1 - FS_scr_1[f, u, p, w])
                                model.addConstr(FS_path[u][p][w][v_1][v_2] - PHY_uv[f, u, v_1, v_2]
                                                <= 1 - FS_scr_1[f, u, p, w])
                                for i in range(topo_num):
                                    # 差值 ∈{-1,0,1}，右边取 2 - des_u - SC_des_1，只有在二者都为 1 时才变成 0
                                    model.addConstr(
                                        PHY_uv[f, i, v_1, v_2] - FS_path[u][p][w][v_1][v_2]
                                        <= 2 - des_u[f, i, u] - FS_des_1[f, u, p, w]
                                    )
                                    model.addConstr(
                                        FS_path[u][p][w][v_1][v_2] - PHY_uv[f, i, v_1, v_2]
                                        <= 2 - des_u[f, i, u] - FS_des_1[f, u, p, w]
                                    )
                else:
                    for f_1 in range(flow_num):
                        for f_2 in range(flow_num):
                            model.addConstr(gp.quicksum(equal_PHY[f_1, f_2, u, v_1, v_2]
                                                        for v_1 in range(topo_num) for v_2 in range(topo_num))
                                            <= (2 - FS_scr_1[f_1, u, p, w] - FS_scr_1[f_2, u, p, w]) * M)
                            for i in range(topo_num):
                                model.addConstr(
                                    gp.quicksum(equal_PHY[f_1, f_2, i, v_1, v_2]
                                                for v_1 in range(topo_num) for v_2 in range(topo_num))
                                    <= (4 - des_u[f_1, i, u] - des_u[f_2, i, u]
                                        - FS_des_1[f_1, u, p, w] - FS_des_1[f_2, u, p, w]) * M)

    # 保证频谱的一致性、连续性和不可重叠性
    # 1. 更新P2MP的起始频段
    for u in range(topo_num):
        for p in range(P2MP_num):
            if new_node_P2MP[u][p][5] != -1:
                model.addConstr(FS_P2MP_start[u, p] == new_node_P2MP[u][p][5])

    # 2. 更新流使用的全局频谱
    for f in range(flow_num):
        for u in range(topo_num):
            for v_1 in range(topo_num):
                for v_2 in range(topo_num):
                    model.addConstr(FS_uv_start[f, u, v_1, v_2] <= PHY_uv[f, u, v_1, v_2] * M)
                    model.addConstr(PHY_uv[f, u, v_1, v_2]  <= FS_uv_start[f, u, v_1, v_2])
                    model.addConstr(FS_uv_end[f, u, v_1, v_2] <= PHY_uv[f, u, v_1, v_2] * M)
                    model.addConstr(PHY_uv[f, u, v_1, v_2] <= FS_uv_end[f, u, v_1, v_2])
                    for p in range(P2MP_num):
                        model.addConstr(FS_uv_start[f, u, v_1, v_2] >= FS_P2MP_start[u, p] + FS_scr_start[f, u, p] + 1
                                        - M * (2 - PHY_uv[f, u, v_1, v_2] - FlexE_scr[f, u, p]))
                        model.addConstr(FS_uv_start[f, u, v_1, v_2] <= FS_P2MP_start[u, p] + FS_scr_start[f, u, p] + 1
                                        + M * (2 - PHY_uv[f, u, v_1, v_2] - FlexE_scr[f, u, p]))
                        model.addConstr(FS_uv_end[f, u, v_1, v_2] >= FS_P2MP_start[u, p] + FS_scr_end[f, u, p] + 1
                                        - M * (2 - PHY_uv[f, u, v_1, v_2] - FlexE_scr[f, u, p]))
                        model.addConstr(FS_uv_end[f, u, v_1, v_2] <= FS_P2MP_start[u, p] + FS_scr_end[f, u, p] + 1
                                        + M * (2 - PHY_uv[f, u, v_1, v_2] - FlexE_scr[f, u, p]))
                        for i in range(topo_num):
                            model.addConstr(FS_uv_start[f, i, v_1, v_2] >= FS_P2MP_start[u, p] + FS_des_start[f, u, p] + 1
                                            - M * (3 - PHY_uv[f, i, v_1, v_2] - FlexE_des[f, u, p] - des_u[f, i, u]))
                            model.addConstr(FS_uv_start[f, i, v_1, v_2] <= FS_P2MP_start[u, p] + FS_des_start[f, u, p] + 1
                                            + M * (3 - PHY_uv[f, i, v_1, v_2] - FlexE_des[f, u, p] - des_u[f, i, u]))
                            model.addConstr(FS_uv_end[f, i, v_1, v_2] >= FS_P2MP_start[u, p] + FS_des_end[f, u, p] + 1
                                            - M * (3 - PHY_uv[f, i, v_1, v_2] - FlexE_des[f, u, p] - des_u[f, i, u]))
                            model.addConstr(FS_uv_end[f, i, v_1, v_2] <= FS_P2MP_start[u, p] + FS_des_end[f, u, p] + 1
                                            + M * (3 - PHY_uv[f, i, v_1, v_2] - FlexE_des[f, u, p] - des_u[f, i, u]))


    # 3. 以0-1变量来展示使用的频谱
    for f in range(flow_num):
        for u in range(topo_num):
            for v_1 in range(topo_num):
                for v_2 in range(topo_num):
                    model.addConstr(gp.quicksum(FS_uv_w[f, u, v_1, v_2, w] for w in range(max_fs)) ==
                                    FS_uv_end[f, u, v_1, v_2] - FS_uv_start[f, u, v_1, v_2] + PHY_uv[f, u, v_1, v_2])
                    for w in range(max_fs):
                        model.addConstr(FS_uv_w[f, u, v_1, v_2, w] <= (w - FS_uv_start[f, u, v_1, v_2]) / M + 1)  # 左侧裁剪)
                        model.addConstr(FS_uv_w[f, u, v_1, v_2, w] <= (FS_uv_end[f, u, v_1, v_2] - w) / M + 1)  # 右侧裁剪)

    # 4. 保证不重叠性
    for u in range(topo_num):
        for p in range(P2MP_num):
            for v_1 in range(topo_num):
                for v_2 in range(topo_num):
                    for w in range(max_fs):
                        model.addConstr(FS_edge_scr[v_1, v_2, w, u, p] >= pre_FS_scr[v_1, v_2, w, u, p])
                        model.addConstr(FS_edge_des[v_1, v_2, w, u, p] >= pre_FS_des[v_1, v_2, w, u, p])
                        for f in range(flow_num):
                            model.addConstr(FS_edge_scr[v_1, v_2, w, u, p] >=
                                            FS_uv_w[f, u, v_1, v_2, w] + FlexE_scr[f, u, p] - 1)
                            for i in range(topo_num):
                                model.addConstr(FS_edge_des[v_1, v_2, w, u, p] >=
                                                FS_uv_w[f, i, v_1, v_2, w] + FlexE_des[f, u, p] + des_u[f, i, u] - 2)


    for v_1 in range(topo_num):
        for v_2 in range(topo_num):
            for w in range(max_fs):
                model.addConstr(gp.quicksum(FS_edge_scr[v_1, v_2, w, u, p] for u in range(topo_num) for p in range(P2MP_num))
                                <= 1)
                model.addConstr(gp.quicksum(FS_edge_des[v_1, v_2, w, u, p] for u in range(topo_num) for p in range(P2MP_num))
                                <= 1)







    # 求解模型
    model.optimize()

    # 获取并输出结果
    if model.status == GRB.OPTIMAL:

        # array_p_uv = np.zeros((flow_num, topo_num, topo_num))
        # for (f, u, v), var in p_uv.items():
        #     array_p_uv[f, u, v] = var.X  # var.X 是变量的解值
        #


        array_FlexE_scr = np.zeros((flow_num, topo_num, P2MP_num))
        for (f, u, p), var in FlexE_scr.items():
            array_FlexE_scr[f, u, p] = var.X

        array_FlexE_des = np.zeros((flow_num, topo_num, P2MP_num))
        for (f, u, p), var in FlexE_des.items():
            array_FlexE_des[f, u, p] = var.X

        array_SC_scr_1 = np.zeros((flow_num, topo_num, P2MP_num, max_sc))
        for (f, u, p, s), var in SC_scr_1.items():
            array_SC_scr_1[f, u, p, s] = var.X

        array_SC_des_1 = np.zeros((flow_num, topo_num, P2MP_num, max_sc))
        for (f, u, p, s), var in SC_des_1.items():
            array_SC_des_1[f, u, p, s] = var.X

        array_SC_scr_2 = np.zeros((flow_num, topo_num, P2MP_num, max_sc))
        for (f, u, p, s), var in SC_scr_2.items():
            array_SC_scr_2[f, u, p, s] = var.X

        array_SC_des_2 = np.zeros((flow_num, topo_num, P2MP_num, max_sc))
        for (f, u, p, s), var in SC_des_2.items():
            array_SC_des_2[f, u, p, s] = var.X

        array_SC_slow = np.zeros((topo_num, P2MP_num, max_sc))
        for (u, p, s), var in SC_slow.items():
            array_SC_slow[u, p, s] = var.X

        array_SC_cap = np.zeros((topo_num, P2MP_num, max_sc))
        for (u, p, s), var in SC_cap.items():
            array_SC_cap[u, p, s] = var.X

        array_equal_PHY = np.zeros((flow_num, flow_num, topo_num, topo_num, topo_num))
        for (f, f, u, v_1, v_2), var in equal_PHY.items():
            array_equal_PHY[f, f, u, v_1, v_2] = var.X

        array_SC_scr_start = np.zeros((flow_num, topo_num, P2MP_num))
        for (f, u, p), var in SC_scr_start.items():
            array_SC_scr_start[f, u, p] = var.X

        array_SC_scr_end = np.zeros((flow_num, topo_num, P2MP_num))
        for (f, u, p), var in SC_scr_end.items():
            array_SC_scr_end[f, u, p] = var.X

        array_SC_des_start = np.zeros((flow_num, topo_num, P2MP_num))
        for (f, u, p), var in SC_des_start.items():
            array_SC_des_start[f, u, p] = var.X

        array_SC_des_end = np.zeros((flow_num, topo_num, P2MP_num))
        for (f, u, p), var in SC_des_end.items():
            array_SC_des_end[f, u, p] = var.X

        array_FS_scr_1 = np.zeros((flow_num, topo_num, P2MP_num, max_sc_fs))
        for (f, u, p, w), var in FS_scr_1.items():
            array_FS_scr_1[f, u, p, w] = var.X

        array_FS_des_1 = np.zeros((flow_num, topo_num, P2MP_num, max_sc_fs))
        for (f, u, p, w), var in FS_des_1.items():
            array_FS_des_1[f, u, p, w] = var.X

        array_FS_scr_start = np.zeros((flow_num, topo_num, P2MP_num))
        for (f, u, p), var in FS_scr_start.items():
            array_FS_scr_start[f, u, p] = var.X

        array_FS_scr_end = np.zeros((flow_num, topo_num, P2MP_num))
        for (f, u, p), var in FS_scr_end.items():
            array_FS_scr_end[f, u, p] = var.X

        array_FS_des_start = np.zeros((flow_num, topo_num, P2MP_num))
        for (f, u, p), var in FS_des_start.items():
            array_FS_des_start[f, u, p] = var.X

        array_FS_des_end = np.zeros((flow_num, topo_num, P2MP_num))
        for (f, u, p), var in FS_des_end.items():
            array_FS_des_end[f, u, p] = var.X

        array_FS_P2MP_start = np.zeros((topo_num, P2MP_num))
        for (u, p), var in FS_P2MP_start.items():
            array_FS_P2MP_start[u, p] = var.X

        array_FS_uv_start = np.zeros((flow_num, topo_num, topo_num, topo_num))
        for (f, i, u, v), var in FS_uv_start.items():
            array_FS_uv_start[f, i, u, v] = var.X

        array_FS_uv_end = np.zeros((flow_num, topo_num, topo_num, topo_num))
        for (f, i, u, v), var in FS_uv_end.items():
            array_FS_uv_end[f, i, u, v] = var.X

        array_FS_uv_w = np.zeros((flow_num, topo_num, topo_num, topo_num, max_fs))
        for (f, u, v_1, v_2, w), var in FS_uv_w.items():
            array_FS_uv_w[f, u, v_1, v_2, w] = var.X

        array_FS_edge_scr = np.zeros((topo_num, topo_num, max_fs, topo_num, P2MP_num))
        for (v_1, v_2, w, u, p), var in FS_edge_scr.items():
            array_FS_edge_scr[v_1, v_2, w, u, p] = var.X

        array_FS_edge_des = np.zeros((topo_num, topo_num, max_fs, topo_num, P2MP_num))
        for (v_1, v_2, w, u, p), var in FS_edge_des.items():
            array_FS_edge_des[v_1, v_2, w, u, p] = var.X

        print(f"Optimal value of the objective: {model.objVal}")  # 输出目标函数的最优值
    else:
        print("Optimal solution not found")  # 如果未找到最优解，输出信息



