#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : ILP.py 
# @File    : ILP_RMP.py
# @Author  : Wumh
# @Time    : 2026/1/3 17:32

import gurobipy as gp  # 导入Gurobi的Python接口，并缩写为gp
import numpy as np
import SC_FS_list
import topology as tp
from gurobipy import GRB  # 导入Gurobi中的常量

def Restoration_ILP(affected_flow, new_flow_acc, new_node_flow, new_node_P2MP, break_node, Tbox_num, Tbox_P2MP,
                    new_P2MP_SC_1, new_link_FS, node_flow_DP, node_P2MP_DP, flow_acc_DP, link_FS_DP, P2MP_SC_DP, flow_path_DP):

    # 参数
    flow_num = len(affected_flow)
    M = 10000
    P2MP_num = Tbox_num * Tbox_P2MP
    max_sc = 16
    max_sc_fs = 6
    max_fs = 20
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
    model = gp.Model("Restoration")

    # 添加变量
    f_i = model.addVars(flow_num, lb=0, ub=1, vtype=GRB.BINARY)
    des_u= model.addVars(flow_num, topo_num, topo_num, lb=0, ub=1, vtype=GRB.BINARY)
    PHY_uv = model.addVars(flow_num, topo_num, topo_num, topo_num, lb=0, ub=1, vtype=GRB.BINARY)
    Reconf_f = model.addVars(flow_num, vtype=GRB.BINARY)
    FlexE_scr = model.addVars(flow_num, topo_num, P2MP_num, lb=0, ub=1, vtype=GRB.BINARY)
    FlexE_des = model.addVars(flow_num, topo_num, P2MP_num, lb=0, ub=1, vtype=GRB.BINARY)
    SC_scr_1 = model.addVars(flow_num, topo_num, P2MP_num, max_sc, lb=0, ub=1, vtype=GRB.BINARY) #是否使用
    SC_des_1 = model.addVars(flow_num, topo_num, P2MP_num, max_sc, lb=0, ub=1, vtype=GRB.BINARY)
    SC_scr_2 = model.addVars(flow_num, topo_num, P2MP_num, max_sc, lb=0, vtype=GRB.INTEGER) # 使用的大小
    SC_des_2 = model.addVars(flow_num, topo_num, P2MP_num, max_sc, lb=0, vtype=GRB.INTEGER)
    z2 = model.addVars(topo_num, P2MP_num, max_sc, lb=0, ub=1, vtype=GRB.BINARY) # 辅助变量，指示该SC只可当作发送/接收SC
    SC_cap = model.addVars(topo_num, P2MP_num, max_sc, lb=0, vtype=GRB.CONTINUOUS)
    equal_PHY = model.addVars(flow_num, flow_num, topo_num, topo_num, topo_num, lb=0, ub=1, vtype=GRB.BINARY)
    FS_scr_1 = model.addVars(flow_num, topo_num, P2MP_num, max_sc_fs, lb=0, ub=1, vtype=GRB.BINARY)  # 是否使用
    FS_des_1 = model.addVars(flow_num, topo_num, P2MP_num, max_sc_fs, lb=0, ub=1, vtype=GRB.BINARY)  # 是否使用
    FS_edge_scr = model.addVars(topo_num, topo_num, max_fs, topo_num, P2MP_num, vtype=GRB.BINARY) # 某条链路上的某个fs被源节点上某个P2MP使用
    FS_edge_des = model.addVars(topo_num, topo_num, max_fs, topo_num, P2MP_num, vtype=GRB.BINARY) # 某条链路上的某个fs被目的节点上某个P2MP使用



    # 设置目标函数
    #model.setObjective(gp.quicksum(f_i[f] for f in range(flow_num)) + gp.quicksum(Reconf_f[f] for f in range(flow_num)), GRB.MINIMIZE)  # 设置目标函数，最大化
    model.setObjective(- gp.quicksum(f_i[f] for f in range(flow_num)) + gp.quicksum(Reconf_f[f] for f in range(flow_num)), GRB.MINIMIZE)


    for u in range(topo_num):
        for p in range(P2MP_num):
            # 2. 保证在该scr或者des上的FlexE Group上分配的流小于FlexE Group的容量
            model.addConstr(gp.quicksum((FlexE_scr[f, u, p] * affected_flow[f][3]) for f in range(flow_num)) +
                            gp.quicksum((FlexE_des[f, u, p] * affected_flow[f][3]) for f in range(flow_num)) <=
                            new_node_P2MP[u][p][4])


    # 4. 保证使用该SC的流不超过该SC的容量
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
                    continue
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

    # 确保共享同一FS的流的路径应该是一致的
    # 和SC的规则相同，共享同一个FS的流的目的地应该是一致的。
    for u in range(topo_num):
        for p in range(P2MP_num):
            for w in range(max_sc_fs):
                if np.sum(FS_path[u, p, w]) != 0:
                    continue
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
        array_f_i = np.zeros(flow_num)
        for f, var in f_i.items():
            array_f_i[f] = var.X

        array_des_u = np.zeros((flow_num, topo_num, topo_num))
        for (f, i, u), var in des_u.items():
            array_des_u[f, i, u] = var.X

        array_PHY_uv = np.zeros((flow_num, topo_num, topo_num, topo_num))
        for (f, i, u, v), var in PHY_uv.items():
            array_PHY_uv[f, i, u, v] = var.X

        array_Reconf_f = np.zeros((flow_num))
        for (f), var in Reconf_f.items():
            array_Reconf_f[f] = var.X

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

        array_SC_cap = np.zeros((topo_num, P2MP_num, max_sc))
        for (u, p, s), var in SC_cap.items():
            array_SC_cap[u, p, s] = var.X

        array_equal_PHY = np.zeros((flow_num, flow_num, topo_num, topo_num, topo_num))
        for (f, f, u, v_1, v_2), var in equal_PHY.items():
            array_equal_PHY[f, f, u, v_1, v_2] = var.X

        array_FS_scr_1 = np.zeros((flow_num, topo_num, P2MP_num, max_sc_fs))
        for (f, u, p, w), var in FS_scr_1.items():
            array_FS_scr_1[f, u, p, w] = var.X

        array_FS_des_1 = np.zeros((flow_num, topo_num, P2MP_num, max_sc_fs))
        for (f, u, p, w), var in FS_des_1.items():
            array_FS_des_1[f, u, p, w] = var.X

        array_FS_edge_scr = np.zeros((topo_num, topo_num, max_fs, topo_num, P2MP_num))
        for (v_1, v_2, w, u, p), var in FS_edge_scr.items():
            array_FS_edge_scr[v_1, v_2, w, u, p] = var.X

        array_FS_edge_des = np.zeros((topo_num, topo_num, max_fs, topo_num, P2MP_num))
        for (v_1, v_2, w, u, p), var in FS_edge_des.items():
            array_FS_edge_des[v_1, v_2, w, u, p] = var.X

        print(f"Optimal value of the objective: {model.objVal}")  # 输出目标函数的最优值
    else:
        print("Optimal solution not found")  # 如果未找到最优解，输出信息

# array_sameUP = np.zeros((flow_num, flow_num, topo_num))
        # for (f_1, f_2, u), var in sameUP.items():
        #     array_sameUP[f_1, f_2, u] = var.X
        #
        # array_sameUP_z = np.zeros((flow_num, flow_num, topo_num, P2MP_num))
        # for (f_1, f_2, u, p), var in sameUP_z.items():
        #     array_sameUP_z[f_1, f_2, u, p] = var.X
        #
        # array_orderFS = np.zeros((flow_num, topo_num, flow_num, topo_num, topo_num, topo_num))
        # for (f_1, u_1, f_2, u_2, v_1, v_2), var in orderFS.items():
        #     array_orderFS[f_1, u_1, f_2, u_2, v_1, v_2] = var.X


