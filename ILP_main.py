#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @IDE     : PyCharm 
# @Project : code
# @File    : ILP.py
# @Author  : Wumh
# @Time    : 2025/2/15 19:16
import gurobipy as gp  # 导入Gurobi的Python接口，并缩写为gp
import numpy as np
import SC_FS_list
import topology as tp
from gurobipy import GRB  # 导入Gurobi中的常量
from ILP_sub1 import Sub_problem

def Restoration_ILP(affected_flow, new_flow_acc, new_node_flow, new_node_P2MP, break_node, Tbox_num, Tbox_P2MP,
                    new_P2MP_SC_1, new_link_FS, node_flow_DP, node_P2MP_DP, flow_acc_DP, link_FS_DP, P2MP_SC_DP, flow_path_DP):

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
    model = gp.Model("Restoration")

    # 添加变量
    f_i = model.addVars(flow_num, lb=0, ub=1, vtype=GRB.BINARY)
    OEO_uv = model.addVars(flow_num, topo_num, topo_num, lb=0, ub=1, vtype=GRB.BINARY)
    scr_u = model.addVars(flow_num, topo_num, topo_num, lb=0, ub=1, vtype=GRB.BINARY)
    des_u= model.addVars(flow_num, topo_num, topo_num, lb=0, ub=1, vtype=GRB.BINARY)
    PHY_uv = model.addVars(flow_num, topo_num, topo_num, topo_num, lb=0, ub=1, vtype=GRB.BINARY)
    ele_len_uv = model.addVars(flow_num, topo_num, topo_num, topo_num, lb=0, vtype=GRB.INTEGER)
    temp_len_uv = model.addVars(flow_num, topo_num, lb=0, vtype=GRB.INTEGER)
    len_uv = model.addVars(flow_num, topo_num, topo_num, lb=0, vtype=GRB.INTEGER)
    # 已有：Modu_uv  ∈ {0,1,2}
    Modu_uv = model.addVars(flow_num, topo_num, topo_num, lb=0, ub=2, vtype=GRB.INTEGER, name="Modu_uv")
    Des_modu = model.addVars(flow_num, topo_num, lb=0, ub=2, vtype=GRB.INTEGER, name="des_modu")
    # 新增：Y_long = 1  ⇔ len_uv > 500
    Y_long = model.addVars(flow_num, topo_num, topo_num, vtype=GRB.BINARY, name="Y_long")
    SC_uv = model.addVars(flow_num, topo_num, lb=0, vtype=GRB.INTEGER)
    z1_scr = model.addVars(flow_num, vtype=GRB.BINARY)
    z1_des = model.addVars(flow_num, vtype=GRB.BINARY) # 辅助变量，判断x和y的大小
    Reconf_f = model.addVars(flow_num, vtype=GRB.BINARY)
    Reconf_scr_u = model.addVars(flow_num, lb=0, ub=1, vtype=GRB.BINARY)
    Reconf_des_u = model.addVars(flow_num, lb=0, ub=1, vtype=GRB.BINARY)




    #SC_FS_f = model.addVars(flow_num, topo_num, P2MP_num, max_fs, lb=0, ub=1, vtype=GRB.BINARY)

    # 设置目标函数
    #model.setObjective(gp.quicksum(f_i[f] for f in range(flow_num)) + gp.quicksum(Reconf_f[f] for f in range(flow_num)), GRB.MINIMIZE)  # 设置目标函数，最大化
    model.setObjective(- gp.quicksum(f_i[f] for f in range(flow_num)) + gp.quicksum(Reconf_f[f] for f in range(flow_num)), GRB.MINIMIZE)

    # 添加约束条件
    # 生成逻辑拓扑，流守恒约束
    for index, f in enumerate(affected_flow):
        for u in range(topo_num):
            model.addConstr(gp.quicksum(OEO_uv[index, u, v] for v in range(topo_num)) <= f_i[index])
            model.addConstr(gp.quicksum(OEO_uv[index, v, u] for v in range(topo_num)) <= f_i[index])
            if f[1] == u:
                model.addConstr(gp.quicksum(OEO_uv[index, u, v] for v in range(topo_num)) -
                                gp.quicksum(OEO_uv[index, v, u] for v in range(topo_num)) == f_i[index])
            elif f[2] == u:
                model.addConstr(gp.quicksum(OEO_uv[index, u, v] for v in range(topo_num)) -
                                gp.quicksum(OEO_uv[index, v, u] for v in range(topo_num)) == -f_i[index])
            else:
                model.addConstr(gp.quicksum(OEO_uv[index, u, v] for v in range(topo_num)) -
                                gp.quicksum(OEO_uv[index, v, u] for v in range(topo_num)) == 0)
                #model.addConstr(gp.quicksum(OEO_uv[index, u, v] for v in range(topo_num)) <= gp.quicksum(p_uv[index, u, v] for v in range(topo_num)))
            # 不能选择中断节点作为中转节点
            for v in range(topo_num):
                if u == break_node or v == break_node:
                    model.addConstr(OEO_uv[index, u, v] <= 0)


    # 确定源目的节点：因为上边的是逻辑链路，而真实的物理路径还需要每一段的逻辑链路的源目的节点来确定，所以先来确定这些
    for f in range(flow_num):
        for i in range(topo_num):
            model.addConstr(scr_u[f, i, i] == gp.quicksum(OEO_uv[f, i, v] for v in range(topo_num)))
            model.addConstr(gp.quicksum(scr_u[f, i, v] for v in range(topo_num)) == gp.quicksum(OEO_uv[f, i, v] for v in range(topo_num)))
    model.addConstrs(des_u[f, i, v] == OEO_uv[f, i, v] for f in range(flow_num) for i in range(topo_num) for v in range(topo_num))

    # 生成物理链路
    for f in range(flow_num):
        for i in range(topo_num):
            model.addConstrs(PHY_uv[f, i, u, v] <= topo_matrix[u, v] for u in range(topo_num) for v in range(topo_num))
            for u in range(topo_num):
                model.addConstr(gp.quicksum(PHY_uv[f, i, u, v] for v in range(topo_num)) <= 1)
                model.addConstr(gp.quicksum(PHY_uv[f, i, v, u] for v in range(topo_num)) <= 1)
                model.addConstr(gp.quicksum(PHY_uv[f, i, u, v] for v in range(topo_num)) -
                                gp.quicksum(PHY_uv[f, i, v, u] for v in range(topo_num)) == scr_u[f, i, u] - des_u[f, i, u])

    # 计算路径距离
    model.addConstrs(ele_len_uv[f, i, u, v] == PHY_uv[f, i, u, v] * topo_dis[u, v] for f in range(flow_num) for i in range(topo_num)
                     for u in range(topo_num) for v in range(topo_num))
    for f in range(flow_num):
        for i in range(topo_num):
            model.addConstr(temp_len_uv[f, i] == gp.quicksum(ele_len_uv[f, i, u, v] for u in range(topo_num) for v in range(topo_num)))
    for f in range(flow_num):
        for u in range(topo_num):
            for v in range(topo_num):
                model.addConstr(des_u[f, u, v] <= len_uv[f, u, v])
                model.addConstr(len_uv[f, u, v] <= des_u[f, u, v] * M)
                model.addConstr(len_uv[f, u, v] <= temp_len_uv[f, u])
                model.addConstr(len_uv[f, u, v] >= temp_len_uv[f, u] - M * (1 - des_u[f, u, v]))

    # 分配调制格式，25-->1 12.5-->2
    for f in range(flow_num):
        for u in range(topo_num):
            for v in range(topo_num):
                # ---------- (A)  Modu=0 ⇒ len=0 ；Modu>0 ⇒ len>0 ----------
                model.addConstr(len_uv[f, u, v] <= M * Modu_uv[f, u, v])
                model.addConstr(len_uv[f, u, v] >= Modu_uv[f, u, v])

                # ---------- (B)  Y_long 表示 “len > 500” ----------
                model.addConstr(len_uv[f, u, v] <= 500 + M * Y_long[f, u, v])
                model.addConstr(len_uv[f, u, v] >= 501 * Y_long[f, u, v])

                # ---------- (C)  将 Y_long 与 Modu 整合成 0/1/2 ----------
                # Y_long = 0  ⇒  Modu ∈ {0,1}
                # Y_long = 1  ⇒  Modu = 2
                model.addConstr(Modu_uv[f, u, v] <= 1 + Y_long[f, u, v])
                model.addConstr(Modu_uv[f, u, v] >= 2 * Y_long[f, u, v])

    # 判断端资源配置是否可以不变，即使用第一种策略进行恢复（源目的节点所在的路径段的调制格式不变）：0：不变 1：重新分配/流未被恢复
    for f in range(flow_num):
        for u in range(topo_num):
            if u == affected_flow[f][1]:
                model.addConstr(gp.quicksum(pre_flow_modu[f, u, v] for v in range(topo_num))
                                - gp.quicksum(Modu_uv[f, u, v] for v in range(topo_num)) <= 2 * Reconf_scr_u[f])
                model.addConstr(gp.quicksum(Modu_uv[f, u, v] for v in range(topo_num))
                                - gp.quicksum(pre_flow_modu[f, u, v] for v in range(topo_num)) <= 2 * Reconf_scr_u[f])
                model.addConstr(gp.quicksum(pre_flow_modu[f, u, v] for v in range(topo_num))
                                - gp.quicksum(Modu_uv[f, u, v] for v in range(topo_num)) >= 1 - M * (1 - z1_scr[f]) - M * (1 - Reconf_scr_u[f]))
                model.addConstr(gp.quicksum(Modu_uv[f, u, v] for v in range(topo_num))
                                - gp.quicksum(pre_flow_modu[f, u, v] for v in range(topo_num)) >= 1 - M * z1_scr[f] - M * (1 - Reconf_scr_u[f]))
            if u == affected_flow[f][2]:
                model.addConstr(gp.quicksum(pre_flow_modu[f, v, u] for v in range(topo_num))
                                - gp.quicksum(Modu_uv[f, v, u] for v in range(topo_num)) <= 2 * Reconf_des_u[f])
                model.addConstr(gp.quicksum(Modu_uv[f, v, u] for v in range(topo_num))
                                - gp.quicksum(pre_flow_modu[f, v, u] for v in range(topo_num)) <= 2 * Reconf_des_u[f])
                model.addConstr(gp.quicksum(pre_flow_modu[f, v, u] for v in range(topo_num))
                                - gp.quicksum(Modu_uv[f, v, u] for v in range(topo_num)) >= 1 - M * (1 - z1_des[f]) - M * (1 - Reconf_des_u[f]))
                model.addConstr(gp.quicksum(Modu_uv[f, v, u] for v in range(topo_num))
                                - gp.quicksum(pre_flow_modu[f, v, u] for v in range(topo_num)) >= 1 - M * z1_des[f] - M * (1 - Reconf_des_u[f]))
        model.addConstr(Reconf_f[f] >= Reconf_scr_u[f])
        model.addConstr(Reconf_f[f] >= Reconf_des_u[f])
        model.addConstr(Reconf_f[f] <= Reconf_scr_u[f] + Reconf_des_u[f])



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

        array_OEO_uv = np.zeros((flow_num, topo_num, topo_num))
        for (f, u, v), var in OEO_uv.items():
            array_OEO_uv[f, u, v] = var.X  # var.X 是变量的解值

        array_scr_u = np.zeros((flow_num, topo_num, topo_num))
        for (f, i, u), var in scr_u.items():
            array_scr_u[f, i, u] = var.X

        array_des_u = np.zeros((flow_num, topo_num, topo_num))
        for (f, i, u), var in des_u.items():
            array_des_u[f, i, u] = var.X

        array_PHY_uv = np.zeros((flow_num, topo_num, topo_num, topo_num))
        for (f, i, u, v), var in PHY_uv.items():
            array_PHY_uv[f, i, u, v] = var.X

        array_ele_len_uv = np.zeros((flow_num, topo_num, topo_num, topo_num))
        for (f, i, u, v), var in ele_len_uv.items():
            array_ele_len_uv[f, i, u, v] = var.X

        array_temp_len_uv = np.zeros((flow_num, topo_num))
        for (f, i), var in temp_len_uv.items():
            array_temp_len_uv[f, i] = var.X

        array_len_uv = np.zeros((flow_num, topo_num, topo_num))
        for (f, u, v), var in len_uv.items():
            array_len_uv[f, u, v] = var.X

        array_Modu_uv = np.zeros((flow_num, topo_num, topo_num))
        for (f, u, v), var in Modu_uv.items():
            array_Modu_uv[f, u, v] = var.X

        array_Modu_uv = np.zeros((flow_num, topo_num, topo_num))
        for (f, u, v), var in Modu_uv.items():
            array_Modu_uv[f, u, v] = var.X

        array_Reconf_scr_u = np.zeros((flow_num))
        for (f), var in Reconf_scr_u.items():
            array_Reconf_scr_u[f] = var.X

        array_Reconf_des_u = np.zeros((flow_num))
        for (f), var in Reconf_des_u.items():
            array_Reconf_des_u[f] = var.X

        array_Reconf_f = np.zeros((flow_num))
        for (f), var in Reconf_f.items():
            array_Reconf_f[f] = var.X

        array_z1_scr = np.zeros((flow_num))
        for (f), var in z1_scr.items():
            array_z1_scr[f] = var.X

        array_z1_des = np.zeros((flow_num))
        for (f), var in z1_des.items():
            array_z1_des[f] = var.X


        print(f"Optimal value of the objective: {model.objVal}")  # 输出目标函数的最优值
    else:
        print("Optimal solution not found")  # 如果未找到最优解，输出信息

    C = Sub_problem(affected_flow, new_flow_acc, new_node_flow, new_node_P2MP, break_node, Tbox_num, Tbox_P2MP,
                    new_P2MP_SC_1, new_link_FS, node_flow_DP, node_P2MP_DP, flow_acc_DP, link_FS_DP, P2MP_SC_DP, flow_path_DP,
                array_Reconf_f, array_scr_u, array_des_u, array_Modu_uv, array_PHY_uv)



