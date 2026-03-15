#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def Create_Empty_Column(U_num, P_num, C_num, W_num, E_num, F_num):
    """
    生成一个“空列模板”。
    你在 pricing problem 里生成一条新列之后，
    只需要把对应位置赋值即可，不用每次手动拼一大堆字典。

    这里每个字段的含义，都是按你 master problem 里的符号来的。

    参数说明
    ----------
    U_num : int
        u 的数量，例如不同节点 / group 所在节点数量
    P_num : int
        p 的数量，例如每个 u 下可选的 P2MP / group 编号上界
    C_num : int
        c 的数量，例如 SC 编号 / capacity level 编号
    W_num : int
        w 的数量，例如 FS 编号 / 小粒度频谱编号
    E_num : int
        e 的数量，这里建议你把 (i,j) 物理链路先映射成 link index = e
    F_num : int
        f 的数量，例如大频谱槽 / fiber slot 编号数量

    返回
    ----------
    col : dict
        一条列的数据结构
    """

    col = {}

    # ----------------------------
    # 目标函数里该列自己的附加代价
    # 对应论文里的 f_rf_clr 或类似含义
    # 如果某条列没有额外代价，就保持为 0
    # ----------------------------
    col["f_rf_clr"] = 0.0

    # ----------------------------
    # 下面开始是“列系数”
    # 这些系数都表示：如果选了这条列，那么它会占用哪些资源
    # 没占用的位置填 0，占用的位置填 1 或对应数值
    # ----------------------------

    # g_tx[u,p], g_rx[u,p]
    # 该列是否在 (u,p) 这个 group 上作为发送端 / 接收端占用
    col["g_tx"] = np.zeros((U_num, P_num), dtype=float)
    col["g_rx"] = np.zeros((U_num, P_num), dtype=float)

    # e_tx[u,p,c], e_rx[u,p,c]
    # 该列是否使用了 (u,p,c) 这个 SC，且方向为 tx / rx
    col["e_tx"] = np.zeros((U_num, P_num, C_num), dtype=float)
    col["e_rx"] = np.zeros((U_num, P_num, C_num), dtype=float)

    # n_tx[u,p,c], n_rx[u,p,c]
    # 该列在 (u,p,c) 上占用了多少个 5G 单位（按你原公式里的 5 * (...)）
    col["n_tx"] = np.zeros((U_num, P_num, C_num), dtype=float)
    col["n_rx"] = np.zeros((U_num, P_num, C_num), dtype=float)

    # a[u,p,c]
    # 该列给 (u,p,c) 提供的容量上界 / 可承载能力（按你的公式原样保留）
    col["a"] = np.zeros((U_num, P_num, C_num), dtype=float)

    # xi[u,p,c]
    # 该列是否激活了 (u,p,c)
    col["xi"] = np.zeros((U_num, P_num, C_num), dtype=float)

    # h[u,p,c,e]
    # 若该列激活了 (u,p,c)，则它在第 e 条 EO link 上的路径使用情况
    col["h"] = np.zeros((U_num, P_num, C_num, E_num), dtype=float)

    # sigma[u,p,w]
    # 该列是否激活了 (u,p,w) 这个 FS 单元
    col["sigma"] = np.zeros((U_num, P_num, W_num), dtype=float)

    # tau[u,p,w,e]
    # 若该列激活了 (u,p,w)，则它在第 e 条 EO link 上的路径使用情况
    col["tau"] = np.zeros((U_num, P_num, W_num, E_num), dtype=float)

    # varsigma_tx[e,f,u,p], varsigma_rx[e,f,u,p]
    # 该列在全局链路 e 的大频谱槽 / fiber slot = f 上，
    # 是否被 (u,p) 这个发送端 / 接收端占用
    col["varsigma_tx"] = np.zeros((E_num, F_num, U_num, P_num), dtype=float)
    col["varsigma_rx"] = np.zeros((E_num, F_num, U_num, P_num), dtype=float)

    return col


def Solve_Master_Problem_By_CG_Type(column_pool,
                                    b_r,
                                    beta,
                                    U_cap,
                                    H_up,
                                    K_upc,
                                    W_upw,
                                    U_num,
                                    P_num,
                                    C_num,
                                    W_num,
                                    E_num,
                                    F_num,
                                    big_M=10000,
                                    lp_relaxation=True,
                                    output_flag=0):
    """
    按列生成（Column Generation）里的 Restricted Master Problem 写法，
    建立并求解 master problem。

    这版写法特意尽量贴近你 ILP.py 的风格：
    - 函数式
    - 下标循环
    - 使用 numpy 数组作为输入类型
    - 用 Gurobi 直接建模
    - 中文详细注释

    ----------------------------------------------------------------------
    一、输入数据格式（非常重要）
    ----------------------------------------------------------------------

    1) column_pool
       类型：list
       结构：column_pool[r] = [col_0, col_1, ..., col_T]
       含义：第 r 条业务流当前已有的列池

       其中每一条 col 都是通过 Create_Empty_Column(...) 创建出来的 dict，
       然后你再往里面填资源占用系数。

       例如：
       column_pool = [
           [col_r0_t0, col_r0_t1],     # 第 0 条流有 2 条候选列
           [col_r1_t0],                # 第 1 条流有 1 条候选列
           [col_r2_t0, col_r2_t1, ...]
       ]

    2) b_r
       类型：np.array, shape = (flow_num,)
       含义：每条 flow 的业务需求大小 b_r

    3) beta
       类型：float
       含义：目标函数里的惩罚系数

    4) U_cap
       类型：np.array, shape = (U_num, P_num)
       含义：每个 (u,p) group 的容量上限

    5) H_up
       类型：np.array, shape = (U_num, P_num)
       含义：
           H_up[u,p] = 1 表示该 (u,p) 是存在的，可以参与建模
           H_up[u,p] = 0 表示该 (u,p) 不存在，直接跳过

    6) K_upc
       类型：np.array, shape = (U_num, P_num, C_num)
       含义：
           K_upc[u,p,c] = 1 表示该 (u,p,c) 存在
           K_upc[u,p,c] = 0 表示该组合不存在

    7) W_upw
       类型：np.array, shape = (U_num, P_num, W_num)
       含义：
           W_upw[u,p,w] = 1 表示该 (u,p,w) 存在
           W_upw[u,p,w] = 0 表示该组合不存在

    8) U_num, P_num, C_num, W_num, E_num, F_num
       都是各个维度的数量

    9) big_M
       大 M 常数

    10) lp_relaxation
        True  -> 解 LP 松弛版 RMP（列生成过程中一般用这个）
        False -> 解整数版 Master Problem（最后收尾时一般用这个）

    ----------------------------------------------------------------------
    二、返回结果
    ----------------------------------------------------------------------
    result : dict
        result["model"]              -> Gurobi 模型对象
        result["mu_value"]           -> 每个 flow 每条列的取值
        result["chi_value"]          -> chi 的取值
        result["obj"]                -> 目标值
        result["duals"]              -> 论文符号下整理后的对偶变量（仅 LP 时有效）
        result["selected_columns"]   -> 每个 flow 当前被选中的列
    """

    flow_num = len(column_pool)

    # ------------------------------------------------------------------
    # 1. 创建模型
    # ------------------------------------------------------------------
    model = gp.Model("Master_Problem_CG_Type")
    model.Params.OutputFlag = output_flag

    # ------------------------------------------------------------------
    # 2. 决定变量类型
    #    列生成过程中一般先解 LP 松弛，所以变量是连续的 [0,1]
    #    最终收尾时再改成 0-1 变量
    # ------------------------------------------------------------------
    if lp_relaxation:
        main_vtype = GRB.CONTINUOUS
    else:
        main_vtype = GRB.BINARY

    # ------------------------------------------------------------------
    # 3. 定义变量
    # ------------------------------------------------------------------

    # mu[r][t] : 第 r 条流是否/多大程度选择第 t 条列
    # 注意：因为每个 flow 的列数不一样，所以这里不适合直接 addVars(flow_num, T_num)
    #      而是用“双层字典”的方式更自然。
    mu = {}
    for r in range(flow_num):
        mu[r] = {}
        for t in range(len(column_pool[r])):
            mu[r][t] = model.addVar(lb=0, ub=1, vtype=main_vtype, name=f"mu_{r}_{t}")

    # chi[r] : 第 r 条流是否被成功恢复 / 被主问题接纳
    chi = model.addVars(flow_num, lb=0, ub=1, vtype=main_vtype, name="chi")

    # z[u,p,c] : 用来控制 SC 的方向互斥
    # 如果 z=1，则更倾向于 tx 使用；如果 z=0，则更倾向于 rx 使用
    z = model.addVars(U_num, P_num, C_num, lb=0, ub=1, vtype=main_vtype, name="z")

    # h_tilde[u,p,c,e] : 对同一个 (u,p,c)，不同列之间要共享一致的 SC 路径
    # 这里用一个“全局代表路径变量”来做一致性约束
    h_tilde = model.addVars(U_num, P_num, C_num, E_num, lb=0, ub=1, vtype=main_vtype, name="h_tilde")

    # tau_tilde[u,p,w,e] : 同理，FS 路径的一致性变量
    tau_tilde = model.addVars(U_num, P_num, W_num, E_num, lb=0, ub=1, vtype=main_vtype, name="tau_tilde")

    model.update()

    # ------------------------------------------------------------------
    # 4. 目标函数
    #    这里按你前面 master problem 的写法：
    #
    #    min  Σ_r (1 - chi_r) + beta * Σ_r Σ_t f_rf_clr(r,t) * mu_{r,t}
    #
    #    含义：
    #    - 第一项希望尽量让 chi_r = 1，也就是尽量恢复更多流
    #    - 第二项是在可恢复的情况下，尽量少用代价高的列
    # ------------------------------------------------------------------
    obj_1 = gp.quicksum(1 - chi[r] for r in range(flow_num))

    obj_2 = gp.quicksum(
        beta * column_pool[r][t]["f_rf_clr"] * mu[r][t]
        for r in range(flow_num)
        for t in range(len(column_pool[r]))
    )

    model.setObjective(obj_1 + obj_2, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # 5. 约束 1：每条 flow 的列选择和 chi 对应
    #
    #    Σ_t mu[r,t] = chi[r]
    #
    #    含义：
    #    - 如果 flow r 被恢复（chi=1），那它必须且只能选中一条列
    #    - 如果 flow r 不恢复（chi=0），那它一条列也不能选
    #
    #    如果你以后想允许“一条 flow 选多条列做 convex combination（LP 阶段）”，
    #    这个式子依然是标准写法，因为 LP 松弛时 mu 可以分数。
    # ------------------------------------------------------------------
    constr_assign = {}
    for r in range(flow_num):
        constr_assign[r] = model.addConstr(
            gp.quicksum(mu[r][t] for t in range(len(column_pool[r]))) == chi[r],
            name=f"assign_{r}"
        )

    # ------------------------------------------------------------------
    # 6. 约束 2：FlexE group 容量约束
    #
    #    Σ_r Σ_t b_r * (g_tx + g_rx) * mu[r,t] <= U_cap[u,p]
    #
    #    含义：
    #    所有被选中的列，在同一个 (u,p) group 上的总负载，不能超过 group 容量
    # ------------------------------------------------------------------
    constr_group_cap = {}
    for u in range(U_num):
        for p in range(P_num):
            if H_up[u, p] == 0:
                continue

            lhs = gp.quicksum(
                b_r[r] * (column_pool[r][t]["g_tx"][u, p] + column_pool[r][t]["g_rx"][u, p]) * mu[r][t]
                for r in range(flow_num)
                for t in range(len(column_pool[r]))
            )

            constr_group_cap[(u, p)] = model.addConstr(
                lhs <= U_cap[u, p],
                name=f"group_cap_{u}_{p}"
            )

    # ------------------------------------------------------------------
    # 7. 约束 3：SC 方向互斥
    #
    #    对同一个 (u,p,c)，不能同时既被大量 tx 使用又被 rx 使用
    #
    #    这里用 z[u,p,c] + Big-M 的写法实现
    #
    #    tx_sum <= M * z
    #    rx_sum <= M * (1 - z)
    # ------------------------------------------------------------------
    constr_sc_dir_tx = {}
    constr_sc_dir_rx = {}

    for u in range(U_num):
        for p in range(P_num):
            if H_up[u, p] == 0:
                continue

            for c in range(C_num):
                if K_upc[u, p, c] == 0:
                    continue

                tx_sum = gp.quicksum(
                    column_pool[r][t]["e_tx"][u, p, c] * mu[r][t]
                    for r in range(flow_num)
                    for t in range(len(column_pool[r]))
                )

                rx_sum = gp.quicksum(
                    column_pool[r][t]["e_rx"][u, p, c] * mu[r][t]
                    for r in range(flow_num)
                    for t in range(len(column_pool[r]))
                )

                constr_sc_dir_tx[(u, p, c)] = model.addConstr(
                    tx_sum <= big_M * z[u, p, c],
                    name=f"sc_dir_tx_{u}_{p}_{c}"
                )

                constr_sc_dir_rx[(u, p, c)] = model.addConstr(
                    rx_sum <= big_M * (1 - z[u, p, c]),
                    name=f"sc_dir_rx_{u}_{p}_{c}"
                )

    # ------------------------------------------------------------------
    # 8. 约束 4：SC 容量一致性约束
    #
    #    这一条我按你前面 master problem 的原公式逻辑“原样落代码”
    #
    #    对每个 (r,u,p,c)：
    #    Σ_{所有流、所有列} 5*(n_tx+n_rx)*mu
    #       <=
    #    Σ_{flow=r 的列} a*mu + M*(1 - Σ_{flow=r 的列} xi*mu)
    #
    #    直白解释：
    #    - 左边是全局在 (u,p,c) 上的总需求
    #    - 右边表示：只有当 flow r 选择了一条真的激活 xi=1 的列时，
    #      它对应的 a[u,p,c] 才会变成有效容量上界；
    #      否则靠 Big-M 把约束放松掉
    #
    #    注意：
    #    这条约束从“建模语义”上看是比较特别的，
    #    但这里我先完全按你原来的 master problem 公式实现，不擅自改。
    # ------------------------------------------------------------------
    constr_sc_cap = {}

    for r in range(flow_num):
        for u in range(U_num):
            for p in range(P_num):
                if H_up[u, p] == 0:
                    continue

                for c in range(C_num):
                    if K_upc[u, p, c] == 0:
                        continue

                    lhs_total = gp.quicksum(
                        5.0 * (column_pool[rr][tt]["n_tx"][u, p, c] + column_pool[rr][tt]["n_rx"][u, p, c]) * mu[rr][tt]
                        for rr in range(flow_num)
                        for tt in range(len(column_pool[rr]))
                    )

                    rhs_local = gp.quicksum(
                        column_pool[r][t]["a"][u, p, c] * mu[r][t]
                        for t in range(len(column_pool[r]))
                    ) + big_M * (
                        1 - gp.quicksum(
                            column_pool[r][t]["xi"][u, p, c] * mu[r][t]
                            for t in range(len(column_pool[r]))
                        )
                    )

                    constr_sc_cap[(r, u, p, c)] = model.addConstr(
                        lhs_total <= rhs_local,
                        name=f"sc_cap_{r}_{u}_{p}_{c}"
                    )

    # ------------------------------------------------------------------
    # 9. 约束 5：SC 路径一致性
    #
    #    如果某条 flow 的某条列激活了 xi[u,p,c]=1，
    #    那么这条列中给出的 h[u,p,c,e] 必须和全局共享路径 h_tilde[u,p,c,e] 一致
    #
    #    常见写法是两条不等式：
    #    h_tilde - h_sum <= 1 - xi_sum
    #    h_sum - h_tilde <= 1 - xi_sum
    #
    #    当 xi_sum = 1 时，两条式子合起来就是 h_tilde = h_sum
    #    当 xi_sum = 0 时，右边变成 1，这两条约束被放松
    # ------------------------------------------------------------------
    constr_sc_path_pos = {}
    constr_sc_path_neg = {}

    for r in range(flow_num):
        for u in range(U_num):
            for p in range(P_num):
                if H_up[u, p] == 0:
                    continue

                for c in range(C_num):
                    if K_upc[u, p, c] == 0:
                        continue

                    xi_sum = gp.quicksum(
                        column_pool[r][t]["xi"][u, p, c] * mu[r][t]
                        for t in range(len(column_pool[r]))
                    )

                    for e in range(E_num):
                        h_sum = gp.quicksum(
                            column_pool[r][t]["h"][u, p, c, e] * mu[r][t]
                            for t in range(len(column_pool[r]))
                        )

                        constr_sc_path_pos[(r, u, p, c, e)] = model.addConstr(
                            h_tilde[u, p, c, e] - h_sum <= 1 - xi_sum,
                            name=f"sc_path_pos_{r}_{u}_{p}_{c}_{e}"
                        )

                        constr_sc_path_neg[(r, u, p, c, e)] = model.addConstr(
                            h_sum - h_tilde[u, p, c, e] <= 1 - xi_sum,
                            name=f"sc_path_neg_{r}_{u}_{p}_{c}_{e}"
                        )

    # ------------------------------------------------------------------
    # 10. 约束 6：FS 路径一致性
    #
    #     完全类似于 SC 路径一致性，只是这里针对的是 (u,p,w)
    # ------------------------------------------------------------------
    constr_fs_path_pos = {}
    constr_fs_path_neg = {}

    for r in range(flow_num):
        for u in range(U_num):
            for p in range(P_num):
                if H_up[u, p] == 0:
                    continue

                for w in range(W_num):
                    if W_upw[u, p, w] == 0:
                        continue

                    sigma_sum = gp.quicksum(
                        column_pool[r][t]["sigma"][u, p, w] * mu[r][t]
                        for t in range(len(column_pool[r]))
                    )

                    for e in range(E_num):
                        tau_sum = gp.quicksum(
                            column_pool[r][t]["tau"][u, p, w, e] * mu[r][t]
                            for t in range(len(column_pool[r]))
                        )

                        constr_fs_path_pos[(r, u, p, w, e)] = model.addConstr(
                            tau_tilde[u, p, w, e] - tau_sum <= 1 - sigma_sum,
                            name=f"fs_path_pos_{r}_{u}_{p}_{w}_{e}"
                        )

                        constr_fs_path_neg[(r, u, p, w, e)] = model.addConstr(
                            tau_sum - tau_tilde[u, p, w, e] <= 1 - sigma_sum,
                            name=f"fs_path_neg_{r}_{u}_{p}_{w}_{e}"
                        )

    # ------------------------------------------------------------------
    # 11. 约束 7：全局 FS non-overlap
    #
    #     对每一条物理链路 e、每一个大频谱槽 f：
    #     所有 flow / 所有列 / 所有(u,p) 在这个位置上的占用总和不能超过 1
    #
    #     这里分别对 tx 和 rx 做一份
    # ------------------------------------------------------------------
    constr_nonover_tx = {}
    constr_nonover_rx = {}

    for e in range(E_num):
        for f in range(F_num):

            lhs_tx = gp.quicksum(
                column_pool[r][t]["varsigma_tx"][e, f, u, p] * mu[r][t]
                for r in range(flow_num)
                for t in range(len(column_pool[r]))
                for u in range(U_num)
                for p in range(P_num)
                if H_up[u, p] == 1
            )

            lhs_rx = gp.quicksum(
                column_pool[r][t]["varsigma_rx"][e, f, u, p] * mu[r][t]
                for r in range(flow_num)
                for t in range(len(column_pool[r]))
                for u in range(U_num)
                for p in range(P_num)
                if H_up[u, p] == 1
            )

            constr_nonover_tx[(e, f)] = model.addConstr(
                lhs_tx <= 1,
                name=f"nonover_tx_{e}_{f}"
            )

            constr_nonover_rx[(e, f)] = model.addConstr(
                lhs_rx <= 1,
                name=f"nonover_rx_{e}_{f}"
            )

    # ------------------------------------------------------------------
    # 12. 开始求解
    # ------------------------------------------------------------------
    model.optimize()

    # ------------------------------------------------------------------
    # 13. 整理结果
    # ------------------------------------------------------------------
    result = {}
    result["model"] = model

    # 如果模型没解出来，这里直接返回基本信息
    if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
        result["status"] = model.Status
        result["obj"] = None
        result["mu_value"] = None
        result["chi_value"] = None
        result["duals"] = None
        result["selected_columns"] = None
        return result

    result["status"] = model.Status
    result["obj"] = model.ObjVal

    # mu_value[r][t]
    mu_value = []
    for r in range(flow_num):
        temp = np.zeros(len(column_pool[r]), dtype=float)
        for t in range(len(column_pool[r])):
            temp[t] = mu[r][t].X
        mu_value.append(temp)
    result["mu_value"] = mu_value

    # chi_value[r]
    chi_value = np.zeros(flow_num, dtype=float)
    for r in range(flow_num):
        chi_value[r] = chi[r].X
    result["chi_value"] = chi_value

    # 被选中的列
    selected_columns = []
    for r in range(flow_num):
        temp = []
        for t in range(len(column_pool[r])):
            if mu[r][t].X > 1e-6:
                temp.append((t, mu[r][t].X))
        selected_columns.append(temp)
    result["selected_columns"] = selected_columns

    # ------------------------------------------------------------------
    # 14. 如果是 LP 松弛，则提取对偶变量
    #
    #     这里特别注意一个符号问题：
    #     在“最小化问题 + <= 约束”下，Gurobi 返回的 Pi 与论文里常写的
    #     非负对偶变量符号方向通常相反。
    #
    #     所以这里统一做一次“论文符号化处理”：
    #     - 等式约束：dual = Pi
    #     - <= 约束：dual = -Pi
    #
    #     这样你后面写 reduced cost 时，直接按论文公式代入就行。
    # ------------------------------------------------------------------
    if lp_relaxation and model.Status == GRB.OPTIMAL:

        duals = {}

        # alpha[r] : 对应 assign 等式约束
        alpha = np.zeros(flow_num, dtype=float)
        for r in range(flow_num):
            alpha[r] = constr_assign[r].Pi

        # delta[u,p] : 对应 group_cap
        delta = np.zeros((U_num, P_num), dtype=float)
        for u in range(U_num):
            for p in range(P_num):
                if (u, p) in constr_group_cap:
                    delta[u, p] = -constr_group_cap[(u, p)].Pi

        # eps_tx[u,p,c], eps_rx[u,p,c]
        eps_tx = np.zeros((U_num, P_num, C_num), dtype=float)
        eps_rx = np.zeros((U_num, P_num, C_num), dtype=float)
        for key in constr_sc_dir_tx:
            u, p, c = key
            eps_tx[u, p, c] = -constr_sc_dir_tx[key].Pi
        for key in constr_sc_dir_rx:
            u, p, c = key
            eps_rx[u, p, c] = -constr_sc_dir_rx[key].Pi

        # nu[r,u,p,c]
        nu = np.zeros((flow_num, U_num, P_num, C_num), dtype=float)
        for key in constr_sc_cap:
            r, u, p, c = key
            nu[r, u, p, c] = -constr_sc_cap[key].Pi

        # theta_pos / theta_neg
        theta_pos = np.zeros((flow_num, U_num, P_num, C_num, E_num), dtype=float)
        theta_neg = np.zeros((flow_num, U_num, P_num, C_num, E_num), dtype=float)
        for key in constr_sc_path_pos:
            r, u, p, c, e = key
            theta_pos[r, u, p, c, e] = -constr_sc_path_pos[key].Pi
        for key in constr_sc_path_neg:
            r, u, p, c, e = key
            theta_neg[r, u, p, c, e] = -constr_sc_path_neg[key].Pi

        # psi_pos / psi_neg
        psi_pos = np.zeros((flow_num, U_num, P_num, W_num, E_num), dtype=float)
        psi_neg = np.zeros((flow_num, U_num, P_num, W_num, E_num), dtype=float)
        for key in constr_fs_path_pos:
            r, u, p, w, e = key
            psi_pos[r, u, p, w, e] = -constr_fs_path_pos[key].Pi
        for key in constr_fs_path_neg:
            r, u, p, w, e = key
            psi_neg[r, u, p, w, e] = -constr_fs_path_neg[key].Pi

        # varpi_tx[e,f], varpi_rx[e,f]
        varpi_tx = np.zeros((E_num, F_num), dtype=float)
        varpi_rx = np.zeros((E_num, F_num), dtype=float)
        for key in constr_nonover_tx:
            e, f = key
            varpi_tx[e, f] = -constr_nonover_tx[key].Pi
        for key in constr_nonover_rx:
            e, f = key
            varpi_rx[e, f] = -constr_nonover_rx[key].Pi

        duals["alpha"] = alpha
        duals["delta"] = delta
        duals["eps_tx"] = eps_tx
        duals["eps_rx"] = eps_rx
        duals["nu"] = nu
        duals["theta_pos"] = theta_pos
        duals["theta_neg"] = theta_neg
        duals["psi_pos"] = psi_pos
        duals["psi_neg"] = psi_neg
        duals["varpi_tx"] = varpi_tx
        duals["varpi_rx"] = varpi_rx

        result["duals"] = duals

    else:
        result["duals"] = None

    return result


def Compute_Column_Reduced_Cost(flow_id,
                                column,
                                b_r_value,
                                beta,
                                duals,
                                H_up,
                                K_upc,
                                W_upw,
                                U_num,
                                P_num,
                                C_num,
                                W_num,
                                E_num,
                                F_num,
                                big_M=10000):
    """
    计算某一条候选列对指定 flow 的 reduced cost。

    这个函数的用途就是：
    pricing problem 里你生成出一条“完整候选列”后，
    用 master problem 的对偶变量来算它的 reduced cost。
    如果 reduced cost < 0（按最小化问题的列生成标准），
    就说明这条列值得加入 master problem。

    ----------------------------------------------------------------------
    参数说明
    ----------------------------------------------------------------------
    flow_id : int
        这条候选列属于哪一条业务流 r

    column : dict
        一条候选列，格式和 Create_Empty_Column 返回的一样

    b_r_value : float
        当前这条 flow 的业务需求 b_r

    beta : float
        目标函数中的系数

    duals : dict
        Solve_Master_Problem_By_CG_Type 返回的 result["duals"]

    ----------------------------------------------------------------------
    返回
    ----------------------------------------------------------------------
    rc : float
        reduced cost
    """

    alpha = duals["alpha"]
    delta = duals["delta"]
    eps_tx = duals["eps_tx"]
    eps_rx = duals["eps_rx"]
    nu = duals["nu"]
    theta_pos = duals["theta_pos"]
    theta_neg = duals["theta_neg"]
    psi_pos = duals["psi_pos"]
    psi_neg = duals["psi_neg"]
    varpi_tx = duals["varpi_tx"]
    varpi_rx = duals["varpi_rx"]

    # ------------------------------------------------------------------
    # 1. reduced cost 的常数部分
    #
    #    rc = beta * f_rf_clr - alpha[r] + ...
    # ------------------------------------------------------------------
    rc = beta * column["f_rf_clr"] - alpha[flow_id]

    # ------------------------------------------------------------------
    # 2. group 容量对偶项
    # ------------------------------------------------------------------
    for u in range(U_num):
        for p in range(P_num):
            if H_up[u, p] == 0:
                continue

            rc -= delta[u, p] * b_r_value * (column["g_tx"][u, p] + column["g_rx"][u, p])

    # ------------------------------------------------------------------
    # 3. SC 相关对偶项
    # ------------------------------------------------------------------
    for u in range(U_num):
        for p in range(P_num):
            if H_up[u, p] == 0:
                continue

            for c in range(C_num):
                if K_upc[u, p, c] == 0:
                    continue

                rc -= eps_tx[u, p, c] * column["e_tx"][u, p, c]
                rc -= eps_rx[u, p, c] * column["e_rx"][u, p, c]

                # 对应 SC 容量一致性约束的对偶项
                rc -= nu[flow_id, u, p, c] * (
                    5.0 * (column["n_tx"][u, p, c] + column["n_rx"][u, p, c])
                    - column["a"][u, p, c]
                    + big_M * column["xi"][u, p, c]
                )

                # 对应 SC 路径一致性的对偶项
                for e in range(E_num):
                    rc -= theta_pos[flow_id, u, p, c, e] * (
                        -column["h"][u, p, c, e] + column["xi"][u, p, c]
                    )

                    rc -= theta_neg[flow_id, u, p, c, e] * (
                        column["h"][u, p, c, e] + column["xi"][u, p, c]
                    )

    # ------------------------------------------------------------------
    # 4. FS 相关对偶项
    # ------------------------------------------------------------------
    for u in range(U_num):
        for p in range(P_num):
            if H_up[u, p] == 0:
                continue

            for w in range(W_num):
                if W_upw[u, p, w] == 0:
                    continue

                for e in range(E_num):
                    rc -= psi_pos[flow_id, u, p, w, e] * (
                        -column["tau"][u, p, w, e] + column["sigma"][u, p, w]
                    )

                    rc -= psi_neg[flow_id, u, p, w, e] * (
                        column["tau"][u, p, w, e] + column["sigma"][u, p, w]
                    )

    # ------------------------------------------------------------------
    # 5. 全局 non-overlap 对偶项
    # ------------------------------------------------------------------
    for e in range(E_num):
        for f in range(F_num):
            for u in range(U_num):
                for p in range(P_num):
                    if H_up[u, p] == 0:
                        continue

                    rc -= varpi_tx[e, f] * column["varsigma_tx"][e, f, u, p]
                    rc -= varpi_rx[e, f] * column["varsigma_rx"][e, f, u, p]

    return rc


# ======================================================================
# 下面给一个很小的使用示意
# 你接入自己工程时，重点看“数据怎么组织”，不是看这个 toy case 的数值
# ======================================================================
if __name__ == "__main__":

    # -------------------------
    # 维度设置（示例）
    # -------------------------
    flow_num = 2
    U_num = 2
    P_num = 2
    C_num = 3
    W_num = 3
    E_num = 4
    F_num = 5

    # -------------------------
    # 参数
    # -------------------------
    b_r = np.array([20.0, 25.0], dtype=float)
    beta = 1.0

    # group 容量
    U_cap = np.array([
        [100, 100],
        [100, 100]
    ], dtype=float)

    # 有效的 (u,p)
    H_up = np.array([
        [1, 1],
        [1, 0]
    ], dtype=int)

    # 有效的 (u,p,c)
    K_upc = np.ones((U_num, P_num, C_num), dtype=int)

    # 有效的 (u,p,w)
    W_upw = np.ones((U_num, P_num, W_num), dtype=int)

    # -------------------------
    # 构造列池
    # -------------------------
    column_pool = []

    for r in range(flow_num):
        temp_cols = []

        # 第 0 条列
        col0 = Create_Empty_Column(U_num, P_num, C_num, W_num, E_num, F_num)
        col0["f_rf_clr"] = 1.0

        col0["g_tx"][0, 0] = 1
        col0["g_rx"][1, 0] = 1

        col0["e_tx"][0, 0, 0] = 1
        col0["n_tx"][0, 0, 0] = 2
        col0["a"][0, 0, 0] = 20
        col0["xi"][0, 0, 0] = 1

        col0["h"][0, 0, 0, 1] = 1

        col0["sigma"][0, 0, 0] = 1
        col0["tau"][0, 0, 0, 1] = 1

        col0["varsigma_tx"][1, 2, 0, 0] = 1

        # 第 1 条列
        col1 = Create_Empty_Column(U_num, P_num, C_num, W_num, E_num, F_num)
        col1["f_rf_clr"] = 0.5

        col1["g_tx"][0, 1] = 1
        col1["g_rx"][1, 0] = 1

        col1["e_tx"][0, 1, 1] = 1
        col1["n_tx"][0, 1, 1] = 3
        col1["a"][0, 1, 1] = 25
        col1["xi"][0, 1, 1] = 1

        col1["h"][0, 1, 1, 2] = 1

        col1["sigma"][0, 1, 1] = 1
        col1["tau"][0, 1, 1, 2] = 1

        col1["varsigma_tx"][2, 3, 0, 1] = 1

        temp_cols.append(col0)
        temp_cols.append(col1)
        column_pool.append(temp_cols)

    # -------------------------
    # 先解 LP 松弛版 RMP
    # -------------------------
    result_lp = Solve_Master_Problem_By_CG_Type(
        column_pool=column_pool,
        b_r=b_r,
        beta=beta,
        U_cap=U_cap,
        H_up=H_up,
        K_upc=K_upc,
        W_upw=W_upw,
        U_num=U_num,
        P_num=P_num,
        C_num=C_num,
        W_num=W_num,
        E_num=E_num,
        F_num=F_num,
        big_M=10000,
        lp_relaxation=True,
        output_flag=1
    )

    print("LP 目标值 =", result_lp["obj"])
    print("LP mu =", result_lp["mu_value"])
    print("LP chi =", result_lp["chi_value"])
    print("LP 选中的列 =", result_lp["selected_columns"])

    # -------------------------
    # 如果你已经有对偶变量，就可以算新列的 reduced cost
    # -------------------------
    if result_lp["duals"] is not None:
        test_col = Create_Empty_Column(U_num, P_num, C_num, W_num, E_num, F_num)
        test_col["f_rf_clr"] = 0.2
        test_col["g_tx"][0, 0] = 1
        test_col["e_tx"][0, 0, 0] = 1
        test_col["n_tx"][0, 0, 0] = 2
        test_col["a"][0, 0, 0] = 20
        test_col["xi"][0, 0, 0] = 1
        test_col["h"][0, 0, 0, 1] = 1
        test_col["sigma"][0, 0, 0] = 1
        test_col["tau"][0, 0, 0, 1] = 1
        test_col["varsigma_tx"][1, 2, 0, 0] = 1

        rc = Compute_Column_Reduced_Cost(
            flow_id=0,
            column=test_col,
            b_r_value=b_r[0],
            beta=beta,
            duals=result_lp["duals"],
            H_up=H_up,
            K_upc=K_upc,
            W_upw=W_upw,
            U_num=U_num,
            P_num=P_num,
            C_num=C_num,
            W_num=W_num,
            E_num=E_num,
            F_num=F_num,
            big_M=10000
        )

        print("测试列的 reduced cost =", rc)

    # -------------------------
    # 最后如果你想做整数版主问题
    # -------------------------
    result_ip = Solve_Master_Problem_By_CG_Type(
        column_pool=column_pool,
        b_r=b_r,
        beta=beta,
        U_cap=U_cap,
        H_up=H_up,
        K_upc=K_upc,
        W_upw=W_upw,
        U_num=U_num,
        P_num=P_num,
        C_num=C_num,
        W_num=W_num,
        E_num=E_num,
        F_num=F_num,
        big_M=10000,
        lp_relaxation=False,
        output_flag=1
    )

    print("IP 目标值 =", result_ip["obj"])
    print("IP mu =", result_ip["mu_value"])
    print("IP chi =", result_ip["chi_value"])
    print("IP 选中的列 =", result_ip["selected_columns"])
