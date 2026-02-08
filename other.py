# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @IDE     : PyCharm
# # @Project : ILP.py
# # @File    : other.py
# # @Author  : Wumh
# # @Time    : 2025/11/3 16:27
#
#
# # for f in range(flow_num):
# #     for u in range(topo_num):
# #         # 1. 保证在该scr或者des上需要分配TS
# #         model.addConstr(gp.quicksum(FlexE_scr[f, u, p] for p in range(P2MP_num)) == gp.quicksum(
# #             scr_u[f, i, u] for i in range(topo_num)))
# #         model.addConstr(gp.quicksum(FlexE_des[f, u, p] for p in range(P2MP_num)) == gp.quicksum(
# #             des_u[f, i, u] for i in range(topo_num)))
# #         if u == affected_flow[f][1]:
# #             for r in range(len(flow_acc_DP)):
# #                 if flow_acc_DP[r][6] == affected_flow[f][0] and flow_acc_DP[r][1] == affected_flow[f][1]:
# #                     p = flow_acc_DP[r][7]
# #                     model.addConstr(FlexE_scr[f, u, p] <= 1 + Reconf_f[f])
# #                     model.addConstr(FlexE_scr[f, u, p] >= 1 - Reconf_f[f])
# #         if u == affected_flow[f][2]:
# #             for r in range(len(flow_acc_DP)):
# #                 if flow_acc_DP[r][6] == affected_flow[f][0] and flow_acc_DP[r][2] == affected_flow[f][2]:
# #                     p = flow_acc_DP[r][8]
# #                     model.addConstr(FlexE_des[f, u, p] <= 1 + Reconf_f[f])
# #                     model.addConstr(FlexE_des[f, u, p] >= 1 - Reconf_f[f])
# # for u in range(topo_num):
# #     for p in range(P2MP_num):
# #         # 2. 保证在该scr或者des上的FlexE Group上分配的流小于FlexE Group的容量
# #         model.addConstr(gp.quicksum((FlexE_scr[f, u, p] * affected_flow[f][3]) for f in range(flow_num)) <=
# #                         new_node_P2MP[u][p][4])
# #         model.addConstr(gp.quicksum((FlexE_des[f, u, p] * affected_flow[f][3]) for f in range(flow_num)) <=
# #                         new_node_P2MP[u][p][4])
#
#
#
#
#
#
# # 额外约束
#     # # 流守恒约束
#     # for index, f in enumerate(affected_flow):
#     #     for u in range(topo_num):
#     #         model.addConstr(gp.quicksum(p_uv[index, u, v] for v in range(topo_num)) <= 1)
#     #         model.addConstr(gp.quicksum(p_uv[index, v, u] for v in range(topo_num)) <= 1)
#     #         for v in range(topo_num):
#     #             model.addConstr(p_uv[index, u, v] <= topo_matrix[u, v])
#     #         if f[1] == u:
#     #             model.addConstr(gp.quicksum(p_uv[index, u, v] for v in range(topo_num)) - gp.quicksum(
#     #                 p_uv[index, v, u] for v in range(topo_num)) == f_i[index])
#     #         elif f[2] == u:
#     #             model.addConstr(gp.quicksum(p_uv[index, u, v] for v in range(topo_num)) - gp.quicksum(
#     #                 p_uv[index, v, u] for v in range(topo_num)) == -f_i[index])
#     #         else:
#     #             model.addConstr(gp.quicksum(p_uv[index, u, v] for v in range(topo_num)) - gp.quicksum(
#     #                 p_uv[index, v, u] for v in range(topo_num)) == 0)
#
#
#                 # 使用的调制格式以及使用的SC个数：路径长度 <= 500 Modu_uv = 1; 路径长度 > 500 Modu_uv = 0
#             # for f in range(flow_num):
#             #     for i in range(topo_num):
#             #         model.addConstr(len_uv[f, i] / 500 - Modu_uv[f, i] <= (1 - Modu_uv[f, i]) * M)
#             #         model.addConstr((1 - Modu_uv[f, i]) * M <= (len_uv[f, i] / 500) * M - 1)
#             #         model.addConstr(SC_uv[f, i] == (2 * gp.quicksum(scr_u[f, i, v] for v in range(topo_num)) - Modu_uv[f, i]) * affected_flow[f][3])
#
#     # # 判断是否需要重配置资源：重配置 = 1; 不重配置 = 0
#     # # 源目的SC资源：加倍后充足 = 1; 加倍后不充足 = 0
#     # for f in range(flow_num):
#     #     for i in range(topo_num):
#     #         for u in range(topo_num):
#     #             # 物理链路的源节点
#     #             if i == affected_flow[f][1] and u == affected_flow[f][1]:
#     #                 model.addConstr((1 - affected_flow[f][6]) * (1 - Modu_uv[f, i]) == Reconf_scr_u[f, i, u])
#     #             else:
#     #                 model.addConstr(scr_u[f, i, u] == Reconf_scr_u[f, i, u])
#     #             # 物理链路的目的节点
#     #             if u == affected_flow[f][2]:
#     #                 model.addConstr(Reconf_des_u[f, i, u] <= des_u[f, i, u])
#     #                 model.addConstr(Reconf_des_u[f, i, u] <= 1 - affected_flow[f][7])
#     #                 model.addConstr(Reconf_des_u[f, i, u] <= 1 - Modu_uv[f, i])
#     #                 model.addConstr(Reconf_des_u[f, i, u] >= des_u[f, i, u] - affected_flow[f][7] - Modu_uv[f, i])
#     #             else:
#     #                 model.addConstr(Reconf_des_u[f, i, u] == des_u[f, i, u])
#
#     # 为需要重配置的物理链路上的源目的节点分配FlexE Group资源
#     # for f in range(flow_num):
#     #     for i in range(topo_num):
#     #         for u in range(topo_num):
#
#
#
#
#
#
#
#
#
#
#
#
#
#     # for f in range(flow_num):
#     #     for i in range(topo_num):
#     #         if i == affected_flow[f][1]:
#     #             model.addConstr((SC_uv[f, i] - affected_flow[f][6] - 1) / M <= Reconf_u[f, i])
#     #             model.addConstr(Reconf_u[f, i] <= (SC_uv[f, i] - affected_flow[f][6] - 1 + M) / M)
#     #         elif i == affected_flow[f][2]:
#     #             model.addConstr((SC_uv[f, i] - affected_flow[f][7] - 1) / M <= Reconf_u[f, i])
#     #             model.addConstr(Reconf_u[f, i] <= (SC_uv[f, i] - affected_flow[f][7] - 1 + M) / M)
#
#
#
#
#     # # # 生成物理链路
#     # for index, f in enumerate(affected_flow):
#     #     for i in range(topo_num):
#     #         model.addConstr(gp.quicksum(OEO_uv[index, i, v] for v in range(topo_num)) <= gp.quicksum(PHY_uv[index, i, u, v] for u in range(topo_num) for v in range(topo_num))
#     #                         <= M * gp.quicksum(OEO_uv[index, i, v] for v in range(topo_num)))
#     #         for u in range(topo_num):
#     #             model.addConstr(gp.quicksum(PHY_uv[index, i, u, v] for v in range(topo_num)) - gp.quicksum(PHY_uv[index, i, v, u] for v in range(topo_num)) == OEO_uv[index, i, u])
#     #
#     #
#     # # 查看距离，判断调制格式
#     #
#     #
#     # # 源目的节点判断是否需要进行重配置
#
# # for u in range(topo_num):
# #     for p in range(P2MP_num):
# #         model.addConstr(FS_P2MP_end[u, p] - FS_P2MP_start[u, p] + 1 == type_P2MP[new_node_P2MP[u][p][3] - 1][2])
# #         model.addConstr(
# #             gp.quicksum(FS_P2MP_used[u, p, w] for w in range(max_fs)) == type_P2MP[new_node_P2MP[u][p][3] - 1][2])
# #         if new_node_P2MP[u][p][5] != -1:
# #             model.addConstr(FS_P2MP_start[u, p] == new_node_P2MP[u][p][5])
# #         for w in range(max_fs):
# #             model.addConstr(FS_P2MP_used[u, p, w] <= (w - FS_P2MP_start[u, p]) / M + 1)
# #             model.addConstr(FS_P2MP_used[u, p, w] <= (FS_P2MP_end[u, p] - w) / M + 1)
#
# # 3.1 判断流在节点处使用的P2MP是否相同
# for f_1 in range(flow_num):
#     for f_2 in range(flow_num):  # 只做上三角即可
#         if f_1 != f_2:
#             for u in range(topo_num):
#                 for p in range(P2MP_num):
#                     model.addConstr(sameUP_z[f_1, f_2, u, p] <= FlexE_scr[f_1, u, p])
#                     model.addConstr(sameUP_z[f_1, f_2, u, p] <= FlexE_scr[f_2, u, p])
#                     model.addConstr(sameUP_z[f_1, f_2, u, p] >=
#                                     FlexE_scr[f_1, u, p] + FlexE_scr[f_2, u, p] - 1)
#                 # 每个 (f1,f2,u) 至多有一个 p 同时为 1，所以求和就是 0/1
#                 model.addConstr(sameUP[f_1, f_2, u] ==
#                                 gp.quicksum(sameUP_z[f_1, f_2, u, p] for p in range(P2MP_num)))
# # 3.2 保证不重叠性
# for f1 in range(flow_num):
#     for f2 in range(flow_num):  # 只考虑不同的两条流；如果你也想约束同一流的不同切片，可以把这一行改掉
#         for u1 in range(topo_num):
#             for u2 in range(topo_num):
#                 for v1 in range(topo_num):
#                     for v2 in range(topo_num):
#
#                         o = orderFS[f1, u1, f2, u2, v1, v2]
#
#                         # 只有在“二者都用这条链路”且“不是同一节点同一 P2MP”时，才需要强制不重叠
#                         # gate = 0 时约束生效；gate >= 1 时约束被 big-M 放宽
#                         if u1 == u2:
#                             # 同一节点 u1：如果 sameUP=1，说明是“同一节点的同一 P2MP”，允许重叠
#                             gate = (2
#                                     - PHY_uv[f1, u1, v1, v2]
#                                     - PHY_uv[f2, u2, v1, v2]
#                                     + sameUP[f1, f2, u1])
#                         else:
#                             # 不同节点：一定不允许重叠，所以这里没有 sameUP，默认当作 0
#                             gate = (2
#                                     - PHY_uv[f1, u1, v1, v2]
#                                     - PHY_uv[f2, u2, v1, v2])
#
#                         # 非重叠的“左右”两种情况（二选一）
#                         # 情况 A： (f1,u1) 在左边，结束点 <= (f2,u2) 的起点
#                         model.addConstr(
#                             FS_uv_end[f1, u1, v1, v2]
#                             <= FS_uv_start[f2, u2, v1, v2]
#                             + M * (o + gate)
#                         )
#                         # 情况 B： (f2,u2) 在左边，结束点 <= (f1,u1) 的起点
#                         model.addConstr(
#                             FS_uv_end[f2, u2, v1, v2]
#                             <= FS_uv_start[f1, u1, v1, v2]
#                             + M * (1 - o + gate)
#                         )