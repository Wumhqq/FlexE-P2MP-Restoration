# `cg_algorithm1_overview.py` 逻辑与流程说明

## 1. 文件定位

这个文件对应论文里的 **Algorithm 1（Overall Procedure of hAG-S1F-CP-CG+）**。

它不是负责单条路径上的列生成，而是负责整套列生成算法的“总控流程”。

它解决的问题是：

- 哪些物理候选路径需要预先构建
- 每条流要构建哪些逻辑候选路径
- warm start 怎么做
- 初始活动列池 `Omega^0` 怎么来
- 候选列仓库 `Lambda` 怎么扩张
- 什么时候解 LP-RMP
- 什么时候根据 reduced cost 把列加进主问题
- 什么时候停止 path-rank 扩张
- 最后怎么做整数主问题求解

可以把它理解成：

> 这份文件不直接创造列，而是负责调度整个列生成过程。

---

## 2. 文件依赖与复用关系

这个文件主要依赖三个方向的模块。

### 2.1 复用 `rmp_master_problem.py`

复用了：

- `Column`
- `RMPData`
- `compute_reduced_cost`
- `get_selected_columns`
- `solve_rmp_ip`
- `solve_rmp_lp`

作用：
- 不重写 master problem 的建模逻辑
- 直接复用你已经整理好的 RMP 求解与 reduced cost 计算接口

---

### 2.2 复用 `cg_algorithm3_generate_pool.py`

复用了：

- `GeneratedColumn`
- `_column_signature`
- `_extract_flow_metadata_from_base`
- `generate_candidate_column_pool_on_path`
- `try_candidate_paths_for_one_flow`

作用：
- Algorithm 1 不自己重写“固定路径生成列池”这一步
- 而是把这一步交给 Algorithm 3 文件完成

因此，这两个文件的关系是：

- `cg_algorithm3_generate_pool.py`：负责单路径生成列池
- `cg_algorithm1_overview.py`：负责决定什么时候、给哪条流、在哪个 path rank 上调用 Algorithm 3

---

### 2.3 可选复用外部模块

文件中还会尝试导入一些你项目里已有的模块：

- `k_shortest_path`
- `other_fx_fix.build_virtual_topology`
- `AG_S1F`

这些依赖都不是强制死绑定，而是：
- 如果环境里有，就优先复用
- 如果没有，就允许你通过回调参数显式传入

#### `k_shortest_path`
用于：
- 预先计算物理候选路径池
- 计算每条流的候选逻辑路径

#### `other_fx_fix.build_virtual_topology`
用于：
- 构建 flow-specific 的 AG（辅助图）

#### `AG_S1F`
目前主要是兼容性导入：
- 如果 `AG_S1F.py` 里有 `math` 或 `TS_UNIT` 缺失，会在导入后做补丁

作用：
- 提高整个流程与已有工程环境的兼容性
- 不另起一套并行实现

---

## 3. 核心数据结构

### 3.1 `WarmStartResult`

表示 warm start 阶段的输出，包含：

- `restored_columns`
  - 记录哪些流在 warm start 阶段已经恢复成功
  - 每条成功流对应一条 `GeneratedColumn`

- `state_after`
  - warm start 执行完成后的网络状态

- `tier1_flow_ids`
  - 被分到 Tier 1 的流

- `tier2_flow_ids`
  - 被分到 Tier 2 的流

这个结构的作用是：
- 保存 warm start 的实际恢复结果
- 作为初始活动列池 `Omega^0` 的来源

---

### 3.2 `CPGRunResult`

表示 Algorithm 1 整体运行后的输出，包含：

- `warm_start`
- `omega`
- `lambda_repo`
- `lp_objectives`
- `improvements`
- `final_rmp`
- `final_selected_columns`

它记录了整个列生成过程中的核心结果。

其中：

- `omega`
  - 活动列池，真正进入 master problem 的列

- `lambda_repo`
  - 候选列仓库，包含已经生成出来但不一定已经进入主问题的列

- `lp_objectives`
  - 每个阶段 LP-RMP 的目标值

- `improvements`
  - 每个 path-rank 扩张带来的目标改善量

- `final_selected_columns`
  - 最终整数 RMP 选中的列

---

## 4. 基础函数的作用

### 4.1 `_deepcopy_state(state)`

作用：
- 深拷贝网络状态
- 避免 warm start 或其他流程直接污染原始输入状态

和 Algorithm 3 文件中的作用一致。

---

### 4.2 `_flow_id(flow)`

作用：
- 统一提取 flow 的 ID
- 简化后续对字典键的访问

因为整个文件内部几乎所有字典都是以 `flow_id` 为 key。

---

### 4.3 `_build_rmp_data(...)`

作用：

> 把当前活动列池 `Omega` 和集合参数 `rmp_sets` 组装成 `RMPData`

输入包括：

- `affected_flows`
- `omega`
- `rmp_sets`

输出：
- `RMPData`

这个函数的意义在于：
- Algorithm 1 本身不关心 RMP 里具体怎么建模
- 它只需要在每次要求解 RMP 时，把当前活动列池和集合参数整理好，交给 `solve_rmp_lp(...)` 或 `solve_rmp_ip(...)`

也就是说，这个函数是：
- “总控层”到“主问题求解层”的数据接口

---

### 4.4 `_merge_generated_columns(...)`

作用：
- 把新生成出来的列并入 `lambda_repo`
- 同时按列签名去重

为什么需要这个函数？

因为：
- 不同 path rank
- 不同 DFS 分支
- 甚至不同路径

都可能生成相同的列。如果不去重：
- `Lambda` 会不断膨胀
- 增加后续 reduced cost 检查和内存消耗

因此，这个函数负责维护：

> 候选列仓库 `Lambda` 的唯一性

---

### 4.5 `_column_ids_in_omega(...)`

作用：
- 返回当前活动列池 `Omega[flow_id]` 中所有列的 `col_id`

用途：
- 在内层 CG 选择新列时，避免把已经进入 `Omega` 的列再次加入

---

## 5. Step 1：构建物理候选路径池

### `build_physical_path_pool(...)`

作用：

> 为所有可能的 `(u,v)` 预先构建最多 `K_phy` 条物理候选路径

输入重点：

- `topo_num`
- `topo_dis`
- `break_node`
- `K_phy`
- `ksp_func`

输出格式：

```python
{
    (u, v): [
        {"path": [0-based nodes], "dist": float},
        ...
    ]
}
```

#### 内部逻辑

- 对所有 `u != v` 且不经过故障节点的节点对 `(u,v)`
- 调用 `k_shortest_path(...)`
- 把 1-based 路径转成 0-based
- 保存到 `phy_pool[(u,v)]`

#### 作用

这一步对应论文里：
- 先构建物理层候选 lightpath 池 `P_{u,v}`

这是后续构建 AG 和逻辑候选路径的基础。

---

## 6. Step 2：构建每条流的逻辑候选路径

### `prepare_candidate_logical_paths(...)`

作用：

> 为每条流构建 flow-specific AG，并从中提取前 `K_lgc` 条逻辑候选路径

输出三类结果：

- `candidate_logical_paths_by_flow[fid]`
- `mapping_tables_by_flow[fid]`
- `path_weights_by_flow[fid]`

#### 输入重点

- `affected_flows`
- `break_node`
- `state_after_failure`
- `topo_num`
- `phy_pool`
- `K_lgc`
- `reconfig_penalty`
- `hop_penalty`
- `base_state`
- `build_ag_func`
- `ksp_func`

#### 内部流程

##### 第一步：逐流处理

对每条受影响流，先提取：
- `fid`
- `src`
- `dst`
- `band`

##### 第二步：如果有 `base_state`，提取原流端资源信息

调用：
- `_extract_flow_metadata_from_base(...)`

作用是：
- 粗略推断源端和宿端原始调制能力
- 为构造 AG 时的重构代价提供依据

##### 第三步：调用 `build_virtual_topology(...)`

得到 flow-specific AG：
- 虚拟节点连通关系
- 虚拟边对应的物理映射关系

##### 第四步：在 AG 上求 K 条逻辑候选路径

调用：
- `k_shortest_path(...)`

得到：
- 候选逻辑路径序列
- 对应路径权重

##### 第五步：整理输出

存入：
- `candidate_paths`
- `mapping_tables`
- `path_weights`

#### 作用总结

这一步回答的是：

> 对于每条流，后面 Algorithm 1 应该优先尝试哪些逻辑路径？

---

## 7. Step 3：Warm Start

### `warm_start_ag_s1f(...)`

作用：

> 用 Strategy-1-first 的方式先构造一个初始可行恢复集合 `Z^0`

它的逻辑基本对应论文中的 warm start / Algorithm 4 思想。

#### 输入重点

- `affected_flows`
- `candidate_logical_paths_by_flow`
- `mapping_tables_by_flow`
- `path_weights_by_flow`
- `state_after_failure`
- `link_index`
- `reconfig_penalty`
- `base_state`

以及两个可选回调：
- `reserve_strategy1_resources_func`
- `rollback_one_reservation_func`

#### 内部流程

##### 第一步：拷贝状态

从 `state_after_failure` 深拷贝出 `state`。

##### 第二步：按最优路径权重把流分层

- 如果某条流的最优逻辑路径权重 `<= reconfig_penalty`
  - 放入 `tier1`
- 否则
  - 放入 `tier2`

这样做的直观含义是：
- `tier1` 更适合优先用 Strategy 1 恢复
- `tier2` 更可能需要允许重构的 Strategy 2

##### 第三步：可选预留 Strategy 1 资源

如果外部传入了 `reserve_strategy1_resources_func`，就执行预留。

这一步是可插拔的，方便以后接你自己更成熟的 reservation 管理逻辑。

##### 第四步：先恢复 Tier 1

对 `tier1` 中每条流：

1. 只保留那些路径权重 `<= reconfig_penalty` 的路径
2. 如果有 `base_state`，提取旧资源 `flow_metadata`
3. 调用 `try_candidate_paths_for_one_flow(...)`
4. 强制 `strict_s1=True`
5. 只要找到第一条成功列，就接受它
6. 用 `gc.state_after` 更新全局状态

如果失败，则加入 `tier1_failed`

##### 第五步：把 `tier1_failed` 转入 `tier2`

这一步与论文思想一致：
- Tier 1 失败的流，后续用 Strategy 2 再尝试

##### 第六步：恢复 Tier 2

对 `tier2` 中还未成功恢复的流：

1. 取全部候选逻辑路径
2. 调用 `try_candidate_paths_for_one_flow(...)`
3. 设 `strict_s1=False`
4. 允许 Strategy 2
5. 找到第一条成功方案就接受
6. 更新状态

#### 输出

返回 `WarmStartResult`，记录：
- warm start 恢复成功的列
- 更新后的网络状态
- Tier 1 / Tier 2 的分层结果

#### 作用总结

warm start 的目标不是枚举大量列，而是：

> 尽快找到一批初始可行恢复列，作为 `Omega^0` 的起点

---

## 8. Step 4-8：Algorithm 1 总入口

### `run_hag_s1f_cp_cg_plus(...)`

这是整个文件最重要的函数，也是论文 Algorithm 1 的主入口。

它串起了：
- warm start
- 初始化 `Omega` / `Lambda`
- path-rank 扩张
- 内层列生成
- LP-RMP 与 IP-RMP 求解

---

## 9. `run_hag_s1f_cp_cg_plus(...)` 的详细执行流程

### 9.1 输入重点

最关键的输入包括：

- `affected_flows`
- `break_node`
- `state_after_failure`
- `rmp_sets`
- `link_index`
- `candidate_logical_paths_by_flow`
- `mapping_tables_by_flow`
- `path_weights_by_flow`
- `base_state`

以及算法控制参数：

- `K_lgc`
- `warm_start_reconfig_penalty`
- `epsilon_a`
- `epsilon_r`
- `I_max`
- `max_columns_per_path`
- `max_schemes_per_hop`
- `verbose`

---

### 9.2 Step A：warm start，得到 `Z^0`

第一步就是调用：

- `warm_start_ag_s1f(...)`

得到 `warm`。

然后初始化两个核心容器：

- `omega`
  - 活动列池
- `lambda_repo`
  - 候选列仓库

对 warm start 成功恢复的流：
- 直接把 `gc.column` 放入 `omega[fid]`

这里的含义是：
- warm start 生成的列，直接作为初始活动列池 `Omega^0`

---

### 9.3 Step B：初始化 LP-RMP，记录 `J^(0)`

接下来：

1. 调用 `_build_rmp_data(...)` 把当前 `omega` 转成 `RMPData`
2. 调用 `solve_rmp_lp(...)` 解初始 LP-RMP
3. 得到初始目标值 `J_prev`

然后计算停止阈值：

- `epsilon = max(epsilon_a, epsilon_r * |J_prev|)`

含义是：
- 只要后续某一轮 path-rank 扩张带来的改善小于这个阈值，就停止继续扩路径

同时初始化记录：
- `lp_objectives[0] = J_prev`
- `improvements = {}`

---

### 9.4 Step C：外层循环——按 path rank 扩张

变量：
- `path_rank = 1`
- `cg_iter = 0`

循环条件：
- `path_rank <= K_lgc`
- `cg_iter < I_max`

这层对应论文里的思想：
- 先只看每条流的第 1 条逻辑候选路径
- 如果还不够，再看第 2 条
- 再不够，再看第 3 条……

也就是按 path rank 逐层扩张列库。

---

### 9.5 Step D：在当前 `path_rank` 上，对每条流调用 Algorithm 3 生成新列

对每条流：

1. 取它当前 rank 对应的逻辑路径 `digamma`
2. 调用：
   - `generate_candidate_column_pool_on_path(...)`
3. 参数中设置：
   - `strategy_mode="both"`
   - 表示 Strategy 1 和 Strategy 2 的列都生成
4. 得到 `new_cols`
5. 调用 `_merge_generated_columns(...)` 把新列并入 `lambda_repo[fid]`

注意：
- 这里新列只是进入 `Lambda`
- 还没有进入 `Omega`

也就是说：

> 这一步是在扩张候选列仓库，而不是立刻扩张主问题。

---

### 9.6 Step E：内层循环——标准 Column Generation

在当前 `Lambda` 已知的情况下，开始经典的 CG 内循环。

变量：
- `added = True`

当 `added=True` 时不断循环：

#### 第一步：解当前 LP-RMP

- `_build_rmp_data(...)`
- `solve_rmp_lp(...)`

得到：
- 当前 LP 解
- 当前 duals

#### 第二步：对每条流找最佳负 reduced cost 列

对每条流：

1. 通过 `_column_ids_in_omega(...)` 获取当前已经进入活动池的列 ID
2. 遍历 `lambda_repo[fid]` 中所有尚未进入 `omega[fid]` 的列
3. 调用 `compute_reduced_cost(...)`
4. 找到 reduced cost 最小的一条 `best_gc`

如果：
- `best_rc < -1e-9`

则把该列加入：
- `omega[fid]`

并将 `added=True`

#### 第三步：如果至少加入了一条列，则继续下一轮内层循环

这意味着：
- 新加入的列可能会改变主问题解与 dual
- 所以需要重新解 LP-RMP，再重新计算 reduced cost

这就是标准列生成的核心过程。

---

### 9.7 Step F：当前 rank 收敛后，记录新的 LP 目标值

当内层 CG 再也找不到负 reduced cost 列后：

1. 再解一次 LP-RMP
2. 得到新的目标值 `J_curr`
3. 保存：
   - `lp_objectives[path_rank] = J_curr`
   - `improvements[path_rank] = J_prev - J_curr`

这里的 `improvements[path_rank]` 表示：
- 当前 path rank 扩张所带来的边际改善

---

### 9.8 Step G：判断是否继续扩下一层 path rank

如果：
- `improvements[path_rank] < epsilon`

则停止外层 path-rank 扩张。

含义是：
- 当前继续引入更高 rank 的逻辑路径，已经几乎不能再明显改进目标值

否则：
- `J_prev = J_curr`
- `path_rank += 1`
- 继续下一轮外层循环

因此，停止条件不是简单的“没有新列”，而是：

> 新增一层逻辑候选路径带来的边际收益已经太小

---

### 9.9 Step H：最终整数 RMP

当外层循环结束后：

1. 用当前活动列池 `omega` 构造最终 `RMPData`
2. 调用 `solve_rmp_ip(...)`
3. 调用 `get_selected_columns(...)`

输出最终整数解中真正被选中的列。

这一步对应论文最后的：
- 在列池收敛后，求整数 master problem

---

## 10. CLI 执行逻辑

文件底部提供了命令行执行入口。

### 输入 pickle 推荐结构

```python
{
  "affected_flows": ...,
  "break_node": ...,
  "state_after_failure": ...,
  "rmp_sets": ...,
  "link_index": ...,
  "candidate_logical_paths_by_flow": ...,
  "mapping_tables_by_flow": ...,
  "path_weights_by_flow": ...,
  "base_state": ...,
  "K_lgc": ...,
  "warm_start_reconfig_penalty": ...,
  "max_columns_per_path": ...,
  "max_schemes_per_hop": ...,
}
```

### 执行逻辑

命令行运行时会：

1. 用 `_load_pickle(...)` 读取输入
2. 调用 `run_hag_s1f_cp_cg_plus(...)`
3. 用 `_save_pickle(...)` 保存输出结果 `CPGRunResult`

因此，这个文件既可以：
- 被其他 Python 文件当模块调用
- 也可以直接作为可执行脚本运行

---

## 11. 与论文 Algorithm 1 的对应关系

可以把论文 Algorithm 1 和代码函数直接对应起来：

### 论文：构建物理路径池 `P_{u,v}`
对应代码：
- `build_physical_path_pool(...)`

### 论文：为每条流构建 AG，并生成逻辑候选路径 `L_r`
对应代码：
- `prepare_candidate_logical_paths(...)`

### 论文：warm start，得到 `Z^0`
对应代码：
- `warm_start_ag_s1f(...)`

### 论文：初始化 `Omega^0`、`Lambda^(0)`
对应代码：
- `run_hag_s1f_cp_cg_plus(...)` 中 warm start 之后的初始化部分

### 论文：按 path rank 扩张列库
对应代码：
- `run_hag_s1f_cp_cg_plus(...)` 外层 while 循环

### 论文：固定当前 rank，调用 Algorithm 3 生成新列
对应代码：
- `generate_candidate_column_pool_on_path(...)`
- 再通过 `_merge_generated_columns(...)` 并入 `lambda_repo`

### 论文：求 LP-RMP，取 dual，加入负 reduced cost 列
对应代码：
- 内层 CG while 循环
- `solve_rmp_lp(...)`
- `compute_reduced_cost(...)`
- `omega[fid].append(...)`

### 论文：若边际改善足够小，则停止
对应代码：
- `if improvements[path_rank] < epsilon: break`

### 论文：最终求整数 RMP
对应代码：
- `solve_rmp_ip(...)`
- `get_selected_columns(...)`

---

## 12. 这个文件的本质作用总结

这份文件本质上做了四件事情：

### 第一件：准备候选路径空间
通过：
- `build_physical_path_pool(...)`
- `prepare_candidate_logical_paths(...)`

建立物理和逻辑两层候选路径空间

### 第二件：给主问题一个好初值
通过：
- `warm_start_ag_s1f(...)`

快速找出一批初始可行恢复列，形成 `Omega^0`

### 第三件：做标准列生成迭代
通过：
- `run_hag_s1f_cp_cg_plus(...)`

反复执行：
- 扩张候选路径 rank
- 调用 Algorithm 3 生成列
- 解 LP-RMP
- 用 reduced cost 选列进入 `Omega`

### 第四件：最终输出整数解
通过：
- `solve_rmp_ip(...)`

从活动列池中选出最终最优的整数解

因此，你可以把这份文件理解成：

> 它不是负责单条路径恢复的，而是负责“整套列生成算法什么时候做什么”的总调度器。

它把论文 Algorithm 1 中原本抽象的高层流程，真正连接成了一个可执行的列生成主循环。
