# `cg_algorithm3_generate_pool.py` 逻辑与流程说明

## 1. 文件定位

这个文件对应论文里的 **Algorithm 3（Pricing Routine / Generate Columns on Path）**。
它的核心任务不是直接求解整个列生成主问题，而是：

- 固定一条受影响流 `r`
- 固定这条流的一条逻辑候选路径 `digamma`
- 在当前失败后网络状态 `C0` 下，沿着这条逻辑路径逐 hop 尝试恢复
- 把所有可行的完整恢复方案转换成 master problem 可以接受的 `Column`
- 返回这条逻辑路径对应的候选列池

可以把它理解成：

> 这是一个“单路径列池生成器”，负责回答：
> **对于流 `r` 的路径 `digamma`，到底能生成哪些可行列？**

---

## 2. 文件依赖与复用关系

这个文件优先复用你已有工程中的实现，而不是重新写底层分配逻辑。

### 2.1 复用 `rmp_master_problem.py`

主要复用了：

- `Column`

作用：
- 保证这里生成出来的列，和 master problem 里的输入格式完全一致
- 避免再单独定义一套列结构，造成后续对接混乱

### 2.2 复用 `Initial_net.py`

主要复用了以下底层函数：

- `_apply_leaf_usage`
- `_apply_p2mp_fs_usage`
- `_clear_p2mp_sc`
- `_find_free_fs_block`
- `_sc_can_use_on_hub`
- `_sc_fs_path_ok`
- `_sc_used_bw_for_path`
- `_find_leaf_sc_segment_with_base`
- `_infer_leaf_base_and_find_sc`
- `_alloc_leaf_fixed_usage`

作用：
- 直接复用你现有代码里对 SC / FS 可用性、hub / leaf 分配、路径频谱对齐等规则
- 保证 Algorithm 3 生成列池时使用的资源语义，和你当前工程里真正执行恢复时一致

### 2.3 `sc_effective_cap`

文件里优先尝试从 `Initial_net.py` 复用 `sc_effective_cap`。
如果导入失败，才使用本文件里的兜底版本。

作用：
- 避免因为某些环境里没有这个函数而导致整份文件不能执行
- 但只要你的项目中已有该函数，就会优先复用你原来的实现

---

## 3. 核心数据结构

### 3.1 `HopScheme`

这个数据结构表示：

> 一条逻辑 hop 的一个可行恢复方案

对应论文 Algorithm 3 中的：
- `q ∈ Q_{u,v}`

它记录了一个 hop 恢复所需的全部关键信息，例如：

- `sub_id`：这个 hop 对应哪个子流
- `orig_flow_id`：原始流 ID
- `src_node / dst_node`：这个 hop 的起止物理节点
- `bandwidth / sc_cap`：带宽需求与 SC 容量
- `hub_p / leaf_p`：hub 和 leaf 使用的 P2MP 端口
- `hub_sc_start / hub_sc_end`：hub 侧 SC 区间
- `leaf_sc_start / leaf_sc_end`：leaf 侧 SC 区间
- `fs_abs_start / fs_abs_end`：全局绝对 FS 区间
- `hub_usage / leaf_usage`：每个 SC 上具体用了多少带宽
- `phys_nodes_1b`：1-based 物理路径节点序列
- `used_links_0b / used_link_pairs_0b`：0-based 物理链路索引及边对
- `strict_s1`：该 hop 是否属于 Strategy 1
- `logical_hop_1b`：对应的逻辑 hop `(u,v)`

也就是说，`HopScheme` 不是列本身，而是列里的“一段 hop 级恢复片段”。

---

### 3.2 `GeneratedColumn`

这个数据结构表示：

> 一条完整列，以及生成这条列时的辅助信息

它主要包含：

- `column`：真正给 master problem 用的 `Column`
- `hop_schemes`：这条列由哪些 `HopScheme` 串起来得到
- `logical_path`：该列对应的逻辑路径 `digamma`
- `state_after`：执行完整方案后网络状态（可选）

这里保留 `state_after` 的原因是：

- 生成列池时，只需要 `column`
- 但 warm start 或“找到一条可行方案后直接更新网络状态”的流程里，还需要知道这条列写回状态之后的结果

---

## 4. 基础工具函数的作用

### 4.1 `_deepcopy_state(state)`

作用：
- 深拷贝当前网络状态
- 保证 DFS 不同分支之间互不干扰

这个函数拷贝的状态包括：

- `flow_acc`
- `node_flow`
- `node_P2MP`
- `P2MP_SC`
- `link_FS`
- `P2MP_FS`
- `flow_path`

在 Algorithm 3 中非常重要，因为：
- 每走一条 DFS 分支，都要假设该分支已经占用了资源
- 但不能污染其他分支

---

### 4.2 `_build_subflow_lookup(flow_acc)`

作用：
- 建立 `(orig_flow_id, hop_a, hop_b) -> subflow_id` 的映射

原因：
- 一条逻辑路径会拆成多个 hop
- 底层网络状态往往是按“子流”来记录，而不是直接按原始流的整条逻辑路径记录

因此在生成某个 hop 的方案时，需要先知道：
- 这个逻辑 hop 对应的底层 `subflow_id` 是多少

---

### 4.3 `_extract_flow_metadata_from_base(...)`

作用：
- 从 `base_state` 中提取 Strategy 1 需要的旧流端资源信息

主要提取：

- `hub_idx / leaf_idx`
- `hub_cap / leaf_cap`
- `hub_sc_range / leaf_sc_range`
- `hub_sc_usage / leaf_sc_usage`

用途：
- 如果当前要生成 Strategy 1 的列，那么就不能随意重构端资源
- 必须尽量沿用原来 hub / leaf 的端口、SC 区间和使用方式

因此，这个函数其实是在为 Strategy 1 提供“固定资源约束条件”。

---

### 4.4 `_compute_used_links_and_pairs(phy_path0, link_index)`

作用：
- 把物理路径上的节点序列转换成：
  - `used_links_0b`
  - `used_link_pairs_0b`

用途：
- 后面构造 `HopScheme`
- 后面写回 `flow_path`
- 构造列中的 `h / tau / varsigma`

这个函数的本质是：
- 把“节点路径表示”转成“链路表示”

---

## 5. 单 hop 枚举逻辑

这一层的核心思想是：

> 先为某一个逻辑 hop 生成所有可行恢复方案 `Q_{u,v}`。

### 5.1 `_enumerate_leaf_allocations(...)`

作用：
- 在 hub 侧方案已经基本确定后，枚举所有可行 leaf 分配方案

输入中最关键的内容包括：

- `dest`：目的节点
- `src_node / src_p`：当前对应的 hub 节点和端口
- `fs_s_glo / fs_e_glo`：hub 侧映射得到的全局绝对 FS 区间
- `phys_nodes`：物理路径
- `usage`：hub 侧每个 SC 的带宽占用情况
- `sc_cap`：SC 容量

函数返回的是多个候选 leaf 方案，每个方案包含：

- `leaf_p`
- `leaf_type`
- `leaf_base_fs`
- `leaf_usage_start`
- `leaf_usage_dict`
- `is_new_leaf`

#### 内部流程

这个函数内部是两阶段：

##### 第一步：优先枚举已启用 leaf

逻辑是：
- 如果某个 leaf 端口已经启用，就先尝试在这个已有端口上对齐 hub 的绝对 FS 区间
- 调用 `_find_leaf_sc_segment_with_base(...)`
- 如果能找到对应的 leaf SC 区间，则该 leaf 方案可行

这样做的好处是：
- 尽可能减少资源重构
- 优先复用已有 leaf

##### 第二步：再枚举空闲 leaf

如果已启用 leaf 中没有合适方案，就尝试新开一个空闲 leaf 端口：
- 调用 `_infer_leaf_base_and_find_sc(...)`
- 找出一个新 leaf 对应的基准 FS 和可行 SC 段

##### 特殊情况：固定 leaf

如果调用时传入了：
- `fixed_leaf`
- `fixed_leaf_sc_range`
- `fixed_leaf_usage`

那么函数就会只尝试该固定 leaf 方案，并调用 `_alloc_leaf_fixed_usage(...)`。

这正对应 Strategy 1 的约束：
- leaf 不能随便换
- SC 区间和用量必须和原方案保持一致

---

### 5.2 `_enumerate_sc_plans(...)`

这是单 hop 方案生成中最关键的函数之一。

作用：

> 在给定 hub 之后，枚举所有可行的 SC / FS / leaf 联合方案。

可以把它看成：
- `Initial_net._plan_sc_allocation` 的“多方案枚举版”

#### 输入重点

主要输入包括：

- `src / dest`
- `hub_p / hub_type / hub_fs0`
- `phys_nodes_1b`
- `bw / sc_cap`
- `flow_acc / P2MP_SC / node_P2MP / node_flow / link_FS / P2MP_FS / flow_path`
- `used_links / used_link_pairs`

以及 Strategy 1 相关的固定条件：
- `fixed_hub_sc_range`
- `fixed_hub_usage`
- `fixed_leaf`
- `fixed_leaf_sc_range`
- `fixed_leaf_usage`

#### 内部流程

##### 情况 A：固定 hub SC 方案（Strategy 1）

如果给了 `fixed_hub_sc_range` 和 `fixed_hub_usage`：
- 直接把 hub 侧 SC 区间固定下来
- 检查区间是否合法
- 检查 usage 是否恰好覆盖该区间
- 检查 usage 总和是否等于该 hop 的带宽
- 对区间内每个 SC 检查：
  - `_sc_can_use_on_hub(...)`
  - `_sc_fs_path_ok(...)`
  - `_sc_used_bw_for_path(...)`

只有这些都通过，hub 侧方案才算可行

##### 情况 B：非固定模式（普通枚举）

如果不是 Strategy 1，就从某个 `sc_start` 开始，按连续 SC 贪心装入带宽：
- 从 `sc_start` 开始逐个 SC 尝试
- 每个 SC 检查是否可用
- 如果可用，则根据该 SC 还能承载多少带宽来分配 usage
- 直到带宽装满或者失败

##### 接着：映射到绝对 FS

一旦 hub 侧的 SC 区间和 usage 确定了，就可以通过：
- hub 的 `base_fs`
- SC 区间跨度

得到全局绝对 FS 区间：
- `fs_abs_start`
- `fs_abs_end`

##### 最后：调用 `_enumerate_leaf_allocations(...)`

leaf 侧枚举并不是在外面单独做，而是在 hub 侧方案已经确定后直接调用 `_enumerate_leaf_allocations(...)`。

所以，这个函数最后得到的是：
- 一个完整的 hub + SC + FS + leaf 组合方案集合

每个方案最后会被整理成一个字典，并返回给上层。

---

### 5.3 `generate_hop_restoration_schemes(...)`

这个函数是单 hop 方案生成的总入口。

作用：

> 给定一个逻辑 hop `(u,v)`，生成这个 hop 的全部可行恢复方案集合 `Q_{u,v}`。

#### 输入重点

- `flow`
- `logical_hop_1b`
- `mapping_table`
- `state`
- `link_index`
- `strict_s1`
- `base_state / flow_metadata`

#### 内部流程

##### 第一步：确定这个 hop 对应哪个 subflow

利用：
- `_build_subflow_lookup(...)`

根据：
- 原始流 ID
- hop 起点
- hop 终点

找到底层对应的 `sub_id`。

##### 第二步：读取物理路径

根据 `mapping_table[(u,v)]`：
- 得到该逻辑 hop 对应的物理路径
- 同时调用 `_compute_used_links_and_pairs(...)` 把路径转成链路形式

##### 第三步：计算该 hop 需要的 SC 容量

根据：
- 带宽
- 路径长度
- 调制相关规则

推导 `sc_cap`。

##### 第四步：如果是 Strategy 1，则提取固定元信息

如果 `strict_s1=True`：
- 从 `flow_metadata` 或 `base_state` 中读取旧方案的 hub / leaf / SC 使用方式
- 后续只能在这些固定资源条件下枚举方案

##### 第五步：枚举 hub

这个函数内部会分成两类 hub：

###### 先尝试已启用 hub
- 查看某个 hub 端口是否已启用
- 如果是，则在这个 hub 上继续调用 `_enumerate_sc_plans(...)`

###### 再尝试新开 hub
- 如果已有 hub 不够用，则尝试在空闲端口上新开 hub
- 找可用 FS block
- 再调用 `_enumerate_sc_plans(...)`

##### 第六步：把 `_enumerate_sc_plans(...)` 的结果转换成 `HopScheme`

每个完整的 hub/leaf/SC/FS 方案都被封装成一条 `HopScheme`。

因此，这个函数最终输出的是：
- `List[HopScheme]`

也就是论文中的：
- `Q_{u,v}`

---

## 6. 状态写回逻辑

### `apply_hop_scheme_to_state(hs, state)`

作用：
- 把一条 `HopScheme` 真实写入当前网络状态

这个函数会修改：

- `flow_acc`
- `node_flow`
- `node_P2MP`
- `P2MP_SC`
- `P2MP_FS`
- `flow_path`
- `link_FS`

为什么必须有这个函数？

因为 Algorithm 3 并不是独立地为每个 hop 找一条局部方案，而是：
- 在一条逻辑路径上逐 hop 生成完整列
- 后面的 hop 必须看到前面的 hop 已经占用了哪些资源

所以每选中一个 hop 方案，都必须把它写进状态，再继续递归。

这就是 DFS 能保证“整条列全局可行”的根本原因。

---

## 7. 从 hop 方案组装成列

### 7.1 `_append_sparse(...)`

作用：
- 向 `Column` 中的稀疏字典追加值
- 如果某个键已经存在，就进行累加

用途：
- 统一处理 `g_tx / g_rx / e_tx / e_rx / n_tx / n_rx / ...` 等稀疏字段

---

### 7.2 `build_column_from_scheme(...)`

作用：

> 把一组 `HopScheme`（即完整的 `Pi`）转换成 master problem 需要的一条 `Column`

#### 输入

- `flow`
- `logical_path`
- `hop_schemes`
- `col_id`

#### 内部构造内容

##### 1）group 层系数

根据 hop 的 hub / leaf 端口，设置：
- `g_tx`
- `g_rx`

##### 2）SC 层系数

根据每个 hop 的 `hub_usage / leaf_usage` 和 SC 区间，构造：
- `e_tx`
- `e_rx`
- `n_tx`
- `n_rx`
- `a`
- `xi`
- `h`

##### 3）FS 层系数

构造：
- `sigma`
- `tau`
- `varsigma_tx`
- `varsigma_rx`

其中：
- `sigma / tau` 更偏相对 FS / 路径一致性层面
- `varsigma_tx / varsigma_rx` 更偏绝对 FS / 全局 non-overlap 层面

##### 4）重构惩罚 `f_rf_clr`

如果任何一个 hop 不是 Strategy 1，通常会把整条列记成需要重构惩罚。

因此，`build_column_from_scheme(...)` 实际上完成的是：

> 从“路径恢复方案”到“主问题列变量系数”的映射

---

### 7.3 `_column_signature(col)`

作用：
- 生成一条列的签名
- 用于去重

原因：
- 不同 DFS 分支有可能在最终系数上完全一样
- 如果不去重，会造成 `Lambda` 里出现重复列

---

## 8. 整条逻辑路径的列池生成

### `generate_candidate_column_pool_on_path(...)`

这是整个文件最核心的总入口。

作用：

> 对固定流 `r` 和固定逻辑路径 `digamma`，生成这条路径上的候选列池

#### 输入重点

- `flow`
- `logical_path`
- `mapping_table`
- `state_after_failure`
- `link_index`
- `strategy_mode`
- `base_state / flow_metadata`
- `max_columns`
- `max_schemes_per_hop`
- `keep_state_after`
- `col_id_prefix`

#### 内部主流程

##### 第一步：准备初始状态

- 深拷贝 `state_after_failure`
- 根据 `strategy_mode` 决定是否允许：
  - 只生成 Strategy 1 列
  - 只生成 Strategy 2 列
  - 两者都生成

##### 第二步：定义 DFS

内部会定义一个递归函数，参数通常包括：
- 当前处理到第几个 hop
- 当前状态
- 当前已选 hop 方案序列 `pi`
- 当前是否严格执行 Strategy 1

##### 第三步：DFS 逐 hop 展开

如果当前还没处理完所有 hop：
- 取当前逻辑 hop `(a,b)`
- 调用 `generate_hop_restoration_schemes(...)`
- 得到这个 hop 的候选集合 `Q_{a,b}`

然后对其中每个 `HopScheme`：
- 拷贝状态
- 调用 `apply_hop_scheme_to_state(...)`
- 把该 hop 方案写入新状态
- 继续递归下一个 hop

##### 第四步：如果所有 hop 都处理完

说明当前 `pi` 已经是一条完整路径上的恢复方案。

这时会：
- 调用 `build_column_from_scheme(...)` 生成 `Column`
- 调用 `_column_signature(...)` 去重
- 如果不是重复列，就加入结果集

如果 `keep_state_after=True`：
- 还会把当前分支下的 `state_after` 一起保存

#### 输出

返回：
- `List[GeneratedColumn]`

也就是论文里这条路径对应的候选列池。

---

## 9. 单流多路径尝试

### `try_candidate_paths_for_one_flow(...)`

作用：
- 按顺序尝试一组候选逻辑路径
- 只要找到第一条可行列，就立刻返回成功结果

这个函数不是完整列池枚举，而是：
- 更适合 warm start 或“快速找一条可行列”场景

它的逻辑是：
- 对每条候选逻辑路径调用 `generate_candidate_column_pool_on_path(...)`
- 如果某条路径生成了至少一条列，则返回第一条
- 如果所有路径都失败，则返回失败

因此它和 `generate_candidate_column_pool_on_path(...)` 的区别是：

- `generate_candidate_column_pool_on_path(...)`：生成整条路径上的完整列池
- `try_candidate_paths_for_one_flow(...)`：在多条路径之间找第一条可行方案

---

## 10. 这个文件的完整执行流程

如果按真正执行顺序来理解，可以把整个文件概括成下面这条链：

### Step 1：准备输入
- 给定 `flow`
- 给定 `logical_path`
- 给定 `mapping_table`
- 给定失败后网络状态 `state_after_failure`

### Step 2：把逻辑路径拆成多个 hop
例如：
- `digamma = [1,4,6]`
- 会拆成两个 hop：
  - `(1,4)`
  - `(4,6)`

### Step 3：对每个 hop 调用 `generate_hop_restoration_schemes(...)`
得到：
- `Q_{1,4}`
- `Q_{4,6}`

### Step 4：DFS 逐 hop 枚举
- 从 `Q_{1,4}`` 中选一个 hop 方案
- 写回状态
- 再去 `Q_{4,6}` 中找后续可行方案

### Step 5：拼成完整 `Pi`
如果所有 hop 都成功，说明得到一条完整恢复方案：
- `Pi = [q1, q2, ..., qk]`

### Step 6：`Pi -> Column`
调用 `build_column_from_scheme(...)` 把它转成 master problem 能用的列

### Step 7：加入结果集并去重
通过 `_column_signature(...)` 去重后加入候选列池

### Step 8：返回该路径上的候选列池
输出若干个 `GeneratedColumn`

---

## 11. 与论文 Algorithm 3 的对应关系

可以直接对应成：

### 论文输入
- 流 `r`
- 逻辑路径 `digamma`
- 当前网络状态 `C0`

### 论文过程
- 对路径中每个逻辑 hop `(u,v)` 生成 `Q_{u,v}`
- 枚举 `Pi ∈ Q_{u,v}` 的组合
- 对每个可行 `Pi` 构造列

### 代码实现
- `generate_hop_restoration_schemes(...)` 对应生成 `Q_{u,v}`
- `generate_candidate_column_pool_on_path(...)` 里的 DFS 对应枚举 `Pi`
- `build_column_from_scheme(...)` 对应从 `Pi` 构造列

---

## 12. 总结

这个文件本质上是在做三件事：

### 第一件：单 hop 枚举
通过：
- `generate_hop_restoration_schemes(...)`
- `_enumerate_sc_plans(...)`
- `_enumerate_leaf_allocations(...)`

生成单 hop 的候选恢复方案集合

### 第二件：路径级 DFS 拼接
通过：
- `generate_candidate_column_pool_on_path(...)`
- `apply_hop_scheme_to_state(...)`

把多个 hop 串起来，得到整条路径上的完整恢复方案

### 第三件：构造成主问题列
通过：
- `build_column_from_scheme(...)`

把完整恢复方案转成 master problem 能使用的 `Column`

所以，这个文件最重要的价值在于：

> 它把论文里“固定路径生成列池”的抽象过程，真正落成了一个和你当前资源状态结构兼容、并且尽量复用你已有底层分配函数的可执行模块。
