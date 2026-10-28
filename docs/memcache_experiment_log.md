# Memory-Cache 训练迭代日志

> **2026-10-28 当前状态**：正式 E1/E2 为 `e1_5w_iter.yaml` / `e2_5w_iter.yaml`：发布 VLM+DiT 初始化、ViT 冻结、LM LoRA32、VLM 当前帧，E2 naive cache 仅接 DiT；两组均完成 50k。五阶段内部 held-out val40 与官方四任务 test200 全部完成。27500/38750/50000 步的 E2−E1 test 收益为 +5.5/+5.5/+2.0 个百分点；完整最终记录统一见 `/Users/liam/Projects/research/mem-vla/EXP_LOG.md` §28–29。下文原生 Qwen、随机 DiT 和旧 E1/E2 编号均为历史记录。

> **2026-10-25 最新同步**：已完成发布VLM替换A/B、发布版单帧消融与latency；结果见文末。旧E1/E2编号保持原生Qwen+随机DiT含义。用户确认下一轮cache接DiT；新full-pretrain两组仍为提案，未启动。统计来源风险不等于已证实test训练。

> 本文记录"单帧输入 + 历史 vision-token cache"方向的每一组训练和消融：条件、命令、产物位置、结果、结论。
> 设计方案见 `MEMCACHE_PLAN.md`；模型结构图见 `memcache_architecture.svg`。代码在服务器分支 `memcache-singleframe` 上。
> 服务器副本：`/mnt/cpfs/heping/SimpleMemVLA/docs/memcache_experiment_log.md`（结构图在同一目录）。

> ⚠️ **勘误（10-24）**：10-24 跑的第一版 E1、E2 不光加载了冻结的 VLM，**还把 DiT 也从 `simplememvla_robomme` 热启动了**。发布版 DiT 带有机器人控制先验，且训练 episode 来源未核实，存在与本地 train/val 重叠的风险。这两组不能作为“随机动作头从头学习 memory”的结果；不能仅由全局 stats 认定包含所有评测 episode。
> 现在这两组改名为 **E1w / E2w**（w = warm-started DiT），标记作废，只保留作记录。当时计划的原生 Qwen E1、E2 后续已启动：E1 完成30k，E2在20k评测前后崩溃、最近保存15k。最新零训练评测与下一轮提案见文末同步章节。

> ⚠️ **再次修改（10-24）**：发布版 `stats.json` 的 episode 编号上界3199，统计覆盖约153.6万帧，规模与官方三份划分合计相符，但这不是训练采样清单，不能据此认定3200条全用于训练或包含test（证据边界见§1.1）。这个 VLM 有可能见过评测 episode，所以 **E1/E2/E3 的冻结 VLM 改用原版 Qwen3.5-4B**（`Qwen/Qwen3.5-4B`，rev `851bf6e`，没有经过任何机器人数据微调），**不再加载任何 SimpleMemVLA checkpoint**。memory cache 也已改用原版 Qwen 的 ViT 重新抽取。用发布版 VLM 的配置改名为 `E*_ckptvlm`，保留作为对照。

---

## 0. 怎么记录

- 每组实验一个 ID（`E<编号>`；消融用 `E2a`、`E2b` 这类后缀）。在 §2 总表加一行，在 §3 写详细记录。
- 配置只写**相对参考组的差异**。完整配置以 run 目录里的 `train_config.yaml`（YAML 改造后由 train.py 自动写出）、`config.json` 和 `training_args.bin` 为准。
- 结果统一按 §1.4 的评测协议，表里填 **best val**、**final val** 和两个 memory 诊断量。
- 结论分三档：✅ 有效 / ❌ 无效 / ❓ 待定，并写明依据。

### 配置从哪来（YAML 化之后）

| 层级 | 位置 | 说明 |
|---|---|---|
| 全局 SFT 配方 | `configs/train/base.yaml` | 所有 benchmark 共用的默认值 |
| memcache 公共配置 | `configs/train/memcache/base.yaml` | **冻结的原版 Qwen3.5-4B**（`backbone_model_name_or_path: ${QWEN35_4B:-/mnt/cpfs/heping/checkpoints/Qwen3.5-4B}`，`init_from_checkpoint: null`），DiT 从头训，用数据集自带的 stats，30k 步 |
| 实验变体 | `configs/train/memcache/E1.yaml`、`E2.yaml`、`E3.yaml` | 只切换 `use_memory`、`mem_fusion`、`mem_feature_dir`（`features/robomme_vit_merger_front_qwen35base`） |
| 对照：发布版 VLM | `configs/train/memcache/base_ckptvlm.yaml`、`E*_ckptvlm.yaml` | VLM 用 `simplememvla_robomme` 的 backbone（`init_modules: [backbone]`），DiT 从头训；cache 用 `features/robomme_vit_merger_front` |
| 作废的旧配置 | `configs/train/memcache/base_warmdit.yaml`、`E*_warmdit.yaml` | `init_modules: [backbone, action_head]`，用来复现 E1w / E2w |
| 防误用 | `train.py` | 如果 `mem_feature_dir/meta.json` 记录的 ViT 来源和当前训练的 VLM 不一致，直接报错 |
| 启动 | `scripts/launch_memcache.sh <EXP> <gpus> [--flag v ...]` | EXP 是 `configs/train/memcache/` 下的任意文件名；按卡数自动设 `GRAD_ACCUM`，凑出 global batch 64；`RESUME=auto`；后台运行，log 带时间戳。`DRY_RUN=1` 只打印命令 |
| 参数定义 | `configs/sft_params.py` | 新增 `init_modules`：从 checkpoint 加载哪些模块，backbone 必须加载，action_head 可选 |

---

## 1. 共享设置（新 E1 / E2 / E3）

### 1.1 数据

| 项 | 值 |
|---|---|
| 数据集 | `yinchenghust/robomme_lerobot`（rev `b72788e`），20 fps |
| 子集 | 4 个任务，每个任务 100 条：PickXtimes（700–799）、VideoUnmaskSwap（1500–1599）、VideoRepick（1300–1399）、PatternLock（500–599） |
| 划分 | 每个任务随机留 10 条做 val（seed 0）：train 360 条，共 113,011 个锚点（已去掉 demo 帧）；val 40 条 |
| 划分文件 | `SimpleMemVLA-data/splits/robomme_small_{train,val}.txt` |
| 归一化 | **数据集自己的 `meta/stats.json`**。DiT 从头训，不需要沿用 checkpoint 的 stats（E1w / E2w 用的是 ckpt 的 stats） |
| 增强 | 关闭 |

**和官方评测集的隔离**（10-24 已核对）：

RoboMME 在 `robomme_sim/robomme/env_metadata/{train,val,test}/` 里固定了三份 episode 列表，每个任务分别是 100 / 50 / 50 条，靠环境 seed 区分。

- **训练数据集就是官方 train 集**：`robomme_lerobot` 的 1600 条 episode，每个任务正好 100 条，编号 0–99，和官方 train 列表逐个对应。
- **三份列表的 seed 互不重叠**：16 个任务里，train 和 test、train 和 val、val 和 test 的 seed 交集都是 0。以 PickXtimes 为例，train 的 seed 在 1000–10900，val 在 1010000–1014900，test 在 510000–514900。
- 因此本实验的 train（360 条）和 val（40 条）**都是从官方 train 集里切出来的**。这里的 "val" 只是 train 集内部留出的一部分，不是官方 val。官方 val 和 test 列表没有用于训练，也没有用于开环评测。
- 以后做闭环评测时，用 `eval_robomme.sh` 默认的 `--dataset_split test`，也就是官方 test 集，每个任务 50 条。官方 test 集只能用于最终评测，不能用来训练或调参。

**发布版 ckpt 的训练数据（存疑）**：

依据是 `checkpoints/simplememvla_robomme/stats.json`。`sft_trainer.py::_save` 会把训练时数据集的 `<root>/meta/stats.json` 原样复制进 checkpoint，它可能是更大数据池的全局统计，不等同于实际采样episode清单，子集训练也可能复制全局stats。和公开数据集对比如下：

| 字段 | 发布版 ckpt | 公开 `robomme_lerobot`（= 官方 train） |
|---|---|---|
| `episode_index` min / max / mean | 0 / **3199** / 1660 | 0 / 1599 / 860 |
| 帧数（各字段的 `count`） | **1,535,626** | 768,897 |
| `task_index` max（非唯一种类计数） | **133** | 115 |
| `subtask_online_index` max（非唯一种类计数） | **6146** | 5174 |
| `frame_index` max | 1620 | 1410 |
| `is_video_demo` mean | 0.384 | 0.380 |

读法（2026-10-25 勘误）：
- `episode_index` 最小0、最大3199，不能单凭这两个值推出编号连续或3200个唯一episode；`count=1,535,626` 是统计覆盖的原始帧数，不是梯度更新样本计数。
- 该统计与公开train的768,897帧、编号上界1599不同，且编号上界与train+val+test合计3200相符，是需要解释的来源疑点；无法仅凭此确定额外数据的seed或split。
- `task_index`/subtask编号上界更大，不足以证明确切新增类别数或“全部为不同seed”。
- 保存代码直接复制根目录全局stats，并不按训练episodes重算；可能发生“训练子集、保存全局stats”。
- 新seed、开环误差和sub-task准确率比较最多提供间接线索；确认评测数据参与训练需要实际训练task/seed清单或可核对的数据来源。

这个问题的影响：发布权重的数据来源尚未充分追溯，评测污染风险不能排除，也未被证实；不能据此认定作弊。旧主实验E1/E2/E3采用原生Qwen并仅在官方train子集上后训练，以避开发版权重的来源不确定性。下一轮完整发布权重后训练应标明其诊断／工程对照角色。

### 1.2 模型

| 项 | 值 |
|---|---|
| VLM | **原版 Qwen3.5-4B**（`Qwen/Qwen3.5-4B`，rev `851bf6e`，放在 `/mnt/cpfs/heping/checkpoints/Qwen3.5-4B`），没有经过任何机器人数据微调，全冻结，4539M 参数。**不加载任何 SimpleMemVLA checkpoint**（`init_from_checkpoint: null`） |
| VLM 输入 | 单帧：front 和 wrist 的当前帧，都作为 image（每张 64 个 token），没有历史视频 |
| DiT 条件 | `condition_source=prompt`：冻结 VLM 对整段 prompt 所有有效 token 的 last hidden 和 token embedding，约 219 个 token；不生成 sub-task；`vl_loss_weight=0` |
| DiT | **从头初始化**，结构和发布版一样（16 层，hidden 2048，16 头，mlp×3.5），1169M 参数；带 proprio state token；动作块 30×8 |
| Memory（E2/E3） | 原版 Qwen ViT 的 merger 特征 cache（`features/robomme_vit_merger_front_qwen35base`，**ViT 用 fp32 抽取**，原因见 §1.5），2 fps，60 s，每帧 4×4 = 16 个 token；4 层 TF，d=1024，Perceiver 16 query × 2 层，125.7M 参数，随机初始化 |
| Loss | flow matching，τ ~ Beta(1.5, 1) |
| 结构图 | `memcache_architecture.svg` |

**机器人后训练数据边界**：旧主E1/E2/E3初始化来自原生Qwen或随机参数，梯度更新只用官方train内的360条，内部val40用于评估。归一化仍使用官方train全部1600条的全局stats（含内部val40）；这不等于把这些episode都送入优化器。官方val/test轨迹未用于本轮梯度训练。

`E*_ckptvlm` 是对照组，冻结的是发布版的 VLM（可能见过所有 RoboMME 数据），只用来衡量机器人数据微调过的 VLM 带来多少提升，不作为主结果。

### 1.3 优化（相对 E1w / E2w 的改动加粗）

| 项 | 值 |
|---|---|
| batch | 每卡 16 × 卡数 × `GRAD_ACCUM` = global batch 64（4 卡时 accum=1，2 卡时 accum=2） |
| 步数 | **30000 步**（约 17 个 epoch），**warmup 1000** |
| LR | **action_head 5e-5**（发布版从头训 DiT 时用的 lr）；memory 1e-4；backbone 0；cosine 衰减到 0.1× |
| 其他 | AdamW（β2=0.95），weight decay 1e-8，梯度裁剪 1.0，seed 429，DeepSpeed ZeRO-2，bf16 |
| 保存 | **每 5000 步**存一次，保留最近 2 个（每个约 45 GB） |

### 1.4 评测协议

- 每 1000 步在 val 上评一次：每卡 50 个 batch × 16，固定的采样顺序。
- 每个 batch 的 noise 和 t 固定，三种模式之间只有 memory 输入不同：
  - `val`：真实 memory；
  - `zero`：m 置 0；
  - `shuffle`：batch 内循环移位，换成别的样本的 memory。
- 诊断量：
  - `gapZ = loss(zero) − loss(val)`；
  - `gapS = loss(shuffle) − loss(val)`：**模型是否依赖 memory 的内容**，最关键的指标；
  - `mem_norm`：训练中 m 的平均范数。
- 训练结束后用 `scripts/diag_memory.py` 检查 Z 和 m 在样本之间的离散程度（方法见 E2w-diag）。
- 目前只有开环 flow loss，还没有闭环成功率，也还没有单独统计记忆关键帧子集。

---

### 1.5 Memory cache 的数值精度（10-24）

- **现象**：用原版 Qwen 的 ViT 抽取的 cache，和模型里实时算出的同一帧特征对比，cos 最小值只有 0.9938（发布版 ViT 是 0.9998），自检没有通过。
- **排查过程**：
  1. 抽取脚本和训练模型里的 ViT 参数、`inv_freq` 都**逐位相同**，同一个 batch 的输出也完全一致，所以不是用错了模型。
  2. 差异来自 **bf16 下 batch 形状不同**：同一帧单独编码和放在 64 帧的 batch 里编码，原版 Qwen ViT 的结果 cos 平均 0.99906，**最小 0.706**，相对误差 4.0%；发布版 ViT 是 0.99995 / 0.9992 / 1.1%。原版 ViT 有少数 token 对 bf16 kernel 的选择特别敏感。
  3. 作为参照：相邻关键帧（间隔 0.5 s）之间的 cos 平均是 0.90。所以这个噪声平均下来比帧间差异小约 100 倍，但个别 token 会差得很多。
  4. **fp32 下这个问题完全消失**：batch 1 和 batch 64 的结果 cos = 1.00000。bf16 和 fp32 之间，个别 token 的 cos 最低到 0.68。
- **处理**：
  - cache 一律用 **fp32 的 ViT** 抽取（`extract_mem_features.py --vit_dtype float32`，现在是默认值，会记进 `meta.json` 的 `vit_dtype` 字段），结果和 batch 形状无关。以后闭环 policy 在线编码关键帧时也必须用 fp32。
  - 当前帧经过 VLM 的路径保持 bf16 不变。
  - 自检脚本改成比较 cos 的**均值**（> 0.995），因为 fp32 的 cache 和 bf16 的实时特征不可能逐 token 一致。
- **日志**：`outputs/memcache/vit_compare.py`、`vit_batch_noise.log`、`vit_fp32_noise.log`。

## 2. 实验总表

| ID | 日期 | 相对参考的改动 | 状态 | best val（step） | final val | gapZ（最后） | gapS（最后） | 结论 |
|---|---|---|---|---|---|---|---|---|
| **E1** | 10-24 | 参考组：单帧，无 memory；**冻结原版 Qwen3.5-4B**，DiT 从头训 | **已完成**（10-25 05:40，30k 步）；10k 步闭环：test **1.0%**，训练用过的 episode **0/40**；30k 还没做闭环 | **0.0473（20k）** | 0.0487 | — | — | 开环在约 15k 步后进入平台期（0.048–0.050） |
| **E2** | 10-24 | E1 + memory cache（`adaln_add`，cache 用原版 Qwen 的 fp32 ViT 抽取） | **第 20k 步崩溃**（10-25 02:19，DataLoader abort），停在 `checkpoint-15000`，没有续训；10k 步闭环：test **1.5%**，训练用过的 episode **0/40** | **0.0381（18k）** | —（19k 时 0.0400） | — | **+0.179（19k）** | ❓ 开环：memory 的内容被用上了（gapS ≫ 0）；闭环：两组都接近 0，分不出差别 |
| E3 | 10-24 | E2，融合方式改为 `prefix_tokens` | 配置就绪，未排期 | — | — | — | — | — |
| E1-ckptvlm / E2-ckptvlm | 10-24 | 同 E1 / E2，但冻结发布版 VLM（只加载 `backbone.*`） | 配置就绪，可选对照 | — | — | — | — | — |
| ~~E1w~~ | 10-24 | 旧 E1：**DiT 也热启动** | ⚠️ 作废；在 13965/15000 步被外部杀掉 | 0.03215（13k） | — | — | — | 泄漏，结论不可信 |
| ~~E2w~~ | 10-24 | 旧 E2：DiT 也热启动，+ memory（`adaln_add`） | ⚠️ 作废；跑完 | 0.03168（13k） | 0.03288 | +1.1e-4 | 0 | 泄漏，结论不可信 |
| E2w-diag | 10-24 | 不训练：诊断 E2w 的 memory 路径 | 完成 | — | 0.03627 | +1.5e-4 | 0 | 诊断方法和工具可以复用；"memory 被训练成常数"这个现象需要在新 E2 上重新验证 |

---

## 3. 详细记录

### E1：单帧、无 memory、DiT 从头训（下界）

- **目的**：在不泄漏的设定下给出下界：冻结的原版 Qwen3.5-4B 只看当前帧，DiT 从头训。
- **配置**：`configs/train/memcache/E1.yaml`，即 §1 的共享设置加 `use_memory: false`。
- **命令**：
  ```bash
  cd /mnt/cpfs/heping/SimpleMemVLA
  bash scripts/launch_memcache.sh E1 0,1,2,3          # 4 卡，GRAD_ACCUM=1
  DRY_RUN=1 bash scripts/launch_memcache.sh E1 0,1,2,3  # 只看完整命令
  ```
- **产物**：输出 `SimpleMemVLA-data/outputs/memcache/robomme_small_E1/`；log `outputs/memcache/run_E1_<时间戳>.log`。
- **启动前验证**（10-24）：
  - **YAML 解析**：`E1`、`E2`、`E3`、`E*_ckptvlm`、`E*_warmdit` 都能解析。E1/E2/E3 的 backbone 是原版 Qwen3.5-4B，`init_from_checkpoint=None`，用数据集的 stats，30k 步，lr 5e-5。
  - **加载检查（原版 Qwen）**：`scripts/sanity_memcache.py --backbone /mnt/cpfs/heping/checkpoints/Qwen3.5-4B` **全部通过**，日志 `outputs/memcache/sanity_qwen35base.log`。
    - backbone 的参数和原版 Qwen3.5-4B 逐位相同；DiT 和 memory 是新初始化的；backbone 冻结（可训练参数：DiT 1169M，memory 126M）。
    - memory cache 和当前 VLM 来自同一个模型（`meta.json` 的 checkpoint 字段核对通过）。fp32 的 cache 和 bf16 的实时特征 cos 均值 0.998（最小 0.965）。
    - 零初始化时有无 memory 的输出逐位相同；backbone 没有梯度。
    - 初始 loss 1.8613，和 `*_ckptvlm` 路径**完全相同**。这是正常的：DiT 的 final layer 零初始化，第一步输出恒为 0，loss = ‖ε − a‖²，和条件信息无关。
    - 用 bf16 cache 跑的那次失败记录在 `sanity_qwen35base_bf16cache.log`（cos 最小值 0.9938，原因见 §1.5）。
  - **加载检查（发布版 VLM 加新 DiT，即 `*_ckptvlm` 的路径）**：`--init_modules backbone` 全部通过。backbone 和 ckpt 逐位相同，DiT 和 ckpt 不同；初始 loss 1.86（DiT 热启动时是 0.89）；日志 `outputs/memcache/sanity_freshdit.log`。
- **状态**：等 review 后再启动。

### E2：单帧 + memory cache（`adaln_add`），DiT 从头训

- **目的**：主方法，在公平设定下看历史 cache 经 AdaLN 注入后能不能降低 loss。
- **配置**：`configs/train/memcache/E2.yaml`。相对 E1 只有三处改动：`use_memory: true`、`mem_fusion: adaln_add`、`mem_feature_dir: features/robomme_vit_merger_front_qwen35base`（原版 Qwen 的 fp32 ViT，17,964 个关键帧；`train.py` 会检查这份 cache 和当前 VLM 是否来自同一个模型）。
- **命令**：
  ```bash
  bash scripts/launch_memcache.sh E2 4,5,6,7
  ```
- **产物**：输出 `outputs/memcache/robomme_small_E2/`。
- **状态**：等 review 后再启动。
- **训练中要看的**：
  - `gapS` 是否从 0 开始变成正数；
  - `mem_norm` 有没有涨起来；
  - 和 E1 的 val 曲线对比。
  - 跑完用 `diag_memory.py` 检查 Z 和 m 在样本之间的离散程度，判断是否又塌缩了。

### 闭环评测：RoboMME 官方 test 集（10-24 新增）

**评测代码**：仓库原有的 `robomme_sim/eval_success.py` 只支持发布版那套做法：历史视频输入 → 生成 sub-task → DiT。E1/E2 用的是单帧 prompt 条件，E2 还有 memory cache，原代码跑不了。新增内容：

- `robomme_sim/memcache_policy.py`：
  - `MemCacheBuffer`：每个环境一份。按 dataset 的 `frame_index` 计数，**RoboMME reset 时回放的 demo 帧也计入**。front 相机每 10 帧存一个关键帧。
  - `MemCacheBatchedPolicy`：一组环境一起做决策。关键帧用和抽取 cache **同一个 fp32 ViT** 在线编码，存成 fp16，再按训练时同样的方式 pool 和选取窗口。prompt 和训练完全一致，然后调用 `predict_action`。
- `eval_success.build_policy`：checkpoint 的 `condition_source == "prompt"` 时自动切到上面这套 policy，其它 checkpoint 仍走原来的流程。暂不支持 `--pipelined`。
- 启动脚本 `outputs/memcache/eval/run_eval_memcache.sh <EXP> <step> <gpus> <tag> [flags]`：固定用 `--dataset_split test`，只评测训练用到的 4 个任务。

**训练与评测输入一致性检查**（`outputs/memcache/verify_eval_inputs.py`，用 E2 的 10k checkpoint，**全部通过**）：

- **帧对齐**：在官方 train 集的 episode 上 reset 仿真器，这些 episode 和数据集里的是同一批 seed。reset 之后当前帧的编号等于数据集里第一个执行帧的 `frame_index`：VideoUnmaskSwap 的 demo 有 168 帧，对应 168；PickXtimes 没有 demo，对应 0。任务指令一致。
- **画面差异**：仿真画面和数据集画面的像素平均绝对误差约 4.3/255，来自数据集 mp4 的压缩。发布版 baseline 也有同样的差异。
- **prompt**：在 val 的 6 个锚点上，token id 和 pixel_values 都和训练时的 collator **逐位相同**。
- **memory**：同样 6 个锚点上，关键帧数 K 和帧龄完全一致，特征 cos = 1.000000，相对误差 ≤ 5e-4（fp16 舍入）。

**冒烟测试**：E2 的 10k checkpoint，VideoUnmaskSwap 的 2 个 test episode，0/2 成功，用时 9 分钟。

- **显存**：评测进程和训练共用一张卡，评测本身约 14 GB，卡上峰值 36 GB。
- **录像**：机械臂能看到被遮住的方块，会去抓起杯子、移动杯子。动作是合理的，只是没完成任务，直到步数上限。

**正式评测**：2026-10-24 21:45 启动，4 个任务 × 每个任务 50 个 test episode，每组 6 个环境并行。

| 实验 | checkpoint | GPU | 输出目录 | 结果 |
|---|---|---|---|---|
| E1 | `robomme_small_E1/checkpoint-10000` | 0–3（和 E1 训练共用） | `outputs/memcache/eval/E1_step10000_test4task/` | 完成（23:24），见下表 |
| E2 | `robomme_small_E2/checkpoint-10000` | 4–7（和 E2 训练共用） | `outputs/memcache/eval/E2_step10000_test4task/` | 完成（23:25），见下表 |

**结果**（官方 test 集，每个任务 n=50；每张卡的峰值显存 34.5–43.3 GB，训练没有受影响）：

| 任务 | E1 @10k | E2 @10k |
|---|---|---|
| PickXtimes | 0% | 0% |
| VideoUnmaskSwap | 0% | 6%（3/50） |
| VideoRepick | 0% | 0% |
| PatternLock | 4%（2/50） | 0% |
| **平均** | **1.0%** | **1.5%** |

**解读**：
- 两组都接近 0，差别只有 1–3 个 episode，是噪声，**这次闭环评测分不出 E1 和 E2 的好坏**。
- PickXtimes 连第一次抓放都完成不了，而这一步根本不需要记忆。看 E2 失败的录像（`PickXtimes__ep0__fail.mp4`，只有 94 帧）：任务是"抓方块放到目标上，重复 N 次，再按按钮结束"，机械臂开局**直接去按按钮**，episode 立刻判失败。
- 所以主要问题是**任务阶段判断错了**，对指令和场景的理解不够，不是 memory 机制的问题。
- **可能的原因**（还没验证）：
  1. 只有 10k 步，DiT 是从头训的；
  2. 冻结的原版 Qwen 没见过机器人数据，单帧特征对"该做哪一步"区分不够；
  3. 每个任务只有 90 条训练轨迹；
  4. 开环 loss 低不代表闭环误差不会一步步累积。

**建议的下一步诊断**：
- **在官方 train 集上做闭环**：这些 episode 训练时见过。成功率仍然接近 0，说明是拟合 / 闭环能力的问题；明显更高，说明是泛化问题。
- **用发布版 ckpt 在同样 4 个任务上跑一遍**：检验评测流程本身没问题。发布版可能见过 test，但这里只用来确认评测代码能跑出正常分数。
- **等 30k 步的 checkpoint 再评测**。

**两点说明**：
- 两个模型都只用这 4 个任务的 360 条 episode 训练，所以只评测这 4 个任务，不能和论文里 16 个任务的平均分直接比。
- 10k 步时的开环 val loss：E1 为 0.0560，E2 为 0.0422。E2 的 `gapS` = +0.117（打乱 memory 后 loss 从 0.042 升到 0.159），这次 memory 确实被用上了。

### 闭环诊断：在训练用过的 episode 上评测（10-24 / 10-25）

**目的**：把"拟合不够"和"泛化不够"两种原因区分开。

**设置**：
- 官方 train 集上，每个任务取 10 个**本实验训练集里用过的** episode，跳过我们留作 val 的那几个：
  - PickXtimes 和 VideoRepick：0–10 里分别去掉 5 和 9；
  - 其余两个任务：0–9。
- 给 `eval_success.py` 新增了 `--episode_ids_json`，可以直接指定要评测的 episode。episode 列表在 `outputs/memcache/eval/train_split_trained10.json`。

**结果**：E1 和 E2 的 10k checkpoint 都是 **0/40**（4 个任务全部 0%）。所以失败不是泛化问题，模型**连训练时见过的 episode 都做不成**。

**进一步排查**（PickXtimes train ep0 = 数据集 ep 700，E2 的 10k checkpoint；脚本 `outputs/memcache/eval/diag_actions.py` 和 `diag_closedloop.py`）：

1. **评测流程没问题**：
   - 在仿真里回放数据集记录的真实动作，第 490 步**成功**；
   - 仿真状态和数据集里的状态平均只差 0.013 rad，动作、状态、夹爪的约定都一致。
2. **开环预测还算准**：用数据集的画面和状态做输入，前 16 步预测的关节误差平均 0.008–0.019 rad，多数时刻比"关节保持不动"的基线好 2–7 倍，夹爪开合指令也基本正确。
   - 但这段时间里关节本身只动 0.01–0.07 rad，误差占了运动量的 20–100%；在 t=264 甚至比保持不动还差。
3. **视频压缩带来的画面差异可以忽略**：
   - 同样的状态下，用仿真原图和数据集画面输入，预测只差 0.002 rad；
   - 把仿真画面先压成 mp4 再解码送进模型，结果和用原图完全一样，同样失败。
4. **闭环很快偏离**：
   - 和 demo 轨迹相比，关节最大偏差在 t=16 时是 0.04 rad，t=32 时 0.24，t=48 时 0.37；
   - 机械臂在 t≈112 合上夹爪，和 demo 的时机相近，但位置偏了约 0.4 rad，夹了个空（夹爪宽度 0.000）；
   - 之后反复开合，直到 1300 步超时。

**结论**：
- 评测代码和输入预处理都经过验证，没有问题。
- 10k 步时，"冻结原版 Qwen + 从头训练 DiT + 每个任务 90 条轨迹"这套配方的开环精度不够，抓小方块这种操作在闭环里误差会被迅速放大。所以 E1 和 E2 在闭环里都是 0%。
- 这是基础控制能力的问题，**不能拿来判断 memory 有没有用**。

**下一步可以做的**：
- 用 `--execute_horizon 8` 更频繁地重新规划；
- 等 30k 步的 checkpoint 再评测；
- 修改配方：VLM 用 LoRA 微调或部分解冻、用更多数据或全部 16 个任务、换更小的 DiT 减少过拟合。

### 用发布版 ckpt 验证评测流程（10-25）

**设置**：
- 完整的 `simplememvla_robomme` 走**原版**流程：历史视频 → 生成 sub-task → DiT。
- 按官方协议评测：`execute_horizon 16`、`max_steps 1300`、10 步去噪。唯一的改动是 `group_size` 用 5，因为和训练共用 GPU；这只影响并行度，不影响单个 episode 的结果。
- 评测的 episode 和 E1/E2 完全相同。
- 脚本 `outputs/memcache/eval/run_eval_released.sh`，输出在 `outputs/memcache/eval/released_{test4task,train10}/`。

| 任务 | 发布版 · 训练用过的 episode（n=10） | 发布版 · test（n=50） | E1 @10k · test | E2 @10k · test |
|---|---|---|---|---|
| PickXtimes | 100% | **100%** | 0% | 0% |
| VideoUnmaskSwap | 100% | **90%** | 0% | 6% |
| VideoRepick | 100% | **60%** | 0% | 0% |
| PatternLock | 100% | **94%** | 4% | 0% |
| **平均** | **100%** | **86.0%** | 1.0% | 1.5% |

**结论**：
- **评测流程没有问题**：同一套评测、同样的 episode，发布版能拿到 86–100%。E1/E2 接近 0 是我们模型本身的问题。这和前面"回放真实动作能成功、输入和训练逐位一致"的结论吻合。
- **关于发布版有没有用 test 训练**：train40的100%和test200的86%既不能证明也不能排除测试污染；它们不是成员清单。全局stats来源仍需追溯，§4的LEAK-check只能提供间接线索，不能“彻底确认”。

**意外：E2 训练在第 20000 步崩溃**（10-25 10:10 核实：E1 已经跑完 30k；E2 还停着；GPU 全部被别人占用；`/mnt/cpfs` 只剩 386 GB）：
- 第 20000 步 eval 的时候，一个 DataLoader worker 报 `LLVM ERROR: pthread_join failed` 后 abort，rank0 抛出 `RuntimeError: DataLoader worker ... killed by signal: Aborted`，02:19 退出（`TRAIN_EXIT=1`）。
- `checkpoint-20000` 没来得及保存，最近的是 `checkpoint-15000`。E1 不受影响。
- 推测和当时机器负载太高有关（训练 + 8 个评测进程 + 大量 dataloader worker 同时在跑），无法确认。
- 恢复：`bash scripts/launch_memcache.sh E2 4,5,6,7`（`RESUME=auto` 会从 15000 接着训，必须仍用 4 卡）。

---

### ⚠️ 作废：E1w / E2w（DiT 热启动，10-24）

两组和新 E1 / E2 的区别：
- `init_modules: [backbone, action_head]`；
- 用 ckpt 的 stats；
- 15000 步，warmup 500，action_head lr 3e-5，每 2500 步保存一次；
- 用 2 卡、`GRAD_ACCUM=2` 跑。

配置可以用 `configs/train/memcache/E1_warmdit.yaml` / `E2_warmdit.yaml` 复现。

**为什么不作旧主结果**：E1w/E2w热启动了发布DiT，与当时要评估的随机动作头设定不同；发布权重与本地train/val重叠风险也未核实。两组早期val loss约0.057和memory塌缩是实测现象，但不能仅由stats断言“完整3200条和test都训练过”。历史组继续保留作记录，不恢复或覆盖其输出。

**产物**：先改名，避免新 run 用 `RESUME=auto` 时误续训这些 checkpoint；**10-25 经用户同意删除了两个输出目录（释放 192 GB）**。下表中的日志和 wandb 都还在，小的元数据文件（`trainer_state.json`、`config.json`、`training_args.bin`、`stats.json`）保存在 `outputs/memcache/warmdit_meta/`。

| 组 | 输出目录 | 日志 | wandb（离线） |
|---|---|---|---|
| E1w | `outputs/memcache/robomme_small_E1_warmdit/`（`checkpoint-10000`、`checkpoint-12500`） | `run_E1_warmdit.log` | `offline-run-20261024_004748-8l59dtnx` |
| E2w | `outputs/memcache/robomme_small_E2_warmdit/`（最终模型、`checkpoint-12500`、`checkpoint-15000`） | `run_E2_warmdit.log` | `offline-run-20261024_004748-edv6qofr` |

**运行情况**：
- E1w 平均 2.17 s/step，09:33 在第 13965 步被外部杀掉：没有 traceback，也没有 OOM 记录。
- E2w 用时 10.6 h，跑完，2.55 s/step。

**val loss 曲线**（只作记录）：

| step | 1k | 3k | 5k | 7k | 9k | 11k | 13k | 15k |
|---|---|---|---|---|---|---|---|---|
| E1w | .0573 | .0415 | .0387 | .0351 | .0355 | .0327 | .0322 | — |
| E2w | .0587 | .0423 | .0382 | .0368 | .0353 | .0323 | .0317 | .0329 |
| E2w gapS | ~0 | 0 | 0 | 0 | ~0 | ~0 | 0 | 0 |

### E2w-diag：memory 路径诊断（不训练，10-24）

- **方法**：
  1. `scripts/diag_memory.py`：用 512 个 val 样本，检查 shuffle 是否真的改变了输入；统计 memory 输入、Z、m 在样本之间的离散程度；看 zero 和 shuffle 对 velocity 和 loss 的影响；比较 |m| 和 |t_emb| 的大小。
  2. `outputs/memcache/diag_init_vs_trained.py`：在同一批输入上，对比新初始化和训练后两种权重下，每一级输出的离散程度和 cross-attn 熵。
- **产物**：`outputs/memcache/diag_E2_warmdit.{json,log,sh}`、`diag_init_vs_trained.log`。
- **结果**：
  - **实现没有问题**：shuffle 改变了 512/512 个样本的输入；输入本身在样本之间差异很大（rel_dev 0.42）。
  - **输出与输入无关**：Z 的 rel_dev 只有 0.0015，m 只有 1.9e-7；|m| 和输出层 bias 的范数分别是 0.08569 和 0.08568，m 基本就是 bias；shuffle 后 velocity 逐位不变。
  - **塌缩是训练造成的，不是结构问题**：Z 的 rel_dev 在新初始化时是 0.24，训练后变成 0.0016，主要在 encoder 里被压掉；Perceiver 的注意力始终接近均匀（熵约 1.0）。
- **结论**：
  - 诊断工具可靠，可以直接用于新 E2。
  - E2w的memory内容塌缩已被诊断；这与发布DiT先验、GT条件捷径等因素相容，但未证明唯一原因是评测数据泄漏。旧主E2已显示正的shuffle gap，详见后续曲线。

---

## 4. 待做（按优先级）

| ID | 改动 | 要回答的问题 | 状态 |
|---|---|---|---|
| 旧主 E1、E2 | 原生VLM冻结，DiT从头训 | 官方train内后训练的memory效果 | E1完成30k；E2最近保存15k，20k崩溃；非下一轮full-pretrain |
| E2-diag | 跑完 E2 后用 `diag_memory.py` 检查 | 公平设定下 memory 还会不会塌缩成常数 | 依赖 E2 |
| E2-need | 不训练：分析 E1 在 val 上的误差按帧怎么分布，找出必须靠历史才能确定动作的帧 | 这个 loss 里"需要记忆"的部分有多大 | 依赖 E1 |
| E2-aux | memory 加辅助监督：用 Z 预测当前 GT sub-task（`subtask_online_index`） | 给 memory 直接的梯度信号，它能不能学到有用的历史 | 待做（如果 E2 也塌缩，优先做这个） |
| E3 | `mem_fusion=prefix_tokens` | AdaLN 全局向量是不是瓶颈 | 配置就绪 |
| ~~E1q、E2q~~ | VLM 换成原版 Qwen3.5-4B | 排除 VLM 层面的泄漏 | ✅ 已经并入主线 E1/E2/E3 |
| E1/E2-ckptvlm | 冻结的 VLM 换成发布版 | 机器人数据微调过的 VLM 带来多少提升 | 配置就绪，可选 |
| LEAK-check | 不训练：用 RoboMME 的 planner 在官方 test seed、官方 val seed 和全新 seed 上各生成一批 demo，对比发布版 ckpt 的开环误差和 sub-task 准确率 | 提供发布版数据成员关系的间接线索，不能单独确认（见§1.1） | 待做 |
| E2s | 更小的 DiT（例如 8 层 / 1024），或更少步数 | 小数据上 1.17B 的 DiT 是否太容易过拟合 | 第二轮 |
| E2p、E2q'、E2w' | 每帧 token 数 64 / 4；query 数 4 / 64；窗口 30 / 120 s | 压缩率、定长长度、历史长度 | 第二轮 |
| 旧主 E1c、E2c | 闭环评测（4任务×50） | 开环loss和成功率关系 | policy已实现，10k已评测；后续checkpoint另行决定 |


## 2026-10-25：发布版 VLM → 原生 Qwen3.5-4B 的零训练 A/B（已完成，严格异常检查）

- 用户授权：先做最简单 A/B，不增加训练；GPU 已释放。
- 输出根目录：`/mnt/cpfs/heping/SimpleMemVLA-data/outputs/memcache/eval/vlm_swap_ab_20261025_104520`。
- A：发布版 VLM + 发布版 DiT；GPU 0,1,2,3；launcher PID 652054。
- B：原生 `/mnt/cpfs/heping/checkpoints/Qwen3.5-4B` 整块 backbone + 同一个发布版 DiT；GPU 4,5,6,7；launcher PID 652055。
- 两组均重新评测：先 `train_split_trained10.json` 指定的每任务 10 条（共 40 条），通过完整性检查后自动跑四任务 test 每任务 50 条（共 200 条）。
- 固定：发布版 processor/chat template/stats、历史视频→生成 sub-task→DiT；group_size=5，execute_horizon=16，max_steps=1300，去噪10步，temperature=1，subtask上限64，bf16，FA2，非pipelined。每个worker在模型加载前、后均设torch/NumPy seed=0；闭环分岔后不保证逐决策噪声逐位配对。
- 入口：`scripts/run_vlm_swap_ab.sh` + `configs/eval/robomme_vlm_swap_ab.yaml`；`eval_success.py --backbone_override` 仅用于原版sub-task policy。`policy.get_vla` 用HF加载器处理tied lm_head/MTP，校验加载键/尺寸/dtype/RoPE fp32；原有action_head和归一化不变。
- 留档：各组 manifest.json、code_diff.patch、eval_config.yaml、train_episode_ids.json；各阶段 eval_args.json、eval.log、results.json 和成功/失败录像。status.txt 记录阶段与退出码；结果有效样本不足或NA/OOM会使wrapper失败并停止下一阶段。
- 2026-10-25 10:45 CST 已后台启动，两组各4个worker进入模型加载。尚无新成功率，不能将启动当作实验完成。

- 启动前复核修正：旧评测会将pool.step异常或status=error/error_message计为普通失败；A/B配置启用fail_on_env_error，严格模式直接使任务无效并让wrapper停止。首轮10:41作业在CPU初始化阶段中止，无评测结果，旧目录有CANCELLED.txt；上方为10:45重启后的正式目录。

- 10:49 CST 运行检查：B的4个worker均已打印原生backbone替换成功；键/尺寸/dtype/RoPE检查通过。A/B八张卡已进入仿真运行，显存约15–16GB/卡，尚无完整train40结果。


### A/B 最终结果（2026-10-25 12:18 CST 核实）

| 任务 | A train（各10条） | B train | A test（各50条） | B test | test下降（百分点） |
|---|---:|---:|---:|---:|---:|
| PickXtimes | 100% | 0% | 100% | 0% | 100 |
| VideoUnmaskSwap | 100% | 0% | 90% | 0% | 90 |
| VideoRepick | 100% | 0% | 60% | 0% | 60 |
| PatternLock | 100% | 0% | 94% | 0% | 94 |
| 总体 | 100%（40/40） | 0%（0/40） | 86%（172/200） | 0%（0/200） | 86 |

训练子集下降100个百分点，test下降86个百分点。A于11:17:29、B于12:05:35完成，均exit=0；每组train40/test200有效样本齐全、无NA/OOM，严格仿真异常检查未报错。B出现subtask超64token预算告警：train34次、test55次（告警次数，不是episode数）。GPU均已空闲。结果表示发布版DiT高度依赖微调VLM的生成及表征适配，不能单独证明冻结原生VLM重新训练DiT不可行。汇总JSON：`/mnt/cpfs/heping/SimpleMemVLA-data/outputs/memcache/eval/vlm_swap_ab_20261025_104520/comparison.json`。


## 2026-10-25 已完成：发布权重、单帧输入与 latency 对照

### 范围、命名与论文复现边界

本节的 E1/E2 仍指旧主实验：原生 Qwen3.5-4B 全冻结、随机 DiT、prompt 条件；用户拟议的“完整发布权重后训练”是下一轮实验，尚未启动，不能混用旧编号、配置和输出目录。

评测只覆盖 PickXtimes、VideoUnmaskSwap、VideoRepick、PatternLock 四任务。train40 是训练 episode 中每任务抽 10 条做闭环评测；它不同于内部留出的 val40。test200 是官方 test 每任务 50 条。此轮没有完整评测全部 16 任务，不能写成“完整复现论文 88.3%”。

用户提供论文 `paper.md` 的 Table 10 中，这四任务分别为 100%、90%、64%、94%，子集平均 87%；本机是 100%、90%、60%、94%，子集平均 86%。三项一致，VideoRepick 低 4 个百分点（2/50 条），宏平均低 1 个百分点。可表述为“四任务子集与论文接近，发布权重的闭环链路已跑通”。

### 三组零训练消融的成功率

| 配置 | train40 | test200 | 相对完整发布版视频策略下降 |
|---|---:|---:|---|
| 完整发布版 VLM + 发布版 DiT，历史视频 | 100%（40/40） | 86%（172/200） | 基准 |
| 原生 Qwen3.5-4B 替换整个 VLM，保留发布版 DiT，历史视频 | 0%（0/40） | 0%（0/200） | train 100 / test 86 个百分点 |
| 完整发布版 VLM + 发布版 DiT，仅当前帧，不后训练 | 10%（4/40） | 2%（4/200） | train 90 / test 84 个百分点 |

| 任务 | 视频 train | 单帧 train | 视频 test | 单帧 test |
|---|---:|---:|---:|---:|
| PickXtimes | 100%（10/10） | 0%（0/10） | 100%（50/50） | 0%（0/50） |
| VideoUnmaskSwap | 100%（10/10） | 30%（3/10） | 90%（45/50） | 4%（2/50） |
| VideoRepick | 100%（10/10） | 0%（0/10） | 60%（30/50） | 0%（0/50） |
| PatternLock | 100%（10/10） | 10%（1/10） | 94%（47/50） | 4%（2/50） |

两轮均正常结束、有效样本齐全、无 OOM/NA 或严格仿真异常。视频基准来自已完成的 A/B A 组；单帧实验复制其结果并注明复用来源，没有伪称重新测了视频成功率。

“仅当前帧”在本轮的精确定义：每个相机只提供当前一个独立时刻；front 复制成 `[current,current]` 以满足视频编码器的 temporal patch size=2，wrist 为当前 image。保留发布 processor、prompt、生成 sub-task → hidden states/token embeddings → DiT，不是旧 E1/E2 的双 image + prompt 条件。

统一闭环参数：group_size=5、execute_horizon=16、max_steps=1300、去噪10步、temperature=1、sub-task 上限64、bf16、FA2、pipelined=false。VLM 替换组改变的是整个 backbone；单帧组未改变任何权重。

### 推理延迟（已完成）

同一发布模型、batch=1、真实连续观测与 state 配对；视频自然缓存策略控制共同轨迹。每个 split 每任务 2 条 episode，每条最多 12 个决策点；提前成功时自然终止。实际 train 75 对、test 79 对；两模式×两缓存条件分别产生 300/316 条有效计时，无 invalid。每模式预热5次，CUDA同步计时；包括预处理、sub-task生成、DiT采样及动作转换，不包括模型加载、仿真reset/step、预热及结果写入。

| 数据／缓存 | 视频 mean / P50 / P95（ms） | 单帧 mean / P50 / P95（ms） | 均值耗时比（视频÷单帧） | P50耗时比 |
|---|---|---|---:|---:|
| train／自然缓存 | 800.9 / 797.7 / 924.3 | 537.9 / 501.9 / 632.3 | 1.489× | 1.589× |
| test／自然缓存 | 777.2 / 784.2 / 893.9 | 663.4 / 490.1 / 597.2 | 1.171× | 1.600× |
| train／冷视觉缓存 | 861.7 / 862.9 / 1001.1 | 482.1 / 478.0 / 571.0 | 1.787× | 1.805× |
| test／冷视觉缓存 | 840.0 / 854.1 / 938.8 | 474.5 / 492.1 / 582.3 | 1.770× | 1.736× |

自然缓存保留连续决策的视觉／prompt缓存；冷视觉缓存只清视觉特征和prompt缓存，不清模型/GPU预热。单帧自然缓存的最大样本为 train 3.394秒、test 14.967秒，全部保留，未剔除。它们拉高均值；目前无 profiling trace，不能断言慢样本原因已定位为编译或系统抖动。自由生成的 sub-task 长度也可能不同，raw 中保存实际token数。

### Prefix prefill 与结论口径

本次是普通 `BatchedEvalPolicy`：ViT 缓存两采样帧的 temporal-patch 特征；每次动作决策仍重新 prefill 完整 LM prompt，KV cache 仅服务本次 sub-task 自回归解码，没有跨决策 prefix cache。发布 checkpoint 实际为固定120帧槽位（60s×2fps），不足历史用最早帧padding，并非无限增长。

execute_horizon=16 而采样stride=10，真实帧对大多重新配对。现有 raw 在各episode首决策之后每次只命中一个独立pair，主要是padding `(0,0)`；不能把“启用视觉缓存”写成“每次只编码新patch”。

仓库标准 `pipelined_policy.py` 每步为下一滑窗重新构建前59个patch的前缀，要求固定窗口及 execute_horizon=20。当前仿真驱动在同步完成下一前缀后才返回动作、调用pool.step，未真正重叠机器人执行；内部critical计时还排除了部分ViT/预处理。因此本次约1.6×典型加速不是“单帧相对论文最优异步streaming”的对比。附录F的绝对时间戳SWA append/evict是另一套训练／部署方案。

### 可以下的结论与不能外推的结论

1. 发布策略在这四任务上强烈依赖历史输入及对应训练适配；直接单帧化还改变了视觉上下文长度与位置分布，不能只归因于“VLM时序理解能力”。
2. 发布DiT不变而替换原生VLM后归零，说明生成行为与隐藏表征／动作头存在强协同适配；支持实践上优先完整发布权重后训练，但不能证明所有有效方案都必须解冻或机器人微调VLM。
3. 发布版统计来源存疑仍未证实测试集参与训练。`episode_index.max=3199` 不证明编号连续、3200个唯一episode都被采样、或额外数据就是官方val/test。根目录全局stats会被直接复制，不能当DataLoader训练清单。原文中“确定训练3200条”“新seed可彻底确认泄漏”的说法已撤回。
4. 新seed/开环误差、train/test成功率差和VLM替换掉点都是间接证据，不能代替可核对的训练task/seed清单。

### 产物与查看入口

- 原生VLM替换A/B：`/mnt/cpfs/heping/SimpleMemVLA-data/outputs/memcache/eval/vlm_swap_ab_20261025_104520/`。
- 单帧成功率／latency主目录：`/mnt/cpfs/heping/SimpleMemVLA-data/output/released_currentframe_20261025_142906/`。
- 最终中文结论：该目录 `ANALYSIS_zh.txt`；精确复核数值：`verified_metrics.json`；自动汇总：`SUMMARY.txt`／`summary.json`。
- prefix核查：同目录 `PREFIX_PREFILL_NOTES_zh.txt`；原始计时：`latency/{train,test}/raw.jsonl`；成功率和录像：`accuracy/`。
- 本次文档同步还在该主目录 `documentation/` 留有日志／交接副本及同步清单，便于只查看数据目录时追溯结论。


## 下一轮提案：完整发布权重的单帧后训练与独立 memory（尚未启动）

用户提出的两组方向合理，定位为“发布模型的单帧适配及记忆恢复”诊断／工程对照；不能混同旧主实验。建议使用新run名 `E1_fullpretrain_sf`、`E2_fullpretrain_sf_cache`，新输出目录，禁止自动续接旧E1/E2或warmdit。

| 暂称 | 共同初始化 | 唯一希望比较的变量 | 目的 |
|---|---|---|---|
| 新e1 | 同一个发布VLM+DiT、相同processor/stats | 当前帧，无历史cache | 给定数据和后训练预算下的无历史基线 |
| 新e2 | 与新e1相同，新增memory从零初始化 | 加入明确界定的naive历史视觉token cache | 测量memory在同等后训练下的增益 |

“单帧上限”应理解为给定配方／预算下的经验基线，不是理论最优。部分可观测任务可能同一当前图像对应不同正确动作，缺失历史的信息不能靠后训练保证恢复。

用户已明确：**cache接DiT，作为独立memory，沿用原研究方向；不接VLM。** 当前只完成方案评审，尚未实现或启动。训练条件接口仍需落实：

- 原版训练与推理要分开：训练把标注sub-task文本放入assistant输入，teacher forcing前向，DiT读取这些GT token对应的VLM hidden states和输入embedding；推理则自回归生成sub-task，DiT读取生成token对应的hidden/embedding。DiT始终接收表征，不是直接接收字符串。“hidden来自VLM”与“训练token来自GT”并不矛盾。
- 原版teacher forcing是正常训练设计，不能表述为“禁止GT”或“原论文这样做是错误／测试泄漏”。新增DiT侧memory后的潜在捷径只是风险：GT答案可能已提供历史线索，导致memory需求减少；是否实际发生需由消融验证，不能预先断言一定不学memory。
- cache接DiT：VLM确实只看当前帧；第一轮保留原版sub-task训练／生成推理接口作为最小改动对照是合理选项。应记录GT条件与实际生成条件下的开环差距，以及memory zero/shuffle后的变化；不能仅凭纯GT条件下“无增益”否定memory。
- cache接VLM：历史可直接帮助sub-task预测，原接口改动较小；此时应称“当前帧在线编码+历史视觉token上下文”，VLM已经看到历史，LLM prefill仍随token量增长，不能当作短上下文单帧方案。

统一prompt hidden条件，或在实际生成／错误sub-task条件上适配DiT，都是处理该风险的候选消融，不是未经验证就必须切换的接口。两组应采用相同条件策略，并在启动前明确；不把“必须改prompt”视为已决策。若改用prompt接口，GT答案必须在因果条件span之外，且10%/2%的旧sub-task单帧结果只能作历史参考；还需新接口step0对照，不能把接口改变与后训练的联合增益全归因于“后训练本身”。

建议共同固定：同一初始化、train360／内部val40、训练seed/global batch/步数、当前帧表示、DiT学习率预算、动作执行horizon与评测episode。先冻结ViT/merger，语言层适配策略与DiT训练策略两组一致；如选LoRA/仅解冻语言层，需要新增实现，当前只有整块backbone冻结开关。cache必须来自同一发布ViT及一致预处理/dtype，不能复用旧原生Qwen cache；若视觉塔更新，离线cache会陈旧。

当前帧表示优先与本次单帧消融保持一致（front当前帧复制为最短video patch、wrist当前image），两组一起适配；训练视觉输入选择与动作条件选择应解耦。工程诊断可先共同保留发布版stats，避免额外改变动作／状态坐标；其统计来源风险必须记录。若改为train-only stats，要两组同时改并重新检查step0行为，不能将归一化变化误算成memory效果。

naive应明确定义帧数、每帧token数、时间／age编码、投影与门控；第一轮不要默认沿用4层Transformer+2层Perceiver而称“最原始memory”。可从固定空间池化+轻量投影+时间／age编码直接接DiT入手。原始120×64=7680个token，池化后120×16=1920个，也可能显著拖慢DiT；可以考虑保留60秒窗口、每帧2×2=4个token（共480），作为待确认的小预算起点，必须同步测实际token预算与latency。新memory分支关闭时要验证能数值退化到E1；仅把prefix token乘0并不能保证attention结果不变。

E2须配套memory zero／跨episode shuffle／时间打乱诊断，检查是否利用历史内容。保留完整视频和未后训练单帧10%／2%作为参照，分开报告“后训练本身恢复”和“memory额外恢复”。若cache接DiT，可额外看基础操作阶段与记忆决策阶段。

代码状态：现有训练 `single_frame=True` 与 `condition_source=prompt` 强绑定，不是此次评测的最短video+sub-task路径；旧 `E*_warmdit` 全冻结VLM且vl_loss=0，不能直接当新方案启动。保留sub-task方案需要独立训练current-frame模式；当前memory仅接DiT，未支持naive cache送VLM。

使用发布完整权重时，预训练数据seed来源仍未核实；本轮应明确为初始化／工程诊断，不能宣称已证明完全无评测数据污染风险。当前没有创建新训练run、修改训练配置或启动训练。
