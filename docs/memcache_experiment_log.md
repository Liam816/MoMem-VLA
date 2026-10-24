# Memory-Cache 训练迭代日志

> 记录"单帧输入 + 历史 vision-token cache"方向的每一组训练和消融：条件、命令、产物位置、结果和结论。
> 设计方案见 `MEMCACHE_PLAN.md`。代码在服务器分支 `memcache-singleframe`。
> 服务器副本：`/mnt/cpfs/heping/SimpleMemVLA/docs/memcache_experiment_log.md`

---

## 0. 怎么记录

- 每组实验一个 ID（`E<编号>`，消融可以写成 `E2a`、`E2b`），在 §2 总表里加一行，在 §3 写详细记录。
- 配置只写**相对参考实验的差异**。完整配置以 run 目录里保存的 `config.json` 和 `training_args.bin` 为准。
- 结果统一按 §1.4 的评测协议，表里填 **best val** 和 **final val**，外加两个 memory 诊断量。
- 结论分三档：✅ 有效 / ❌ 无效 / ❓ 待定。写清楚依据是哪个数。

### 配置从哪来

| 层级 | 位置 | 说明 |
|---|---|---|
| 实验变体 | `scripts/train_mem.sh` 里的 `case "$EXP"` | `E1/E2/E3` 只切换 `USE_MEMORY` 和 `MEM_FUSION`，其余超参都是脚本里的默认值，可以用环境变量覆盖 |
| 启动 | `scripts/launch_memcache.sh <EXP> <gpus>` | 负责 conda 环境、`GRAD_ACCUM` 自动凑出 global batch 64、`RESUME=auto`、后台运行和带时间戳的 log；`DRY_RUN=1` 只打印完整命令 |
| 参数定义 | `configs/sft_params.py` | `ModelArguments`、`DataArguments`、`SFTTrainingArguments` |
| 实际生效的配置 | `<run>/config.json`（模型）、`<run>/checkpoint-*/training_args.bin`（训练） | 以这里为准 |

---

## 1. 所有实验共享的设置

### 1.1 数据

| 项 | 值 |
|---|---|
| 数据集 | `yinchenghust/robomme_lerobot`（rev `b72788e`），20 fps |
| 子集 | 4 个任务，每个任务 100 条：PickXtimes（700–799）、VideoUnmaskSwap（1500–1599）、VideoRepick（1300–1399）、PatternLock（500–599） |
| 划分 | 每个任务随机留 10 条做 val（seed 0）。train 360 条，**113,011 个锚点**（已去掉 demo 帧）；val 40 条 |
| 划分文件 | `SimpleMemVLA-data/splits/robomme_small_{train,val}.txt` |
| 归一化 | `checkpoints/simplememvla_robomme/stats.json`，用的是 ckpt 自带的 stats，不是数据集的（原因见 MEMCACHE_PLAN §10.3） |
| 增强 | 关闭（`image_aug=false`） |

### 1.2 模型

| 项 | 值 |
|---|---|
| 初始化 | `simplememvla_robomme` ckpt：backbone 和 DiT 都从这里热启动 |
| VLM | Qwen3.5-4B，**全冻结**（4539M 参数，不参与训练） |
| VLM 输入 | 单帧：front + wrist 当前帧，都作为 image（每张 64 个 token），没有历史视频 |
| DiT 条件 | `condition_source=prompt`：冻结 VLM 对整段 prompt 所有有效 token 的 last hidden（约 219 个 token），不生成 sub-task；`vl_loss_weight=0` |
| DiT | 16 层，hidden 2048，1169M 可训练参数，带 proprio state token，动作块 30×8 |
| Loss | flow matching，`t ~ Beta(1.5, 1)` |

### 1.3 优化

| 项 | 值 |
|---|---|
| 硬件 | 2× RTX 4090（48G），DeepSpeed ZeRO-2，bf16 + tf32 |
| batch | 每卡 16 × 2 卡 × `GRAD_ACCUM` 2 = **global batch 64** |
| 步数 | 15000 步（约 8.5 个 epoch），warmup 500 步 |
| LR | action_head 3e-5；memory 1e-4；backbone 0。cosine 衰减到 0.1× |
| 其他 | AdamW（β2=0.95），weight decay 1e-8，梯度裁剪 1.0，seed 429，不开 gradient checkpointing |
| 保存 | 每 2500 步存一次，保留最近 2 个（每个 checkpoint 约 45 GB） |

### 1.4 评测协议

- 每 1000 步在 val 上评一次，每卡 50 个 batch × 16，2 卡共 **1600 个样本**。采样顺序固定，每次 eval 用的是同一批样本。
- 每个 batch 的 noise 和 t 固定，因此下面三种模式之间只有 memory 输入不同：
  - `val`：真实 memory；
  - `zero`：memory 置零（m=0）；
  - `shuffle`：batch 内循环移位，换成别的样本的 memory。
- 诊断量：
  - `gapZ = loss(zero) − loss(val)`：> 0 说明模型依赖 memory 的存在；
  - `gapS = loss(shuffle) − loss(val)`：> 0 说明模型依赖 **memory 的内容**。这是最关键的指标；
  - `mem_norm`：训练时融合向量 m 的平均 L2 范数。
- 目前只有开环 flow loss，还没有闭环成功率，也还没有单独统计"记忆关键帧"子集。

---

## 2. 实验总表

| ID | 日期 | 相对参考的改动 | 状态 | best val（step） | final val | gapZ（最后） | gapS（最后） | 结论 |
|---|---|---|---|---|---|---|---|---|
| E1 | 10-24 | 参考组：单帧，**无 memory** | 被外部杀掉，停在 13965/15000 | 0.03215（13k） | —（13k 时 0.03215） | — | — | 下界 |
| E2 | 10-24 | E1 + memory cache，融合方式 `adaln_add` | 完成 | 0.03168（13k） | 0.03288（15k） | +1.1e-4 | **0** | ❌ memory 的内容没有被用上 |

---

## 3. 详细记录

### E1：单帧、无 memory（下界）

- **目的**：给出冻结 VLM、只看当前帧时的下界。
- **配置**：§1 的共享设置，`--use_memory false`。
- **命令**：
  ```bash
  bash scripts/launch_memcache.sh E1 4,5
  ```
  首次启动用的是 `run_E1.sh`，其中 `EXP=E1 CUDA_VISIBLE_DEVICES=4,5 GRAD_ACCUM=2 bash scripts/train_mem.sh`。
- **产物**：
  - 输出：`SimpleMemVLA-data/outputs/memcache/robomme_small_E1/`，含 `checkpoint-10000`、`checkpoint-12500`
  - 日志：`outputs/memcache/run_E1.log`
  - wandb（离线）：`SimpleMemVLA/wandb/offline-run-20261024_004748-8l59dtnx`
- **代码版本**：分支 `memcache-singleframe`，基于 `e39ab1a`，未 commit（工作区有 17 个文件改动）。
- **运行情况**：
  - 00:41 启动，平均 2.17 s/step。
  - 09:33 在第 13965 步被外部杀掉：没有 traceback，外层 shell 的 `TRAIN_EXIT` 也没打出来，说明整组进程一起被杀；dmesg 里没有当天的 OOM 记录。之后 8 张 GPU 都被另一个容器的任务占用。
  - 可以用 `launch_memcache.sh E1 4,5` 从 `checkpoint-12500` 续训，必须用 2 卡。
- **曲线**：

  | step | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k | 11k | 12k | 13k |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | val | .0573 | .0466 | .0415 | .0401 | .0387 | .0379 | .0351 | .0370 | .0355 | .0346 | .0327 | .0328 | .0322 |
  | train（eval 前 200 步均值） | .0531 | .0446 | .0404 | .0322 | .0310 | .0298 | .0280 | .0263 | .0239 | .0219 | .0195 | .0186 | .0175 |

- **结论**：val 到 13k 步还在缓慢下降，train 和 val 的差距持续拉大（13k 时 train 0.0175、val 0.0322），存在过拟合。

### E2：单帧 + memory cache（`adaln_add`）

- **目的**：主方法第一版，看历史 cache 经 AdaLN 注入 DiT 后能不能降低 loss。
- **配置**：E1 加上下面这些。和 E1 实际生效的差异**只有 `use_memory`**，以及它带来的 `mem_feature_dir`、`mem_feature_dim=2560`、`mem_camera_key=front`。
  - **Cache 特征**：`vit_merger`，取自 `features/robomme_vit_merger_front`，front 相机。
  - **采样窗口**：2 fps，窗口 60 s，最多 120 帧。
  - **帧内压缩**：每帧 pool 成 4×4，即 16 个 token。
  - **MemoryEncoder**：d=1024，4 层，16 头，mlp×4。
  - **压缩到定长**：Perceiver，16 个 query，2 层。
  - **融合**：`adaln_add`，即 `c = t_emb + g(flatten(Z))`，`g` 的最后一层零初始化。
  - **其他**：`mem_dropout_prob=0`，memory 模块 125.7M 参数，lr 1e-4。
- **命令**：
  ```bash
  bash scripts/launch_memcache.sh E2 6,7
  ```
  首次启动用的是 `run_E2.sh`，其中 `EXP=E2 CUDA_VISIBLE_DEVICES=6,7 GRAD_ACCUM=2`。
- **产物**：
  - 输出：`outputs/memcache/robomme_small_E2/`，含最终模型、`checkpoint-12500`、`checkpoint-15000`
  - 日志：`run_E2.log`
  - wandb（离线）：`offline-run-20261024_004748-edv6qofr`
- **代码版本**：同 E1。
- **运行情况**：00:41 启动，11:25 完成，`TRAIN_EXIT=0`。共 10.6 h，2.55 s/step，25 samples/s。
- **曲线**：

  | step | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k | 11k | 12k | 13k | 14k | 15k |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | val | .0587 | .0471 | .0423 | .0387 | .0382 | .0380 | .0368 | .0353 | .0353 | .0367 | .0323 | .0337 | .0317 | .0328 | .0329 |
  | zero | .0587 | .0471 | .0423 | .0387 | .0383 | .0381 | .0369 | .0353 | .0354 | .0367 | .0324 | .0337 | .0318 | .0330 | .0330 |
  | gapZ（×1e-4） | −0.1 | −0.2 | 0.2 | −0.3 | 0.7 | 0.9 | 1.0 | 0.5 | 0.9 | 0.1 | 1.1 | 0.8 | 1.7 | 1.2 | 1.1 |
  | gapS | ~0 | ~0 | 0 | ~0 | 0 | 0 | 0 | ~0 | ~0 | 0 | ~0 | 0 | 0 | 0 | 0 |
  | train | .0531 | .0449 | .0401 | .0328 | .0306 | .0295 | .0279 | .0255 | .0239 | .0219 | .0193 | .0189 | .0179 | .0168 | .0171 |
  | mem_norm | .056 | .059 | .063 | .067 | .073 | .076 | .081 | .083 | .085 | .085 | .086 | .086 | .085 | .086 | .086 |

- **和 E1 对比**：同一步数下 val 差值在 ±0.0017 以内，有正有负，属于噪声，没有系统性改善。
- **结论**：❌ memory 的**内容**没有被利用。
  - `gapS` 全程约为 0：换成别的样本的 memory，loss 完全不变。memory 分支学成了一个和输入无关的常数偏置。
  - `gapZ` 只有 1e-4 量级（约占 loss 的 0.3%），对应的正是这个常数偏置。
  - `mem_norm` 一直很小，只有 0.06–0.09。
- **可能的原因**（待逐一验证）：
  1. 小数据上过拟合：8.5 个 epoch 后 train 0.017、val 0.033。DiT 靠单帧加 proprio 背下了训练轨迹，没有动力去用 memory。
  2. 评测稀释：val 里大部分帧不依赖记忆，即使 memory 在少数帧上有用，平均 loss 也看不出来。
  3. 融合瓶颈：AdaLN 只是一个全局向量的仿射调制，加上零初始化，梯度信号太弱。
  4. **需要排查实现**：`gapS` 正好等于 0（不是很小），应该确认 shuffle 模式确实改变了 memory 输入。具体做法是用 E2 的 ckpt 统计 m 在不同样本之间的方差，再做一次单样本的前向对比。

---

## 4. 待做的消融（按优先级）

| ID | 改动（相对 E2） | 要回答的问题 | 状态 |
|---|---|---|---|
| E2-diag | 不训练：用 E2 ckpt 统计 m 在样本间的方差，并检查 shuffle 是否真的生效 | `gapS=0` 是实现问题还是模型行为 | 待做，优先 |
| E2-crit | 不训练：在"记忆关键帧"子集上重算 E1 和 E2 的 val、gapZ、gapS | 差异是不是被大量无关帧稀释了 | 待做，需要先标注子集 |
| E3 | `mem_fusion=prefix_tokens`：16 个 memory token 直接拼进 DiT 序列 | AdaLN 全局向量是不是瓶颈 | 待做 |
| E2d | `mem_dropout_prob=0.3` | 随机置零 memory 能否逼模型利用它 | 待做 |
| E2s | `MAX_STEPS=5000` 或 `action_head_lr=1e-5` | 减轻 DiT 的过拟合后，memory 是否开始起作用 | 待做 |
| E2p | `mem_frame_pool`：64 / 4 token/帧 | 帧内压缩率 | 第二轮 |
| E2q | `mem_num_queries`：4 / 64 | 定长 token 数 | 第二轮 |
| E2w | 窗口 30 / 120 s | 历史长度 | 第二轮 |
| E1c、E2c | 闭环评测（4 个任务 × 50 个 episode） | 开环 loss 和成功率之间的关系 | 需要先写 policy |
