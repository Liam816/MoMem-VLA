# 交接文档：SimpleMemVLA 上的"单帧 + 历史 Vision-Token Cache"记忆实验

> **2026-10-28 当前状态**：正式配置为 `configs/train/memcache/e1_5w_iter.yaml` 和 `e2_5w_iter.yaml`。两组完整加载发布 VLM+DiT、冻结 ViT、LM decoder LoRA32、VLM 当前帧；E1 无 memory，E2 naive vision-token cache 接 DiT。两组均已完成 50000 步。下文原生 Qwen、随机 DiT、Perceiver/adaLN 等内容是历史方案，不能代替这两份正式配置。
> 五阶段 val40/test200 已完成：E1 test 为 13.5/15/13.5/14.5/17%，E2 为 13/16/19/20/19%（5000/16250/27500/38750/50000）。完整最终记录统一见 Mac `/Users/liam/Projects/research/mem-vla/EXP_LOG.md` §28–29。
> 复现注意：canonical 模型先初始化 memory 再注入 LoRA，单设相同 seed 不保证 E1/E2 的 LoRA 初值相同。已完成的 50k v2 实验由各 run 的启动快照隔离 memory 的 CPU RNG；复现这些实验应同时沿用对应启动快照。本次提交整理保留模型计算语义。

> 读者：接手这个项目的 agent。
> 初稿写于2026-10-25；§9保留10:10的历史快照。**本轮已同步评测、latency与prefix核查，最新结果见§12，下一轮提案见§13**。
> **当前 SSH 端口为 8096（2026-10-28 复核）**；下文有日期的旧连接与实验记录按历史快照阅读。
>
> 配套文档（服务器仓库 `docs/` 下有同名副本，可能比本文件旧）：
> - 实验日志：`docs/memcache_experiment_log.md`。每组实验的条件、命令、曲线、结论都在里面，最全。
> - 方案设计：`MEMCACHE_PLAN.md`（只在本地会话目录里，服务器上没有）。
> - 结构图：`docs/memcache_architecture.svg`。
> - 项目总览：用户自己维护的 `MEMORY_VLA_CONTEXT.md`（服务器、目录约定、数据集清单）。

---

## 0. 一句话概括

以 SimpleMemVLA（Qwen3.5-4B VLM + DiT flow-matching 动作头，原版把历史视频直接喂给 VLM）为 baseline，设计并验证一种新的记忆机制：

- VLM **冻结**，只看**当前帧**；
- 历史帧的 ViT token 存进 **cache**，经过几层 Transformer 和 Perceiver 压成**定长 token**；
- 定长 token 以 **adaLN** 的方式融合进 DiT。

在 RoboMME 的 4 个任务的小数据集上做验证。

**进度**：
- 代码、数据、评测都已经打通，并逐项核对过训练和评测的输入一致。
- 公平设定下（原版 Qwen 冻结，DiT 从头训）：
  - 开环：E2 确实用上了 memory，val loss 0.042，E1 是 0.056；打乱 memory 后 loss 涨 0.117。
  - 闭环：10k步E1/E2成功率接近0，train40也为0，说明问题不只在泛化，基础控制尚不足；不能由此断言与memory完全无关或把原因唯一归于冻结VLM。
- 发布版视频策略train40为100%、test200为86%；同权重仅改单帧为10%／2%；发布DiT不变、VLM换原生Qwen为0%／0%。仅是四任务子集，不是全16任务论文复现。latency与解释见§12。

---

## 1. 服务器与环境

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/heping6580_ed_pai -p 8096 root@dsw.quicrobot.xyz
```
（2026-10-28 当前用 8096 端口；scp 对应用 `-P 8096`。后续以用户提供的连接为准。）

- 阿里云 DSW 容器，8× RTX 4090（每卡 46 GB），120 核 CPU，885 GB 内存，共享盘 `/mnt/cpfs`。
- **共享盘快满了**：10-25 10:10 只剩 386 GB（使用率 99%）。删掉作废的 warmdit run 之后**剩 576 GB（98%）**。每个 checkpoint 约 45 GB，新开 run 之前一定要先看 `df -h /mnt/cpfs`。
- 10-25 端口改成 8097 后，`uptime` 显示的启动时间从 11 天变成了 3 天，容器可能被迁移或重启过。运行中的进程不会自动恢复，但 `/mnt/cpfs` 上的文件都还在。
- **机器是多人共用的**：GPU 和 CPU 都会被别人的任务占用，load 经常在 100–300 之间。启动任何任务前先看 `nvidia-smi`。
- conda 环境：**训练和 RoboMME 评测都用 `simplememvla-robomme`**（torch 2.4.1，transformers 5.13.1，sapien 3.0.3）。非交互方式执行命令时要手动激活：
  ```bash
  source /mnt/cpfs/heping/miniconda3/etc/profile.d/conda.sh
  conda activate simplememvla-robomme
  source /mnt/cpfs/heping/SimpleMemVLA-data/eval_env.sh   # HF mirror、各类 cache 目录、MUJOCO_GL 等
  ```
- 外网只能通过 `HF_ENDPOINT=https://hf-mirror.com` 访问 HF；`eval_env.sh` 已经设置好。
- 长任务一律用 `setsid nohup bash xxx.sh > log 2>&1 < /dev/null &` 放到后台。**SSH 命令里不要用 `pkill -f`**，会把当前 SSH 会话一起杀掉，要用 `kill <PID>`。

### 已知的坑（都踩过）

| 问题 | 现象 | 处理 |
|---|---|---|
| `eval_env.sh` 把 `TMPDIR` 设在 CPFS 上 | DataLoader worker 回传张量要建 Unix socket，CPFS 不支持，报 `OSError: [Errno 95]`，训练卡住不动 | 训练和评测都设 `TMPDIR=/dev/shm/simplememvla_tmp`。`train_mem.sh` 和下文的评测 wrapper 已经设好 |
| checkpoint 很大 | 每个约 45 GB（冻结的 backbone 12.9 GB 也会存进去，再加上 DeepSpeed 优化器状态） | `save_total_limit: 2`；注意共享盘空间 |
| CPU 上构建模型很慢 | 4.5B 参数先以 fp32 构建再转换，机器负载高时一个进程要约 10 分钟 | 正常现象，等着就行 |
| 训练和评测共用 GPU | 评测每卡约多占 14 GB，峰值到 43–44 GB（总共 46 GB）；**10-25 E2 训练在第 20000 步崩溃**，原因是一个 DataLoader worker `LLVM ERROR: pthread_join failed` 后 abort | 尽量别和训练挤同一批卡，或者调小评测的 group_size |
| SSH 偶尔断开 | `kex_exchange_identification: Connection closed`，或 `Permission denied (publickey,password)` | 10-25 出现过 publickey 被拒，原因是用户把端口换成了 8097。遇到这种情况先问用户端口是否变了 |
| macOS 打的 tar 包 | 解包后会带出 `._*` AppleDouble 文件 | 本地往服务器推文件时用 `COPYFILE_DISABLE=1 tar ...`，或者用 `git apply` 打 patch |

---

## 2. 研究内容和方法

### 2.1 Baseline：原版 SimpleMemVLA
- VLM 输入是主相机 60 s、2 fps 的历史视频，加上 wrist 的当前帧和 prompt。VLM 先**生成 sub-task 文本**，再把这段 sub-task span 的 hidden 状态和 embedding 作为 DiT 的唯一条件。VLM 全量微调。
- 详细链路见 `docs/model_pipeline.md`。

### 2.2 我们的方法（E2）
结构图见 `docs/memcache_architecture.svg`。
- **VLM**：**原版 Qwen3.5-4B，冻结**，只看当前的 front 和 wrist 两帧（每帧作为单张 image，64 个 token）。不生成 sub-task。DiT 的条件是**整段 prompt 所有有效 token 的 last hidden 和 token embedding**（`condition_source: prompt`，约 219 个 token）。
- **Cache**：front 相机每 0.5 s 取一帧（`frame_index % 10 == 0`，**从 episode 第 0 帧开始计，RoboMME reset 时回放的 demo 帧也算在内**），最近 60 s，最多 120 帧。每帧的 ViT merger 输出是 64×2560，平均池化成 4×4=16 个 token；再加上帧龄 age = t − t_k（秒）。
- **Memory 分支**（约 126M 参数，随机初始化）：
  - LN + Linear（2560→1024）+ 空间位置编码 + 帧龄嵌入；
  - 4 层双向 Transformer（d=1024）；
  - Perceiver：16 个可学习 query，2 层 → 得到 Z（16×1024，定长）；
  - 融合 MLP：flatten → 2048 → 2048，末层零初始化，输出 m。
- **融合方式**（`mem_fusion: adaln_add`）：c = TimestepEmb(τ) + m，c 送进 DiT 每一层和 final layer 的 adaLN。另一种变体 `prefix_tokens`（E3）：Z 投影成 16 个 token，直接插进 DiT 的序列。
- **DiT**：结构和发布版一样（16 层，2048 维，1.17B 参数），**从头训练**。输入序列是 [cond tokens | state token | 30 个加噪动作 token]。

### 2.3 为什么改用 prompt 条件（用户确认的决定 D1）
GT sub-task 本身就带着记忆的答案（比如"move left"、"pick up the green cube for the third time"）。如果继续用 sub-task 作为条件，训练时就等于把答案泄露给了 DiT，memory 分支学不到东西。

### 2.4 公平性：几次踩坑后的修正（重要）
1. **E1w / E2w（作废）**：最早把发布版 ckpt 的 **DiT 也一起热启动**了。它带有发布模型的控制先验，且训练episode来源不明、存在与本地train/val重叠风险，不能作为随机DiT从头学习memory的对照；不能仅凭stats认定已泄漏。
2. 发布版`stats.json`记录episode编号上界3199、覆盖1,535,626帧，公开train则为1600条、768,897帧。编号上界与官方train+val+test合计3200相符是来源疑点，但stats是全局归一化统计，不证明编号连续、实际训练3200条或包含test。用户怀疑原作者用 test 训练过，所以主实验**完全不用任何 SimpleMemVLA 权重**，VLM 改用原版 `Qwen/Qwen3.5-4B`。
3. 用户确认过：**训练用的 train 和 val 都只来自官方 train 集**。官方的 val 和 test 列表（`robomme_sim/robomme/env_metadata/{train,val,test}`，每个任务分别是 100/50/50 条，seed 互不重叠）从没用于训练或调参。test 只用于最终的闭环评测。

---

## 3. 代码状态

- 仓库：`/mnt/cpfs/heping/SimpleMemVLA`，分支 **`memcache-singleframe`**。
- 已经 commit 的（用户提交，作者 ping.he）：
  - `f7cc0c1` memcache：单帧 + 历史 vision-token memory 实验（第一版代码）
  - `58d4d6b` config：训练和评测改为由 YAML 驱动（`simplememvla/config.py`）
  - `c1ba78b` scripts：拒绝残留的旧超参环境变量（`scripts/legacy_env.sh`）
- **还没 commit 的改动**（都是在 `c1ba78b` 之后做的，**提交前先 `git diff` 看一遍**）：

| 文件 | 改动 |
|---|---|
| `configs/sft_params.py`、`simplememvla/model/configuration_simplememvla.py` | 新增 `init_modules`（从 ckpt 加载哪些模块：backbone 必选，action_head 可选），config 里记录 `init_from_checkpoint` 和 `init_modules` |
| `simplememvla/model/modeling_simplememvla.py` | `from_simplememvla_checkpoint(..., init_modules=...)` 可以只加载部分模块，没加载的 DiT 按 config 新建 |
| `train.py` | 只有热启动 DiT 时才默认用 ckpt 的 stats；**检查 memory cache 的 `meta.json` 和当前 VLM 是否来自同一个模型**，不一致直接报错；打印 `[init]` 信息 |
| `configs/train/memcache/*.yaml` | 新的 `base.yaml`、`E1`/`E2`/`E3` 用原版 Qwen；新增 `*_ckptvlm.yaml` 和 `*_warmdit.yaml` 两组变体（见 §5） |
| `scripts/extract_mem_features.py` | 支持直接用 HF 原版 backbone 目录；新增 `--vit_dtype`（默认 float32，fp32 时自动用 sdpa），`meta.json` 里记录 `vit_dtype` |
| `scripts/sanity_memcache.py` | 新增 `--backbone`（原版 HF VLM）和 `--init_modules`；特征一致性检查改为看 cos 均值 |
| `scripts/launch_memcache.sh`、`scripts/train_mem.sh` | EXP 可以是 `configs/train/memcache/` 下任意一个 yaml 的名字；如果会续训会提前提示 |
| `robomme_sim/memcache_policy.py`（**新文件**） | memcache 模型的闭环评测 policy，见 §7 |
| `robomme_sim/eval_success.py` | 遇到 `condition_source=prompt` 的 ckpt 自动切到 memcache policy；新增 `--episode_ids_json` |
| `scripts/diag_memory.py`（**新文件，未跟踪**） | 诊断 memory 分支是否塌缩 |
| `docs/memcache_experiment_log.md`、`docs/memcache_architecture.svg` | 实验日志、结构图 |
| `data/`（未跟踪） | 软链接 `data/datasets -> /mnt/cpfs/heping/SimpleMemVLA-data/datasets` |

### 代码地图（memcache 相关）
- `simplememvla/model/memory.py`：`VisionTokenMemory`（encoder + Perceiver + 融合）
- `simplememvla/model/dit_action_head.py`：`memory_cond`（加到 time_emb 上）、`memory_tokens`（prefix 变体）
- `simplememvla/model/modeling_simplememvla.py`：`_memory_condition`（`mem_mode` 支持 normal/zero/shuffle）、`condition_source=prompt`、`from_simplememvla_checkpoint`
- `simplememvla/data/messages.py`：`build_single_frame_messages`
- `simplememvla/data/dataset.py`：`single_frame` 模式；`select_mem_keyframes`、`pool_mem_tokens`、`mem_keyframe_stride`（**训练和推理共用这几个函数**）
- `simplememvla/data/collator.py`：prompt 模式下 `add_generation_prompt=True`，输出 `condition_span`；`pad_memory`
- `simplememvla/training/sft_trainer.py`：backbone / action_head / memory 三组参数各自的 lr；`evaluate()` 输出 `eval_action_loss` 以及 `eval_mem_gap_zero`、`eval_mem_gap_shuffle`

---

## 4. 数据与权重

| 东西 | 路径 | 说明 |
|---|---|---|
| RoboMME 训练集 | `/mnt/cpfs/heping/SimpleMemVLA-data/datasets/yinchenghust/robomme_lerobot` | rev `b72788e`，LeRobot v3，20 fps，1600 条 episode = **官方 train 集**（每个任务 100 条，编号 0–99 和 `env_metadata/train` 一一对应） |
| 小数据集划分 | `SimpleMemVLA-data/splits/robomme_small_{train,val}.txt` | 4 个任务：PickXtimes（数据集 ep 700–799）、VideoUnmaskSwap（1500–1599）、VideoRepick（1300–1399）、PatternLock（500–599）。每个任务随机留 10 条做 val：train 360 条（113,011 个锚点），val 40 条 |
| 原版 VLM | `/mnt/cpfs/heping/checkpoints/Qwen3.5-4B` | `Qwen/Qwen3.5-4B` rev `851bf6e`，9.3 GB，14 个文件都核对过 |
| **E1/E2/E3 用的 cache** | `SimpleMemVLA-data/features/robomme_vit_merger_front_qwen35base` | **原版 Qwen 的 ViT，fp32 抽取**，400 条 episode，17,964 个关键帧，5.9 GB。每个 episode 存成 `episode_XXXXXX.npy`，形状 (K, 64, 2560)，fp16；附 `meta.json` |
| `*_ckptvlm` / `*_warmdit` 用的 cache | `SimpleMemVLA-data/features/robomme_vit_merger_front` | 发布版 ckpt 的 ViT，bf16 抽取 |
| 冒烟测试用的 cache | `SimpleMemVLA-data/features/_smoke` | 只有 ep 700 和 1500，发布版 ViT |
| 发布版 ckpt | `SimpleMemVLA-data/checkpoints/simplememvla_{robomme,mikasa,rmbench,robomemarena,libero}` | 只用作参考 baseline 和评测链路验证，**主实验不用** |

**归一化**：
- DiT 从头训的配置用数据集自己的 `meta/stats.json`；
- 训练时会把这份 stats 复制进每个 checkpoint 目录，评测读的是 `ckpt/stats.json`。

**为什么 cache 必须用 fp32 抽取**：
- 原版 Qwen ViT 在 bf16 下，同一帧单独编码和放进 batch 里编码，个别 token 的 cos 只有 0.706；fp32 下完全一致。
- 闭环评测时关键帧是在线逐帧编码的，所以抽取和评测**都必须用 fp32**。`memcache_policy.py` 已经这样做了。

---

## 5. 配置体系

YAML 支持 `_base_` 继承和 `${VAR:-default}` 写法，命令行参数优先级更高。train.py 会把最终解析出的配置写到 `<output_dir>/train_config.yaml`。

| 配置 | VLM | DiT | cache | 用途 |
|---|---|---|---|---|
| `configs/train/memcache/base.yaml` → **E1 / E2 / E3** | **原版 Qwen3.5-4B**，冻结（`init_from_checkpoint: null`） | 从头训 | `_qwen35base`（fp32） | **主实验**：E1 无 memory，E2 用 `adaln_add`，E3 用 `prefix_tokens` |
| `base_ckptvlm.yaml` → `E*_ckptvlm` | 发布版 ckpt 的 backbone（`init_modules: [backbone]`），冻结 | 从头训 | `robomme_vit_merger_front` | 对照组：看机器人数据微调过的 VLM 带来多少差别（发布版 VLM 可能见过 test） |
| `base_warmdit.yaml` → `E*_warmdit` | 发布版 backbone | **热启动（来源风险未核实）** | `robomme_vit_merger_front` | 作废，只用来复现 E1w/E2w |

主实验的关键超参（`base.yaml`）：
- 每卡 batch 16，global batch 64；30000 步，warmup 1000；
- lr：action_head 5e-5，memory 1e-4，backbone 0；cosine 衰减到 0.1×；
- 每 1000 步 eval（每卡 50 个 batch），每 5000 步保存，保留最近 2 个；
- `image_aug: false`，不开 gradient checkpointing。

---

## 6. 怎么训练

```bash
cd /mnt/cpfs/heping/SimpleMemVLA
bash scripts/launch_memcache.sh E1 0,1,2,3          # 后台启动；GRAD_ACCUM 自动设为 4/卡数，凑出 global batch 64
bash scripts/launch_memcache.sh E2 4,5,6,7
DRY_RUN=1 bash scripts/launch_memcache.sh E2 4,5,6,7  # 只打印完整的 torchrun 命令
bash scripts/launch_memcache.sh E2 4,5,6,7 --max_steps 20000 --mem_depth 6   # 后面跟的参数会覆盖 YAML
```

- **输出**：`/mnt/cpfs/heping/SimpleMemVLA-data/outputs/memcache/robomme_small_<EXP>/`。
- **日志**：`outputs/memcache/run_<EXP>_<时间戳>.log`。日志里有大量 `\r` 进度条，查看时先转一下：
  ```bash
  tr '\r' '\n' < LOG | grep -a -E 'eval_|/30000'
  ```
- **续训**：默认 `RESUME=auto`，会自动从输出目录里最新的 `checkpoint-*` 接着训。**续训必须用和原来相同的卡数**，因为 DeepSpeed 的分片和卡数绑定。要从头开始就换一个 `RUN_NAME`。
- **指定卡被占用时**：脚本会拒绝启动；确认要共用就加 `FORCE=1`。
- **停止**：用启动时打印出来的 `kill -- -<pid>`。
- **旧的环境变量**（`MAX_STEPS=...` 这类）会被直接拒绝，超参只能写在 YAML 里或者用命令行参数。
- **训练中重点看**：
  - `eval_action_loss`；
  - `eval_mem_gap_shuffle`：把 batch 内的 memory 打乱后 loss 涨了多少，> 0 说明用到了 memory 的**内容**；
  - `eval_mem_gap_zero`；
  - `mem_norm`。

### 换 VLM 或换数据后需要重做的准备
```bash
# 1) 划分（已经生成过了）
python scripts/make_episode_splits.py --ranges 700-799 1500-1599 1300-1399 500-599 --val_per_range 10 \
  --out_prefix /mnt/cpfs/heping/SimpleMemVLA-data/splits/robomme_small
# 2) memory cache：必须用训练时的那个 VLM 来抽，fp32 是默认值；可以按 shard 拆到多张卡上跑
CUDA_VISIBLE_DEVICES=0 python scripts/extract_mem_features.py --checkpoint /mnt/cpfs/heping/checkpoints/Qwen3.5-4B \
  --benchmark robomme --root <DATA> --episodes_file <train.txt> <val.txt> --out <FEAT_DIR> \
  --shard 0 --num_shards 4      # 约 7 帧/s/卡，400 条 episode 4 卡大约 10 分钟
# 3) 自检：数据、加载、cache 一致性、零初始化等价性、梯度
CUDA_VISIBLE_DEVICES=0 python scripts/sanity_memcache.py --backbone /mnt/cpfs/heping/checkpoints/Qwen3.5-4B \
  --root <DATA> --mem_feature_dir <FEAT_DIR> --episodes 700 1500 --batch_size 8 --stats_path <DATA>/meta/stats.json
```
这几条都要在激活了环境、设好 `PYTHONPATH=$PWD` 和 `TMPDIR=/dev/shm/...` 之后运行。

---

## 7. 怎么评测

### 7.1 开环（训练过程中自动跑）
`SimpleMemVLATrainer.evaluate()`：在 val 上用固定的 noise 和 t，分别算 normal / zero / shuffle 三种 memory 设置下的 flow loss。

### 7.2 训练结束后诊断 memory 分支
```bash
python scripts/diag_memory.py --run_dir <ckpt或输出目录> --root <DATA> --mem_feature_dir <FEAT> \
  --episodes_file <val.txt> --out diag.json
```
会检查 shuffle 是否真的改变了输入，统计 Z 和 m 在样本之间的离散程度，看 velocity 和 loss 受影响多少，以及 |m| 相对 |t_emb| 的大小。

### 7.3 RoboMME 闭环（官方 test 集）
- **memcache 模型**：用 `robomme_sim/memcache_policy.py`（`MemCacheBuffer` + `MemCacheBatchedPolicy`）。
  - 单帧 prompt 和训练完全一样；
  - 按 dataset 的 `frame_index` 计数，demo 帧也算；
  - 关键帧用 **fp32 ViT** 在线编码，存成 fp16，pool 和窗口选取都和训练共用同样的函数；
  - 不支持 `--pipelined`。
- **用法**：
  ```bash
  E=/mnt/cpfs/heping/SimpleMemVLA-data/outputs/memcache/eval
  # memcache 模型：<EXP> <ckpt step> <gpus> <tag> [额外参数]；固定用 --dataset_split test，只跑 4 个任务
  bash $E/run_eval_memcache.sh E2 10000 4,5,6,7 test4task --episodes_per_task 50 --group_size 6
  # 在训练用过的 train 集 episode 上评测（区分拟合和泛化）
  bash $E/run_eval_memcache.sh E2 10000 4,5,6,7 train10 --dataset_split train \
    --episode_ids_json $E/train_split_trained10.json --group_size 5
  # 发布版 ckpt（原版流程：视频 → sub-task → DiT）
  bash $E/run_eval_released.sh 0,1,2,3 test4task --dataset_split test --episodes_per_task 50 --group_size 5
  ```
- **底层是** `scripts/eval_robomme.sh`（参数来自 `configs/eval/robomme.yaml`）：`execute_horizon 16`，`max_steps 1300`，10 步去噪，`group_size` 默认 17。和训练共用 GPU 时我们用 5–6。
- **运行方式**：每张 GPU 起一个 worker，任务轮流分给各个 worker；4 个任务配 4 张卡时正好一卡一个任务。
- **输出**：`outputs/memcache/eval/<名字>/{eval.log, results.json, videos/}`。每个任务保留一个成功录像和一个失败录像，front 和 wrist 左右拼在一起。
- **时间**：4 个任务 × 50 条 episode，memcache 模型大约 1.5 小时，发布版大约 3 小时。
- `train_split_trained10.json` 里每个任务 10 条训练用过的 episode，跳过了我们自己的 val：
  - PickXtimes：0–4、6–10
  - VideoRepick：0–8、10
  - 其余两个任务：0–9

### 7.4 评测链路已经做过的验证（都通过）
- `outputs/memcache/verify_eval_inputs.py`：
  - 在 train 集 episode 上 reset 之后，当前帧编号正好等于数据集里第一个执行帧的 `frame_index`（VideoUnmaskSwap 是 168，PickXtimes 是 0）；
  - 指令一致；
  - prompt 和 pixel_values 和训练时的 collator 逐位相同；
  - memory 的 K、帧龄一致，cos = 1.000000。
- `outputs/memcache/eval/diag_actions.py`：
  - 在仿真里**回放数据集的真实动作**，第 490 步成功；状态平均误差 0.013 rad；
  - 开环预测误差 0.01–0.02 rad。
- `outputs/memcache/eval/diag_closedloop.py`：逐个决策记录闭环 rollout 和 demo 轨迹的偏差；`codec` 模式会把仿真画面先过一遍 mp4 编解码再送进模型。
- **发布版 ckpt 在同一套评测上**：训练集 100%，test 86%（见 §8），说明评测链路是对的。

---

## 8. 实验历史与结果

### 8.1 作废：E1w / E2w（DiT 热启动，2026-10-24）
- val loss 从 1k 步起就只有约 0.057，最终约 0.033。
- E2w 的 `gapS` 约等于 0。诊断（E2w-diag）显示 memory 被训成了一个和输入无关的常数：Z 的 rel_dev 从初始化时的 0.24 降到 0.0016，m 基本等于输出层的 bias。原因是 DiT 早就见过答案，用不上 memory。
- 结论不可信。输出目录已改名为 `outputs/memcache/robomme_small_E{1,2}_warmdit/`。

### 8.2 公平设定：E1 / E2（原版 Qwen 冻结，DiT 从头训；10-24 16:25 / 16:30 启动，各用 4 卡）

**开环 val loss**（每 1000 步一次）：

| | 约 10k 步 val loss | memory 置零 | memory 打乱 | gapS |
|---|---|---|---|---|
| E1（无 memory） | 0.0560 | — | — | — |
| E2 | **0.0422** | 0.143 | 0.159 | **+0.117** |

这说明在公平设定下，**memory 的内容确实被用上了**。

**完整的开环曲线**（每 1000 步一次 val）：
- E1（已跑完 30k）：val 从 0.163 降下来，在约 15k 步后**进入平台期，0.048–0.050**。**最好 0.0473（20k）**，最终 0.0487（30k）。
- E2（第 20k 步崩溃，记录到 19k）：val 一直在降，**最好 0.0381（18k）**，19k 时 0.0400。**gapS 持续上升**：1k 时 0.0002，2k 时 0.051，10k 时 0.117，19k 时 **0.179**，说明模型越来越依赖 memory 的内容。
- 同一步数下，E2 的 val 比 E1 低约 20%（例如 18k 时 0.0381 vs 0.0500）。

**闭环 @10k**（官方 test 集，4 个任务，每个任务 50 条 episode）：

| 任务 | E1 | E2 | 发布版 · test | 发布版 · train 集用过的 10 条 |
|---|---|---|---|---|
| PickXtimes | 0% | 0% | 100% | 100% |
| VideoUnmaskSwap | 0% | 6% | 90% | 100% |
| VideoRepick | 0% | 0% | 60% | 100% |
| PatternLock | 4% | 0% | 94% | 100% |
| **平均** | **1.0%** | **1.5%** | **86.0%** | **100%** |

- E1/E2 在**训练用过的 40 条 train 集 episode 上也是 0/40**。所以失败不是泛化问题，而是拟合 / 闭环能力不够。
- **失败的样子**：
  - PickXtimes：开局直接去按"结束"按钮，判定失败；
  - 训练用过的 episode：t=16 时和 demo 偏差 0.04 rad，t=32 时 0.24，t=48 时 0.37。夹爪在 t≈112 合上，时机对，但位置偏了约 0.4 rad，夹了个空，之后反复开合直到超时。
- **开环精度不够**：16 步一个 chunk 的关节误差（0.01–0.02 rad）占这段时间关节运动量（0.01–0.07 rad）的 20–100%。
- **视频压缩带来的画面差异可以忽略**：仿真原图和数据集画面输入，预测只差 0.002 rad。
- **结论**：目前这套配方（冻结原版 Qwen + 从头训 DiT + 每个任务 90 条轨迹，10k 步）连基础操作都做不好，闭环成功率**还没法用来衡量 memory**。

**发布版是否用test训练过（2026-10-25勘误）**：
- 尚不能确认或排除；train100%／test86%不是训练成员关系证据。
- 服务器stats/config与公开HF文件哈希一致，stats来源比公开train更大；模型头只有format=pt，未发布训练task/seed清单。
- 新seed对比只能提供间接线索；确认必须取得实际训练清单或可核对的数据来源。当前不认定作弊，亦不宣称发布权重完全无污染风险。

---

## 9. 历史运行快照（2026-10-25 10:10；最新评测见§12）

- **E1 训练：已完成**。05:40 跑完 30000 步，`TRAIN_EXIT=0`。
  - 输出 `outputs/memcache/robomme_small_E1/`：最终模型（`model.safetensors` 加 config 和 stats）、`checkpoint-25000`、`checkpoint-30000`，共 100 GB。
  - `checkpoint-10000` 已经被轮换删除；E1 的 10k 评测是在删除之前跑的。
  - 最终 val 0.0487，最好 0.0473（20k）。**30k 的 checkpoint 还没有做闭环评测。**
- **E2 训练：停在 02:19 的崩溃处，没有续训**。
  - 崩溃原因：第 20000 步做 eval 时，一个 DataLoader worker 报 `LLVM ERROR: pthread_join failed` 后 abort，`TRAIN_EXIT=1`。
  - 输出 `outputs/memcache/robomme_small_E2/`：`checkpoint-10000`、`checkpoint-15000`，共 92 GB；崩溃前的 val 曲线到 19k（最好 0.0381）。
  - 续训：
    ```bash
    bash scripts/launch_memcache.sh E2 4,5,6,7
    ```
    `RESUME=auto` 会从 15000 接着训，必须用 4 卡；续训前先确认有 4 张卡空闲、磁盘也够。
- **GPU：8 张卡全部被别人的任务占着**（每卡 12.7 GB，100% 利用率）。我们这边现在没有任何训练或评测在跑。
- **磁盘：`/mnt/cpfs` 剩 576 GB**。10-25 经用户同意，删掉了作废的 `robomme_small_E{1,2}_warmdit`（释放 192 GB）。它们的日志（`run_E*_warmdit.{log,sh}`、`diag_E2_warmdit.*`）还在，小的元数据文件（`trainer_state.json`、`config.json`、`training_args.bin`、`stats.json`）保存在 `outputs/memcache/warmdit_meta/`。我们的 `outputs/memcache` 现在一共 192 GB（E1 100 GB，E2 92 GB）。
- **代码**：分支 `memcache-singleframe`，HEAD 是 `c1ba78b`；§3 列出的改动（`git status` 显示 26 项）**都还没 commit**。

接手后建议先跑这几条，确认现状：
```bash
df -h /mnt/cpfs; nvidia-smi
ps -eo pid,etime,cmd | grep -E "[t]rain.py|[e]val_success" | cut -c1-120
ls /mnt/cpfs/heping/SimpleMemVLA-data/outputs/memcache/robomme_small_E{1,2}/
cd /mnt/cpfs/heping/SimpleMemVLA && git status --short
```

## 10. 历史下一步建议（新提案已记入§13，不自动执行）

1. **E2 是否续训**（从 15000 步）。需要 4 张空闲的卡，目前都被别人占着；磁盘还要再留出约 90 GB（现在够用）。
2. **低成本的闭环试验**：
   - 评测 E1 的 30k checkpoint（`robomme_small_E1/checkpoint-30000`，或者输出目录里的最终模型），test 集和 train 集各跑一遍；
   - 用 `--execute_horizon 8` 更频繁地重新规划，重跑 40 条 train 集 episode；
   - E2 补跑完后，评测它的 30k checkpoint。
3. **如果闭环还是接近 0，就要改配方，先把基础控制能力提上来**：
   - VLM 用 LoRA 或部分解冻；
   - 用更多数据或全部 16 个任务；
   - 换更小的 DiT 来减少过拟合（现在 train loss 约 0.017，val 约 0.03–0.04）；
   - 训更多步数。

   目标是让 E1 先在不需要记忆的环节上成功，这样 E1 和 E2 的闭环差异才能反映 memory 的作用。
4. **开环层面还可以做**：
   - 在"记忆关键帧"子集（必须靠历史才能决定动作的帧）上分别统计 E1/E2 的 loss 和 gapS；
   - 跑 E3（prefix_tokens）；
   - E2 跑完后用 `diag_memory.py` 检查 memory 有没有塌缩。
5. **LEAK-check**：可在官方test/val和新seed上比较开环误差与sub-task准确率，但只能提供间接线索，不能据此判定训练成员关系；优先追溯训练task/seed清单。

---

## 11. 文件索引（服务器）

```
/mnt/cpfs/heping/SimpleMemVLA/                     代码（分支 memcache-singleframe）
  configs/train/memcache/{base,E1,E2,E3,*_ckptvlm,*_warmdit}.yaml
  scripts/{launch_memcache.sh,train_mem.sh,extract_mem_features.py,make_episode_splits.py,sanity_memcache.py,diag_memory.py}
  robomme_sim/{memcache_policy.py,eval_success.py}
  docs/{memcache_experiment_log.md,memcache_architecture.svg,model_pipeline.md}
/mnt/cpfs/heping/checkpoints/Qwen3.5-4B/           原版 VLM
/mnt/cpfs/heping/SimpleMemVLA-data/
  datasets/yinchenghust/robomme_lerobot/           训练数据（= 官方 train 集）
  splits/robomme_small_{train,val}.txt
  features/robomme_vit_merger_front_qwen35base/    E1–E3 的 cache（fp32，原版 Qwen ViT）
  features/robomme_vit_merger_front/               ckptvlm / warmdit 的 cache
  checkpoints/simplememvla_robomme/                发布版 ckpt
  outputs/memcache/
    robomme_small_E1/                              公平设定下的 E1：已跑完，最终模型 + checkpoint-25000/30000
    robomme_small_E2/                              公平设定下的 E2：停在崩溃处，checkpoint-10000/15000
    warmdit_meta/                                  作废的 E1w/E2w 的小元数据（模型权重已于 10-25 删除）
    run_E1_20261024_162504.log, run_E2_20261024_163054.log
    diag_*.{py,log,json}, vit_*.{py,log}, sanity_*.log, verify_eval_inputs.{py,log}
    eval/
      run_eval_memcache.sh, run_eval_released.sh, train_split_trained10.json
      E{1,2}_step10000_test4task/, E{1,2}_step10000_train10/, E2_step10000_smoke/
      released_test4task/, released_train10/
      diag_actions.py, diag_closedloop.py (+ .log)
```


## 12. 2026-10-25 已完成：发布权重、单帧输入与 latency 对照

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


## 13. 下一轮提案：完整发布权重的单帧后训练与独立 memory（尚未启动）

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
