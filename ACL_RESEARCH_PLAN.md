# BeyondStyle：ACL 主线与完成清单

更新日期：2026-07-29  
论文标题：**Machine–Human Realization Signatures from Surface Form to Hidden Response**

## 1. 唯一研究问题

在命题内容受到近似控制时：

1. 人类与机器 continuation 还存在哪些稳定的词汇和句法实现差异？
2. 这些显式 surface measurements 能否跨 generator 映射到两个模型家族的 hidden states？
3. 沿同一 hidden route 干预时，固定 token 的内部响应与自由生成的表层响应是否同向？

论文不把 detector 分数、人类“AI 味”判断、LM-eval 能力变化或中文复现作为成立条件。

## 2. 研究对象与边界

研究对象由两个互补 interface 共同界定：

> **Machine–human realization signature under matched continuation
> contexts**

proposition-conditioned 主语料包含 500 个 English human–machine pairs；
planner 从 human continuation 抽取 proposition plan，generator 只看到共享
prefix 和 plan。匹配的 prefix-only control 另生成 500 条 continuation，
保留 sample、prefix、human reference、generator 与目标长度，但不给 plan、
entity list、endpoint 或 human text。

前者增强 semantic alignment，后者移除不对称 planning interface。
两臂方向一致支持 recurrent realization contrast，但任何一臂都不能单独识别
content-invariant authorship factor。不声称 proposition-equivalent、universal
authorship axis 或自动指标等价于人类感知。

## 3. Surface construct

冻结 35 个自动语言指标：

- lexical：10 个；
- syntactic：12 个；
- discourse：13 个；
- nuisance：长度、实体、数字等 7 个。

先在训练数据中回归 nuisance，再定义 group-level machine-minus-human direction：

\[
u_g=\mathbb E[f_g(x^{M})-f_g(x^{H})],
\qquad
s_g(x)=f_g(x)^\top u_g.
\]

只有跨 generator 和跨 domain 稳定的 lexical、syntactic groups 进入 confirmatory hidden analysis。discourse 只作为 off-target measurement。

## 4. Hidden readout

正文只使用 continuation mean 和 nullspace-constrained states。

目标模型：

- Llama 3.1 8B Instruct：intervention L7，readout anchor L32；
- Qwen3 4B：intervention L8，readout anchor L36。

对每个 held-out generator，在其余三个 generator 上拟合 surface-score readout，并分别在 human 与 machine states 上检验。四个 generator environment 是推断单位。

## 5. Canonical causal pullback

对 downstream machine-oriented readout \(v_g^k\)，在 early layer 构造：

\[
w_g=
\operatorname{norm}\left[
\operatorname{consensus}
\left(
\overline{PJ^\top v_g^k}^{\,H},
\overline{PJ^\top v_g^k}^{\,M}
\right)
\right].
\]

正方向只由 machine-oriented readout 决定：

- 不使用 development generation 翻转符号；
- 不根据 sealed prompts 选层；
- lexical 与 syntactic 使用同一 declared final-layer endpoint；
- prompt prefill 不受干预，hook 只作用于 decoding token。

## 6. 严格 pipeline-matched null

普通随机向量不能控制 \(J^\top v\) 的构造优势。主 hidden null 必须：

1. 在 corpus × generator strata 内置换 surface target；
2. 保留真实 hidden-side nuisance 结构；
3. 将 surface features 与其配套 nuisance 一起置换；
4. 重新拟合每一层 cross-source readout；
5. 经过同一 Jacobian pullback；
6. 在同一固定 token sequence 上干预。

主指标为符号不变的 transport capacity：

\[
C_{ig}=|\tau_{ig}|,
\qquad
\tau_{ig}=
\frac{
(h_i^k(+\alpha w)-h_i^k(-\alpha w))^\top v_g^k
}{
2\alpha\lVert w\rVert_2
}.
\]

construction endpoint 与 direct injection layer 都只作内部诊断。主传播证据
来自二者之间的五个 sampled layers。

最终 confirmatory audit 使用 100 条固定轨迹与 100 个完整 matched-null
pipelines，并通过逐 prompt token hash 验证 learned 与 null 的轨迹完全一致。
以五个 intermediate layers 的 depth-normalized curve AUC 为全局统计量：

- Llama lexical：.368 vs null .201（range .123--.289），\(p=.0099\)；
- Qwen lexical：.315 vs .214（.122--.297），\(p=.0099\)；
- Llama syntactic：.255 vs .202，\(p=.1089\)；
- Qwen syntactic：.253 vs .225，\(p=.1980\)。

因此跨架构成立的是 lexical hidden route；syntax 虽可读、局部响应为正，但
不通过同一 curve-level null。

## 7. Free-generation audit

对 166 个 sealed prompts、同 seed、128 new tokens，比较 \(-1\) 与 \(+1\)。

必须同时报告完整 \(2\times3\) response matrix：

- interventions：lexical、syntactic；
- outcomes：lexical、syntactic、discourse。

component selectivity 定义为：

\[
b_{ig}
=p_{ig}
-\frac12\sum_{c\ne g}p_{ic}.
\]

行为校准使用八个正交 random axes。由于 random pole 没有语义，比较 axis-level absolute means；不从 learned effect 中减去某一个随机方向。

sealed full response matrix 及八个 random axes 使用一组 paired seed；
另对 learned axes 在同一 166 prompts 上运行三组独立 paired seeds，并先在
prompt 内平均种子再 bootstrap prompts。

最终 learned-axis 结果：

- lexical selectivity 在两个模型中均未可靠偏离 0；
- 原 one-seed Qwen syntactic inversion 不跨三种子复制，三组 target means
  为 -0.245 / -0.064 / +0.384，clustered mean 为
  .025[-.189,.295]；
- Llama syntactic target projection 在三种子平均后为
  .249[.043,.518]，但 selectivity 为 .152[-.092,.445]；
- 四个 model×component selectivity intervals 全部跨 0；
- full random-axis ranks、完整 response matrix 与 guardrails 已完成。

因此 one-seed inversion 应解释为 autoregressive sampling sensitivity，而不
是稳定反向控制。hidden trace 与自由生成 selectivity 仍形成清晰的
representation--realization gap，但不能称为已直接测得 behavior-gradient
alignment。

## 8. Content 与 quality guardrails

仅保留与主结果直接相关的自动 guardrails：

- prompt content recall / precision；
- entity 与 number recall；
- paired-output semantic cosine；
- word-count shift；
- fragment rate；
- repeated-trigram rate。

guardrails 用来排除 wholesale collapse，不证明 semantic 或 pragmatic equivalence。

## 9. 三张主图

1. **Surface construct atlas**  
   paired construction + feature forest + generator/domain stability。

2. **Hidden causal trace**  
   canonical readout→pullback pipeline + learned vs strict matched-null layer traces。

3. **Representation-to-realization audit**  
   cross-generator readout + five-layer propagation matched-null fingerprint +
   full one-seed response matrix + prompt distribution / clustered three-seed
   interval / seed-specific means。

统一使用明亮浅蓝与浅粉配色，不使用全局大标题，不拆成大量低信息图。

## 10. 论文主张

允许的主张：

- proposition-plan protocol 下存在稳定、分布式的 lexical/syntactic realization signature；
- surface scores 在两个模型中可跨 generator 线性读取；
- canonical pullback 在多个 intermediate propagation 层上产生超过 pipeline-matched null 的 fixed-token response；
- hidden readout direction 与 generated-language direction 不能视为同一对象；
- fixed-token hidden response 不保证跨模型可移植的 generated-language control。

禁止的主张：

- 定义了人类感知的“AI 味”；
- 找到 universal AI direction；
- proposition plan 消除了全部 content confound；
- readout–behavior alignment 已被直接估计；
- steering 提升通用能力；
- detector 响应验证了语言构念。

## 11. 已完成工程与证据项

- [x] 完成两模型 × lexical/syntactic/8 random axes 自由生成；
- [x] 生成 full response matrix、behavior-null ranks 和最终 guardrails；
- [x] 用严格 nuisance-preserving 版本重跑两模型 matched null；
- [x] 刷新 Figure 2、Figure 3 和全部论文数字；
- [x] 删除绘图脚本中的旧 coupling / single-random Figure 3 代码；
- [x] 运行全部 unit tests（143 passed，6 subtests passed）；
- [x] 连续编译 LaTeX，逐页检查 25 页完整 PDF；
- [x] 自动生成 35 项 feature、24 项 environment holdout、16 项 readout
  fold、28 项 layer trace、12 项 response matrix、8 条 behavior null 与完整
  guardrails；
- [x] 修正 injection layer 被误计为 propagation evidence 的口径；
- [x] 完成 321 unflagged / 179 flagged 的 plan-serialization sensitivity：
  lexical 与 syntactic 子集方向 cosine 分别为 0.976 与 0.966，主要效应不只
  集中在 flagged pairs；
- [x] 新一轮独立 ACL 审稿为 2/5：表格可审查性已补齐，但 construct
  asymmetry、12 个 fixed-token prompts、8 个 matched permutations 仍是
  acceptance blockers；
- [x] 实现 100-prompt / 100-pipeline matched-null runner、propagation-only
  depth-normalized curve AUC 和 finite-null \(p=(b+1)/(R+1)\)；
- [x] 实现同轨迹 pullback / encoding / raw mean difference / nullspace mean
  difference / 8 projected-random 消融；
- [x] 将 \(-1/+1\) 与四个 direction 同批 forward，带 CUDA OOM 自动拆分和逐
  prompt 原子断点续跑；
- [x] 完成 prefix-only control 的 prompt 泄漏测试与四-generator 真实 API
  smoke（12/12 accepted）；
- [x] 完成 500 条 matched prefix-only English continuations：样本 ID、prefix、
  human reference、source/domain 与 generator assignment 均逐条匹配，生成端只
  收到 prefix 与标量长度提示；
- [x] 完成 generator-leave-one-out 的表层构念迁移：每个轴只由其余三个
  generator 的 proposition-aligned 样本拟合，测试样本及其 human reference
  均不参与拟合；lexical / syntactic shift cosine 为 0.885 / 0.869，冻结轴正向
  pair 比例为 88.0% / 79.6%；
- [x] 完成独立 frozen-encoder semantic audit：proposition-conditioned
  human/machine cosine .925[.922,.929]，prefix-only .727[.720,.734]，
  deranged baseline .469；
- [x] 完成 prefix-only hidden transfer：proposition-conditioned 三个
  generator 拟合、held-out generator plan-free 测试；Llama lexical/syntactic
  mean \(r=.470/.421\)，Qwen \(r=.544/.451\)；
- [x] 完成两模型 100 prompts × 100 complete matched-null pipelines；
- [x] 完成 pullback / encoding / raw source / nullspace source /
  8 projected-random 同轨迹消融；
- [x] 完成 nullspace rank \(k=0,5,10,20,50\) 与 fit pairs
  \(n=8,16,24,32\) 方向估计敏感性；\(k\) sweep cosine .919--.996，
  \(n=16\) cosine .967--.982；
- [x] 完成两模型 × 两 learned axes × 166 prompts × 三 paired decode seeds，
  并按 prompt 聚类推断；
- [x] Figure 3D 改为 prompt-mean density + clustered CI + three seed means；
- [x] 完成 95% 方差 PCA 截断审计：lexical/syntactic 方向 cosine 为
  .994/1.000，prefix-only 投影保持正向；
- [x] 完成 pairwise selectivity 和 top-3 原始语言学效应换算；
- [x] 记录现有 confirmatory stages 的 3.345 GPU-process hours；
- [x] 全量单元测试通过；LaTeX 无 undefined reference、overfull box 或空白页。

## 12. 下一步

当前只保留三个直接回应审稿意见的 GPU 任务，不增加语言、benchmark、
大规模 strength grid 或 70B 模型：

1. encoding pattern 与 nullspace source difference 的同协议三种子行为基线；
2. greedy / temperature 1.0 解码稳健性；
3. Gemma 4 E2B 的 prefix-only transfer、100-pipeline hidden null 与三种子行为复现。

上述任务由 `scripts/run_server_revision_bundle.sh` 一次调度；完成后只更新
表格、主张与 limitations，并进行最后一轮严格 ACL reviewer pass。

## 13. 已完成运行

- canonical learned axes + 8 behavior-null axes：完成；
- 两模型严格 nuisance-preserving matched null：完成。
- prefix-only 真实 API smoke（四 generator）：完成。
- 500 条 matched prefix-only English continuations：完成。
- prefix-only 词汇/句法特征提取与 generator-leave-one-out 表层迁移：完成。

既有主实验已经完成并同步本地；审稿补充实验尚待服务器启动。

日志：

- `outputs/en/05_linguistic/causal_pullback/logs/canonical_behavior_ensemble.log`
- `outputs/en/05_linguistic/causal_pullback/logs/strict_matched_null_rerun.log`
