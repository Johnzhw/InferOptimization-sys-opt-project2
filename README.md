## 关于本仓库
本仓库是一个基于 https://github.com/vllm-project/vllm 仓库的一个fork，用于完成系统优化课程的project2

本仓库实现的优化基于以下测量与观察：

vLLM通过页式注意力与连续批处理显著提升了吞吐，但其请求准入默认采用先来先服务策略，执行层默认将每个解码步组织为单次前向。在消费级单卡环境下，这一默认组合存在两类可量化的效率损失：

1.混合负载下短请求的首token时延呈现重尾，本研究的测量中p99达中位数的十三倍，源于长提示的分块预填充对每步token预算的独占；

2.解码阶段算力利用存在结构性缺口，本研究的测量中投机解码执行组织的每轮GPU空闲占28.1%，源于控制面串行与小批量kernel的低占用。

针对上述两类损失，本仓库分别提出并实现：

1.一个软件层优化：长度分区加权调度器。

2.一个硬件执行层优化：draft模型投机解码及其双流重叠扩展。

并量化了各自的生效条件、收益与代价，并验证两层优化组合时的交互效应。

部分问题发现与瓶颈定位和剖析图如下：

1.混合负载下先来先服务基线的短/长请求首token时延分布（左：1.6请求/秒；右：2.4请求/秒；箱体为四分位，离群点隐藏）

<img width="940" height="420" alt="image" src="https://github.com/user-attachments/assets/68d8d0ae-6fa7-46f7-9a77-d48cb2cd0070" />

2.投机轮时间分解

<img width="940" height="134" alt="image" src="https://github.com/user-attachments/assets/99f76842-262b-45d1-a916-714da44ba361" />

3.引擎核心进程处理器时间构成

<img width="940" height="561" alt="image" src="https://github.com/user-attachments/assets/1d33c8c7-1b84-47a8-a211-746cadf90148" />


## 环境需求

nvcc-12.8/12.9，以及支持nvcc-12.8/12.9的对应cuda版本及驱动版本

target model: Qwen3-8B

draft model: Qwen3-0.6B

## 复现所有实验

请按照 ./复现指南.md进行逐步环境配置，并运行一键复现脚本完成本仓库所有实验的复现。

## 实验报告
本仓库实验报告随邮件一起提交。

## 实验结果速览
<img width="941" height="381" alt="image" src="https://github.com/user-attachments/assets/520c9bd1-220f-43c1-9eb5-a8ceab8a9c2b" />
