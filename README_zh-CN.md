# VASNet 期刊版本代码

论文：*Volumetric Anatomy-aware Structural Modeling for Perinatal Brain Tissue Segmentation*。

本仓库以本机实际实验使用的 `models/MambaUXV32.py` 为基础整理，保留模型的参数名称与结构，包含 RASE、GSA、SCA 模块及训练、预处理、预测、评估入口。不是仅对会议版代码改名。

完整命令见 [英文 README](README.md)。建议先阅读 [实现对应关系](docs/IMPLEMENTATION.md) 和 [复现说明](docs/REPRODUCIBILITY.md)：

- 论文使用 VAS-Net，仓库及 Python 类使用 VASNet，目前按期刊投稿状态描述。
- 论文写 Adam + cosine，本机脚本为 SGD + polynomial；提供两个独立配置，不能混称为同一实验。
- 原模型评估时仍随机选择扫描方向；新版默认固定方向，`--scan legacy_random` 可使用原有行为。
- 输入 patch 为 128³，训练 batch size 至少为 2。完整模型需要 Linux/CUDA 与真实 Mamba 依赖。
- 未包含病例数据、原始患者划分或已训练 V32 权重；本机未找到可核实的 V32 权重文件。
- CPU 检查不能证明完整 CUDA 模型已跑通，也不代表重新复现了论文结果。

原始 ZIP、本机实验源码均未修改。发布前还需作者确认开源许可证、最终实验协议及数据/权重可公开范围。
