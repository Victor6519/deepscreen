# DeepScreen — AI-Enhanced Virtual Screening Pipeline

> **分子对接 + PyTorch 深度学习 + PyMOL 可视化**  
> 端到端的计算机辅助虚拟筛选平台

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2026.03-green)](https://www.rdkit.org/)
[![AutoDock Vina](https://img.shields.io/badge/AutoDock_Vina-1.2.7-orange)](https://github.com/ccsb-scripps/AutoDock-Vina)
[![PyMOL](https://img.shields.io/badge/PyMOL-Open_Source-purple)](https://pymol.org/)

---

##  项目概述

DeepScreen 是一个**全自动化的虚拟筛选管线**，将传统分子对接与 AI 深度学习相结合，用于从化合物库中快速发现潜在的药物先导化合物。

### 核心能力

```
SMILES 化合物库 ──→ [RDKit 前处理] ──→ [Vina 对接] ──→ [PyTorch AI 评分] ──→ [PyMOL 可视化]
                                                              │
                                                        融合排序报告
```

| 模块 | 工具 | 功能 |
|------|------|------|
| **分子前处理** | RDKit + meeko | SMILES → 3D 构象 → PDBQT (自动 Gasteiger 电荷、原子类型) |
| **受体准备** | meeko | PDB → 加氢 → 合并非极性 H → PDBQT |
| **分子对接** | AutoDock Vina 1.2.7 | 批量配体对接，自动盒子计算 |
| **AI 评分** | PyTorch | Morgan 指纹 → 深度神经网络 → 亲和力预测 |
| **可视化** | PyMOL API | 蛋白卡通 + 表面 + 配体球棍 + 氢键标注 → 高分辨率 PNG |
| **结果融合** | 集成模块 | Vina 得分 + AI 预测 → 加权融合排序 |

---

##  快速开始

### 环境要求

```bash
# conda 环境 (推荐)
conda create -n deepscreen python=3.11 -y
conda activate deepscreen

# 核心依赖
conda install -c conda-forge rdkit pymol-open-source -y
pip install torch meeko gemmi

# AutoDock Vina (Windows)
# 从 https://github.com/ccsb-scripps/AutoDock-Vina/releases 下载
# 放置到 PATH 或修改 config.py 中的 VINA_EXE 路径
```

### 一键运行

```powershell
cd D:\vsproject\deepscreen
$env:PYTHONUTF8 = "1"

# 完整管线: 对接 + AI 训练 + 可视化
python main.py -r 1A42.pdb -l data/sample_ligands.smi --train --viz

# 仅对接 (快速模式)
python main.py -r 1A42.pdb -l data/sample_ligands.smi
```

### 自定义化合物库

创建自己的 SMILES 文件 (格式: `SMILES NAME`)：

```text
# my_ligands.smi
CC(=O)OC1=CC=CC=C1C(=O)O Aspirin
CC(C)CC1=CC=C(C=C1)C(C)C(=O)O Ibuprofen
CN1C=NC2=C1C(=O)N(C(=O)N2C)C Caffeine
```

---

##  项目结构

```
deepscreen/
├── main.py                      # 主入口: 端到端管线
├── config.py                    # 全局配置 (工具路径、超参数)
├── README.md
│
├── docking/                     # 对接模块
│   ├── __init__.py
│   ├── prepare.py               # meeko 受体/配体 PDBQT 自动制备
│   └── vina_dock.py             # Vina 对接封装 & 批量调度
│
├── model/                       # AI 模块
│   ├── __init__.py
│   ├── affinity_model.py        # 网络架构 + 分子指纹编码器
│   └── train.py                 # 训练/验证/预测流程
│
├── viz/                         # 可视化模块
│   ├── __init__.py
│   └── pymol_viz.py             # PyMOL Python API 自动渲染
│
├── utils/                       # 工具函数
│   └── __init__.py
│
├── data/                        # 示例数据
│   └── sample_ligands.smi       # 25 个 HIV-1 蛋白酶抑制剂类似物
│
└── output/                      # 输出结果
    ├── ligands/                 # 配体 PDBQT 文件
    ├── docked/                  # 对接结果 (多构象 PDBQT)
    ├── viz/                     # PyMOL 渲染图 (PNG)
    ├── models/                  # 训练好的 PyTorch 模型
    ├── screening_summary.json   # 筛选结果 (JSON)
    └── ranking_results.json     # AI+Vina 融合排序
```

---

##  案例演示: HIV-1 蛋白酶抑制剂虚拟筛选

### 靶点

| 项目 | 值 |
|------|-----|
| PDB ID | [1A42](https://www.rcsb.org/structure/1A42) |
| 蛋白 | HIV-1 蛋白酶 (同源二聚体, 198 残基/链) |
| 共晶配体 | BZU (环脲类抑制剂) |
| 分辨率 | 2.0 Å |

### 配体库

25 个化合物，包括：
- 已知 HIV-1 蛋白酶抑制剂骨架 (Amprenavir, Darunavir, Atazanavir, Saquinavir 等)
- 环脲类/磺酰胺类/异噁唑类衍生物
- 小分子药物类似物 (Paracetamol, Oxindole 等)

### 对接参数

| 参数 | 值 |
|------|-----|
| 搜索空间 | 22×22×22 Å³ (覆盖活性位点) |
| Exhaustiveness | 8 |
| 输出构象数 | 10 / 配体 |

### 筛选结果 (Top 10)

| Rank | 配体 | Vina ΔG (kcal/mol) |
|------|------|---------------------|
| **1** | **Cyclohexyl_urea** | **-6.75** |
| 2 | Phenylbenzamide | -6.66 |
| 3 | Cyanophenyl_inhibitor | -6.61 |
| 4 | Urea_inhibitor_1 | -6.60 |
| 5 | Glycoside_inhibitor | -6.54 |
| 6 | Saquinavir_analog | -6.51 |
| 7 | Atazanavir_core | -6.42 |
| 8 | Sulfonamide_inhibitor | -6.28 |
| 9 | Nelfinavir_core | -6.25 |
| 10 | Benzamide_phenyl | -6.24 |

### AI 模型架构

```
Input (2048-bit Morgan FP, r=2)
  → BatchNorm
  → Dense(1024) + ReLU + Dropout(0.3)
  → Dense(512)  + ReLU + Dropout(0.3)
  → Dense(256)  + ReLU + Dropout(0.3)
  → Dense(1)    # ΔG 预测
```

> **注意**: 当前演示用 24 个对接结果训练，数据量不足以泛化。实际应用中应使用 PDBbind 等大规模数据集 (数千~数万个蛋白-配体复合物) 训练以获得可靠的 AI 评分。

### 可视化示例

Top 3 配体的结合模式三维渲染图位于 `output/viz/`：

- `Cyclohexyl_urea_binding.png` (~480 KB)
- `Phenylbenzamide_binding.png` (~465 KB)
- `Cyanophenyl_inhibitor_binding.png` (~486 KB)

每张图包含: 蛋白卡通模型 → 结合位点表面 → 配体球棍模型 → 氢键虚线标注

---

## ⚙️ 配置说明

编辑 `config.py` 调整参数：

```python
# 对接参数
DOCKING = {
    "exhaustiveness": 8,    # 搜索穷举度 (8=标准, 16=高精度)
    "num_modes": 10,        # 输出构象数
    "box_padding": 8.0,     # 对接盒子扩展 (Å)
}

# AI 模型参数
MODEL = {
    "fingerprint_radius": 2,  # Morgan 指纹半径
    "fingerprint_bits": 2048,  # 指纹位数
    "hidden_dims": [1024, 512, 256],  # 隐藏层维度
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "epochs": 100,
}
```

---

##  命令行参考

```
usage: python main.py [-h] [-r RECEPTOR] [-l LIGANDS]
                      [--train] [--viz] [--ai-only] [--skip-dock]

选项:
  -r, --receptor  受体 PDB 文件 (默认: 1A42.pdb)
  -l, --ligands   配体 SMILES 文件 (默认: data/sample_ligands.smi)
  --train         训练 AI 评分模型
  --viz           生成 PyMOL 可视化
  --ai-only       仅 AI 预测 (跳过对接)
  --skip-dock     跳过对接步骤 (使用已有结果)
  -o, --output    输出目录
```

---

##  技术依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | ≥3.11 | 运行环境 |
| PyTorch | ≥2.5 | 深度学习框架 |
| RDKit | ≥2024 | 化学信息学 |
| AutoDock Vina | 1.2.7 | 分子对接引擎 |
| meeko | ≥0.5 | PDBQT 前处理 |
| PyMOL | Open Source | 三维可视化 |
| NumPy | — | 数值计算 |
| Pandas | — | 数据处理 |

---

##  引用

如果此项目对你的研究有帮助，请引用以下工作：

- **AutoDock Vina**: Eberhardt et al., *J. Chem. Inf. Model.* (2021) [DOI: 10.1021/acs.jcim.1c00203](https://doi.org/10.1021/acs.jcim.1c00203)
- **RDKit**: [https://www.rdkit.org/](https://www.rdkit.org/)
- **meeko**: [https://github.com/forlilab/Meeko](https://github.com/forlilab/Meeko)
- **PyMOL**: Schrödinger, LLC

---

##  许可

MIT License — 仅用于学术研究和教育目的。
