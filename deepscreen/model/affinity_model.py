"""
DeepScreen 结合亲和力预测模型
===============================
基于分子指纹 + 深度神经网络的蛋白质-配体结合亲和力预测
架构: Morgan Fingerprint → MLP (Multi-Layer Perceptron)

参考: PDBbind 数据集标准, 使用 RDKit Morgan 指纹作为分子表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AffinityPredictor(nn.Module):
    """
    分子指纹 → 结合亲和力 (ΔG / kcal/mol) 预测模型

    架构:
        Input (2048-bit Morgan FP)
        → BatchNorm
        → Dense(1024) + ReLU + Dropout
        → Dense(512) + ReLU + Dropout
        → Dense(256) + ReLU + Dropout
        → Dense(1)  # 输出标量: 预测的 ΔG

    特点:
        - 批归一化加速收敛
        - Dropout 防过拟合 (适合小数据集)
        - 残差连接思想 (可扩展到 GNN)
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dims: list[int] = [1024, 512, 256],
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        # 输入归一化
        self.bn_input = nn.BatchNorm1d(input_dim)

        # 隐藏层
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.fc_layers = nn.Sequential(*layers)

        # 输出层
        self.output = nn.Linear(in_dim, 1)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.bn_input(x)
        x = self.fc_layers(x)
        return self.output(x).squeeze(-1)


class MolecularFingerprintEncoder:
    """
    分子指纹编码器: SMILES → Morgan Fingerprint (bit vector)
    """

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.radius = radius
        self.n_bits = n_bits

    def encode(self, smiles: str) -> torch.Tensor:
        """单个 SMILES → 指纹张量"""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(self.n_bits)

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.radius, nBits=self.n_bits
        )
        bits = [float(c) for c in fp.ToBitString()]
        return torch.tensor(bits, dtype=torch.float32)

    def encode_batch(self, smiles_list: list[str]) -> torch.Tensor:
        """批量 SMILES → 指纹张量 [batch_size, n_bits]"""
        fps = [self.encode(s) for s in smiles_list]
        return torch.stack(fps)


class DockingScoreCalibrator:
    """
    对接得分校准器

    用途: 将 Vina 对接得分与 AI 预测进行融合
    - Vina 得分: 基于力场的经验打分函数 (快速)
    - AI 预测: 基于数据驱动的亲和力预测 (可学习系统偏差)

    融合策略: score_final = w1 * score_vina + w2 * score_ai
    """

    def __init__(self):
        self.vina_weight = 0.6
        self.ai_weight = 0.4

    def calibrate(self, vina_score: torch.Tensor,
                  ai_score: torch.Tensor) -> torch.Tensor:
        return self.vina_weight * vina_score + self.ai_weight * ai_score

    def rank_ligands(
        self,
        vina_scores: torch.Tensor,
        ai_scores: torch.Tensor,
        names: list[str],
    ) -> list[dict]:
        """
        融合打分 → 排序配体列表
        """
        final_scores = self.calibrate(vina_scores, ai_scores)

        ranked = sorted(
            zip(names, vina_scores, ai_scores, final_scores),
            key=lambda x: x[3],
        )

        return [
            {
                "rank": i + 1,
                "name": name,
                "vina_score": round(vs.item(), 3),
                "ai_score": round(ais.item(), 3),
                "final_score": round(fs.item(), 3),
            }
            for i, (name, vs, ais, fs) in enumerate(ranked)
        ]
