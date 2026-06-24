"""
DeepScreen 模型训练模块
========================
使用对接得分数据训练 AI 亲和力预测模型
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL, MODEL_DIR
from model.affinity_model import AffinityPredictor, MolecularFingerprintEncoder


class Trainer:
    """模型训练器"""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = MODEL["learning_rate"],
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.encoder = MolecularFingerprintEncoder(
            radius=MODEL["fingerprint_radius"],
            n_bits=MODEL["fingerprint_bits"],
        )
        self.history = {"train_loss": [], "val_loss": []}

    def prepare_data(
        self, smiles_list: list[str], scores: list[float],
        val_split: float = 0.2,
    ) -> tuple[DataLoader, DataLoader | None]:
        """
        准备训练/验证数据

        Args:
            smiles_list: SMILES 列表
            scores: 结合亲和力标签
            val_split: 验证集比例
        """
        X = self.encoder.encode_batch(smiles_list)
        y = torch.tensor(scores, dtype=torch.float32)

        # 打乱
        perm = torch.randperm(len(X))
        X, y = X[perm], y[perm]

        # 划分
        n_val = int(len(X) * val_split)
        X_train, y_train = X[n_val:], y[n_val:]
        X_val, y_val = X[:n_val], y[:n_val]

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val) if n_val > 0 else None

        train_dl = DataLoader(train_ds, batch_size=MODEL["batch_size"], shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=MODEL["batch_size"]) if val_ds else None

        return train_dl, val_dl

    def train_epoch(self, train_dl: DataLoader) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0

        for X_batch, y_batch in train_dl:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(X_batch)
            loss = self.criterion(pred, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(X_batch)

        return total_loss / len(train_dl.dataset)

    @torch.no_grad()
    def evaluate(self, val_dl: DataLoader) -> float:
        """评估"""
        self.model.eval()
        total_loss = 0

        for X_batch, y_batch in val_dl:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            pred = self.model(X_batch)
            loss = self.criterion(pred, y_batch)
            total_loss += loss.item() * len(X_batch)

        return total_loss / len(val_dl.dataset)

    def fit(
        self, train_dl: DataLoader, val_dl: DataLoader = None,
        epochs: int = MODEL["epochs"], verbose: bool = True,
    ):
        """完整训练流程"""
        best_loss = float("inf")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dl)
            self.history["train_loss"].append(train_loss)

            if val_dl:
                val_loss = self.evaluate(val_dl)
                self.history["val_loss"].append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save("best_model.pt")

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                msg = f"  Epoch {epoch+1:3d}/{epochs}  train_loss={train_loss:.4f}"
                if val_dl:
                    msg += f"  val_loss={val_loss:.4f}"
                print(msg)

        if not val_dl:
            self.save("best_model.pt")
        print(f"  训练完成! best_loss={best_loss:.4f}")

    def save(self, filename: str):
        path = os.path.join(MODEL_DIR, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }, path)

    def load(self, filename: str):
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            print(f"  模型文件不存在: {path}")
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.history = ckpt.get("history", {"train_loss": [], "val_loss": []})
        print(f"  ✓ 加载模型: {path}")
        return True

    def predict(self, smiles_list: list[str]) -> torch.Tensor:
        """批量预测亲和力"""
        self.model.eval()
        X = self.encoder.encode_batch(smiles_list).to(self.device)
        with torch.no_grad():
            return self.model(X).cpu()


def train_from_docking_results(
    docked_ligands: list[dict],
    model_name: str = "affinity_model",
) -> Trainer:
    """
    从对接结果训练 AI 模型

    适用场景: 对接完成后，用 Vina 得分作为标签训练模型，
    以便后续对更大库做快速 AI 预筛选。
    """
    smiles_list = []
    scores = []

    for lig in docked_ligands:
        if lig.get("affinity") is not None and lig.get("smiles"):
            smiles_list.append(lig["smiles"])
            scores.append(lig["affinity"])

    if len(smiles_list) < 20:
        print(f"  ⚠ 训练数据不足 ({len(smiles_list)} 条)，至少需要 20 条")
        return None

    print(f"\n  训练数据: {len(smiles_list)} 个配体")
    print(f"  亲和力范围: {min(scores):.2f} ~ {max(scores):.2f} kcal/mol")

    model = AffinityPredictor(
        input_dim=MODEL["fingerprint_bits"],
        hidden_dims=MODEL["hidden_dims"],
        dropout=MODEL["dropout"],
    )

    trainer = Trainer(model, device="cpu")
    train_dl, val_dl = trainer.prepare_data(smiles_list, scores)
    trainer.fit(train_dl, val_dl, epochs=MODEL["epochs"])

    return trainer
