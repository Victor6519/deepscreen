"""
DeepScreen 配置文件
===================
AI-Enhanced Virtual Screening Pipeline
"""

import os

# ── 路径 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── 外部工具 ──
VINA_EXE = r"D:\vina\vina.exe"
PYTHON_EXE = r"C:\Users\v\miniconda3\envs\aienv\python.exe"

# ── 对接参数 ──
DOCKING = {
    "exhaustiveness": 8,
    "num_modes": 10,
    "energy_range": 3.0,
    "box_padding": 8.0,        # 配体周围扩展 Å
}

# ── AI 模型参数 ──
MODEL = {
    "fingerprint_radius": 2,
    "fingerprint_bits": 2048,
    "hidden_dims": [1024, 512, 256],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 100,
}

# ── PyMOL 可视化参数 ──
PYMOL = {
    "surface_transparency": 0.4,
    "stick_radius": 0.15,
    "image_width": 1200,
    "image_height": 800,
    "dpi": 300,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
