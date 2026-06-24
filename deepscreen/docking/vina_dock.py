"""
AutoDock Vina 对接模块
======================
封装 Vina 命令行，支持单配体和批量对接
"""

import os
import re
import subprocess
import math
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VINA_EXE, DOCKING


def get_ligand_center(ligand_pdbqt: str) -> tuple[float, float, float]:
    """从配体 PDBQT 计算几何中心"""
    coords = []
    with open(ligand_pdbqt) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
                except ValueError:
                    continue
    if not coords:
        return (0, 0, 0)
    n = len(coords)
    return (sum(c[0] for c in coords) / n,
            sum(c[1] for c in coords) / n,
            sum(c[2] for c in coords) / n)


def get_ligand_extent(ligand_pdbqt: str) -> tuple[float, float, float]:
    """计算配体的空间跨度 (dx, dy, dz)"""
    coords = []
    with open(ligand_pdbqt) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    coords.append((
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ))
                except ValueError:
                    continue
    if not coords:
        return (20, 20, 20)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


def dock_ligand(receptor_pdbqt: str, ligand_pdbqt: str,
                output_pdbqt: str, config_file: Optional[str] = None,
                center: Optional[tuple] = None,
                box_size: Optional[tuple] = None) -> dict:
    """
    单个配体对接

    Args:
        receptor_pdbqt: 受体 PDBQT 路径
        ligand_pdbqt: 配体 PDBQT 路径
        output_pdbqt: 对接结果输出路径
        config_file: 自定义配置文件
        center: 盒子中心 (x, y, z)，None=自动从配体计算
        box_size: 盒子尺寸 (x, y, z)，None=自动从配体计算

    Returns:
        {"affinity": float, "modes": [...], "output": str}
    """
    # 自动计算盒子
    if center is None:
        center = get_ligand_center(ligand_pdbqt)
    if box_size is None:
        extent = get_ligand_extent(ligand_pdbqt)
        padding = DOCKING["box_padding"]
        box_size = (extent[0] + padding, extent[1] + padding, extent[2] + padding)

    # 生成配置文件
    if config_file is None:
        config_file = "vina_config.tmp"

    config_content = f"""receptor = {receptor_pdbqt}
ligand = {ligand_pdbqt}
out = {output_pdbqt}
center_x = {center[0]:.3f}
center_y = {center[1]:.3f}
center_z = {center[2]:.3f}
size_x = {box_size[0]:.1f}
size_y = {box_size[1]:.1f}
size_z = {box_size[2]:.1f}
exhaustiveness = {DOCKING['exhaustiveness']}
num_modes = {DOCKING['num_modes']}
energy_range = {DOCKING['energy_range']}
"""
    with open(config_file, "w") as f:
        f.write(config_content)

    # 运行 Vina
    cmd = [VINA_EXE, "--config", config_file]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # 解析结果
    results = _parse_vina_output(result.stdout)

    if config_file == "vina_config.tmp":
        os.remove(config_file)

    return results


def _parse_vina_output(output: str) -> dict:
    """解析 Vina 输出，提取亲和力和模式信息"""
    modes = []
    best_affinity = None

    # 匹配表格行:    1       -6.605      1.348      1.638
    pattern = re.compile(
        r'^\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)'
    )
    in_table = False
    for line in output.split("\n"):
        if "-----+" in line:
            in_table = True
            continue
        if in_table:
            m = pattern.match(line)
            if m:
                mode = {
                    "mode": int(m.group(1)),
                    "affinity": float(m.group(2)),
                    "rmsd_lb": float(m.group(3)),
                    "rmsd_ub": float(m.group(4)),
                }
                modes.append(mode)
                if best_affinity is None:
                    best_affinity = mode["affinity"]

    return {
        "affinity": best_affinity,
        "num_modes": len(modes),
        "modes": modes,
    }


def batch_dock(receptor_pdbqt: str, ligand_list: list[dict],
               output_dir: str, center: Optional[tuple] = None,
               box_size: Optional[tuple] = None) -> list[dict]:
    """
    批量对接：多个配体依次对接到同一受体

    Args:
        receptor_pdbqt: 受体 PDBQT
        ligand_list: [{"name": ..., "pdbqt": ...}, ...]
        output_dir: 结果输出目录
        center: 盒子中心
        box_size: 盒子尺寸

    Returns:
        ligand_list 附加 affinity 字段
    """
    os.makedirs(output_dir, exist_ok=True)

    # 用第一个配体确定盒子
    if center is None or box_size is None:
        first_pdbqt = next((l["pdbqt"] for l in ligand_list if l["pdbqt"]), None)
        if first_pdbqt:
            if center is None:
                center = get_ligand_center(first_pdbqt)
            if box_size is None:
                extent = get_ligand_extent(first_pdbqt)
                p = DOCKING["box_padding"]
                box_size = (extent[0] + p, extent[1] + p, extent[2] + p)

    print(f"\n  批量对接 {len(ligand_list)} 个配体...")
    print(f"  盒子中心: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    print(f"  盒子尺寸: ({box_size[0]:.1f}, {box_size[1]:.1f}, {box_size[2]:.1f})")

    for i, lig in enumerate(ligand_list):
        if lig["pdbqt"] is None:
            lig["affinity"] = None
            lig["error"] = lig.get("error", "PDBQT 准备失败")
            continue

        name = lig["name"]
        out_path = os.path.join(output_dir, f"{name}_docked.pdbqt")

        try:
            result = dock_ligand(
                receptor_pdbqt, lig["pdbqt"], out_path,
                center=center, box_size=box_size,
            )
            lig["affinity"] = result["affinity"]
            lig["num_modes"] = result["num_modes"]
            lig["docked_output"] = out_path
            lig["modes"] = result["modes"]
            print(f"  {i+1:3d}. {name:20s}  affinity = {result['affinity']:.2f} kcal/mol"
                  if result['affinity'] else f"  {i+1:3d}. {name:20s}  对接失败")
        except Exception as e:
            lig["affinity"] = None
            lig["error"] = str(e)
            print(f"  ✗ {name}: {e}")

    return ligand_list
