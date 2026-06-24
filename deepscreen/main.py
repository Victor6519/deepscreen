#!/usr/bin/env python
"""
DeepScreen — AI-Enhanced Virtual Screening Pipeline
=====================================================
端到端虚拟筛选管线: 配体准备 → 分子对接 → AI 评分 → 可视化

工作流:
  Step 1: 受体前处理   (meeko: PDB → PDBQT)
  Step 2: 配体前处理   (RDKit + meeko: SMILES → PDBQT)
  Step 3: 分子对接     (AutoDock Vina)
  Step 4: AI 亲和力预测 (PyTorch: 分子指纹 → ΔG)
  Step 5: 可视化       (PyMOL: 结合模式渲染)

用法:
  python main.py --receptor 1A42.pdb --ligands data/sample_ligands.smi
  python main.py --receptor 1A42.pdb --ligands data/sample_ligands.smi --train
  python main.py --receptor 1A42.pdb --ligands data/sample_ligands.smi --viz
"""

import os
import sys
import argparse
import json
import time

# Windows UTF-8 编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 确保项目根目录为工作目录 & 添加到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, MODEL_DIR, DATA_DIR


def banner():
    print("""
    ╔══════════════════════════════════════════════════╗
    ║   DeepScreen — AI-Enhanced Virtual Screening     ║
    ║   分子对接 + PyTorch 评分 + PyMOL 可视化         ║
    ╚══════════════════════════════════════════════════╝
    """)


def step1_prepare_receptor(pdb_path: str) -> str:
    """Step 1: 受体前处理"""
    print("\n" + "─" * 55)
    print("  [Step 1/5] 受体前处理 (meeko)")
    print("─" * 55)

    from docking.prepare import prepare_receptor

    receptor_pdbqt = prepare_receptor(pdb_path, "receptor")
    return receptor_pdbqt


def step2_prepare_ligands(smiles_file: str) -> list[dict]:
    """Step 2: 配体库前处理"""
    print("\n" + "─" * 55)
    print("  [Step 2/5] 配体库前处理 (RDKit + meeko)")
    print("─" * 55)

    from docking.prepare import batch_prepare_ligands

    ligand_dir = os.path.join(OUTPUT_DIR, "ligands")
    ligands = batch_prepare_ligands(smiles_file, ligand_dir)
    return ligands


def step3_docking(receptor_pdbqt: str, ligands: list[dict]) -> list[dict]:
    """Step 3: 分子对接"""
    print("\n" + "─" * 55)
    print("  [Step 3/5] 分子对接 (AutoDock Vina)")
    print("─" * 55)

    from docking.vina_dock import batch_dock

    dock_dir = os.path.join(OUTPUT_DIR, "docked")
    ligands = batch_dock(receptor_pdbqt, ligands, dock_dir)

    # 统计
    success = [l for l in ligands if l.get("affinity") is not None]
    if success:
        affinities = [l["affinity"] for l in success]
        print(f"\n  ┌{'─'*50}┐")
        print(f"  │  对接结果统计{'':38s}│")
        print(f"  ├{'─'*50}┤")
        print(f"  │  成功对接: {len(success):3d} / {len(ligands):3d}{'':27s}│")
        print(f"  │  最佳亲和力: {min(affinities):6.2f} kcal/mol{'':25s}│")
        print(f"  │  最差亲和力: {max(affinities):6.2f} kcal/mol{'':25s}│")
        print(f"  │  平均亲和力: {sum(affinities)/len(affinities):6.2f} kcal/mol{'':25s}│")
        print(f"  └{'─'*50}┘")

        # Top 5
        top5 = sorted(success, key=lambda l: l["affinity"])[:5]
        print(f"\n  Top 5 候选分子:")
        for i, lig in enumerate(top5):
            print(f"    {i+1}. {lig['name']:25s}  ΔG = {lig['affinity']:6.2f} kcal/mol")

    return ligands


def step4_ai_scoring(ligands: list[dict], train_model: bool = True):
    """Step 4: AI 亲和力预测"""
    print("\n" + "─" * 55)
    print("  [Step 4/5] AI 亲和力预测 (PyTorch)")
    print("─" * 55)

    from model.train import train_from_docking_results
    from model.affinity_model import (
        AffinityPredictor, MolecularFingerprintEncoder,
        DockingScoreCalibrator,
    )
    from config import MODEL as MODEL_CFG

    # 准备数据
    valid = [l for l in ligands
             if l.get("affinity") is not None and l.get("smiles")]

    if len(valid) < 10:
        print(f"  ⚠ 有效对接结果不足 ({len(valid)} 条)，跳过 AI 训练")
        return ligands

    encoder = MolecularFingerprintEncoder(
        radius=MODEL_CFG["fingerprint_radius"],
        n_bits=MODEL_CFG["fingerprint_bits"],
    )

    if train_model:
        print("\n  训练 AI 亲和力预测模型...")
        trainer = train_from_docking_results(ligands)
        if trainer:
            # AI 预测所有配体
            smiles = [l["smiles"] for l in valid]
            ai_scores = trainer.predict(smiles)

            import torch
            vina_scores = torch.tensor([l["affinity"] for l in valid])
            calibrator = DockingScoreCalibrator()

            names = [l["name"] for l in valid]
            ranked = calibrator.rank_ligands(vina_scores, ai_scores, names)

            print(f"\n  AI+Vina 融合排序 Top 5:")
            for r in ranked[:5]:
                print(
                    f"    {r['rank']}. {r['name']:25s}"
                    f"  Vina={r['vina_score']:6.2f}"
                    f"  AI={r['ai_score']:6.2f}"
                    f"  Final={r['final_score']:6.2f}"
                )

            # 保存排序结果
            result_path = os.path.join(OUTPUT_DIR, "ranking_results.json")
            with open(result_path, "w") as f:
                json.dump(ranked, f, indent=2, ensure_ascii=False)
            print(f"\n  ✓ 排序结果已保存: {result_path}")

    return ligands


def step5_visualization(ligands: list[dict], receptor_pdb: str = None):
    """Step 5: PyMOL 可视化"""
    print("\n" + "─" * 55)
    print("  [Step 5/5] 可视化 (PyMOL)")
    print("─" * 55)

    from viz.pymol_viz import (
        generate_docking_figure,
        generate_interaction_analysis_script,
    )

    viz_dir = os.path.join(OUTPUT_DIR, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    # 只对 Top 3 可视化
    success = sorted(
        [l for l in ligands if l.get("affinity") is not None and l.get("docked_output")],
        key=lambda l: l["affinity"],
    )

    top_n = min(3, len(success))

    for i, lig in enumerate(success[:top_n]):
        print(f"\n  可视化 {i+1}/{top_n}: {lig['name']}")
        try:
            img_path = os.path.join(viz_dir, f"{lig['name']}_binding.png")
            generate_docking_figure(
                receptor_pdb or "1A42.pdb",
                lig["docked_output"],
                img_path,
                ligand_name=lig["name"],
                affinity=lig["affinity"],
            )
        except Exception as e:
            print(f"  ✗ PyMOL 可视化失败: {e}")
            print(f"  → PyMOL 可能未安装，跳过可视化")

    # 生成交互分析脚本
    if success:
        lig = success[0]
        if lig.get("docked_output"):
            script_path = os.path.join(viz_dir, "interactive_analysis.pml")
            try:
                generate_interaction_analysis_script(
                    receptor_pdb or "1A42.pdb",
                    lig["docked_output"],
                    script_path,
                )
            except Exception:
                pass

    return ligands


def save_summary(ligands: list[dict], elapsed: float):
    """保存筛选摘要"""
    summary = {
        "pipeline": "DeepScreen AI-Enhanced Virtual Screening",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "total_ligands": len(ligands),
        "successful_docks": sum(1 for l in ligands if l.get("affinity") is not None),
        "results": [],
    }

    success = sorted(
        [l for l in ligands if l.get("affinity") is not None],
        key=lambda l: l["affinity"],
    )

    for lig in success:
        summary["results"].append({
            "rank": len(summary["results"]) + 1,
            "name": lig["name"],
            "smiles": lig.get("smiles", ""),
            "affinity_kcal_mol": round(lig["affinity"], 3),
            "num_modes": lig.get("num_modes", 0),
        })

    path = os.path.join(OUTPUT_DIR, "screening_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ 筛选摘要: {path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="DeepScreen — AI增强虚拟筛选管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py -r 1A42.pdb -l data/sample_ligands.smi
  python main.py -r 1A42.pdb -l data/sample_ligands.smi --train --viz
  python main.py -r 1A42.pdb -l data/sample_ligands.smi --skip-dock --ai-only
        """,
    )
    parser.add_argument("-r", "--receptor", default="1A42.pdb",
                        help="受体 PDB 文件路径")
    parser.add_argument("-l", "--ligands", default="data/sample_ligands.smi",
                        help="配体 SMILES 文件")
    parser.add_argument("--train", action="store_true",
                        help="训练 AI 评分模型")
    parser.add_argument("--viz", action="store_true",
                        help="生成 PyMOL 可视化")
    parser.add_argument("--ai-only", action="store_true",
                        help="仅用 AI 预测 (跳过对接)")
    parser.add_argument("--skip-dock", action="store_true",
                        help="跳过对接步骤")
    parser.add_argument("-o", "--output", default=OUTPUT_DIR,
                        help="输出目录")
    args = parser.parse_args()

    banner()
    start_time = time.time()

    # ── Step 1: 受体前处理 ──
    receptor_pdbqt = step1_prepare_receptor(args.receptor)

    # ── Step 2: 配体前处理 ──
    ligands = step2_prepare_ligands(args.ligands)

    # ── Step 3: 分子对接 ──
    if not args.skip_dock:
        ligands = step3_docking(receptor_pdbqt, ligands)

    # ── Step 4: AI 评分 ──
    ligands = step4_ai_scoring(ligands, train_model=args.train)

    # ── Step 5: 可视化 ──
    if args.viz:
        ligands = step5_visualization(ligands, args.receptor)

    # ── 保存结果 ──
    elapsed = time.time() - start_time
    summary = save_summary(ligands, elapsed)

    # ── 结束 ──
    print("\n" + "=" * 55)
    print(f"  DeepScreen 筛选完成!")
    print(f"  总耗时: {elapsed:.1f} 秒")
    print(f"  结果目录: {args.output}")
    print("=" * 55)


if __name__ == "__main__":
    main()
