"""
受体 & 配体前处理模块
=====================
使用 meeko + RDKit 自动化 PDBQT 制备流程
支持: PDB 受体前处理 + SMILES 配体库批量前处理
"""

import os
import sys
import subprocess

# meeko CLI
MK_PREPARE_RECEPTOR = os.path.join(
    os.path.dirname(sys.executable), "Scripts", "mk_prepare_receptor.exe"
)


def clean_pdb_for_meeko(pdb_path: str, output_path: str = None) -> str:
    """
    清洗 PDB：移除 meeko 无法处理的内容
    - Hg (汞), Zn (锌) 等非标准金属
    - 结晶水 (HOH)
    - 不标准的 HETATM
    """
    if output_path is None:
        output_path = pdb_path.replace(".pdb", "_clean.pdb")

    # 要排除的元素和残基
    excluded_elements = {"HG", "CD", "PT", "AU", "AG", "ZN"}
    excluded_residues = {"HOH", "WAT"}

    kept = []
    removed = set()
    with open(pdb_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("ATOM"):
                kept.append(line)
            elif line.startswith("HETATM"):
                resname = line[17:20].strip()
                element = line[76:78].strip() if len(line) >= 78 else ""
                if element in excluded_elements:
                    removed.add(f"{resname}({element})")
                    continue
                if resname in excluded_residues:
                    removed.add(resname)
                    continue
                kept.append(line)
            elif line.startswith("TER") or line.startswith("END"):
                kept.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line)
        if not kept[-1].startswith("END"):
            f.write("END\n")

    if removed:
        print(f"  [清洗] 移除了: {', '.join(sorted(removed))}")

    return output_path


def prepare_receptor(pdb_path: str, output_basename: str = "receptor",
                     allow_bad_res: bool = True) -> str:
    """
    使用 meeko mk_prepare_receptor 将 PDB 转换为 PDBQT 受体

    Args:
        pdb_path: 受体 PDB 文件路径
        output_basename: 输出文件基名
        allow_bad_res: 是否跳过模板匹配失败的残基

    Returns:
        生成的 .pdbqt 文件路径
    """
    # 先清洗 PDB
    clean_pdb = clean_pdb_for_meeko(pdb_path)

    output_pdbqt = output_basename + ".pdbqt"

    cmd = [MK_PREPARE_RECEPTOR, "--read_pdb", clean_pdb,
           "-o", output_basename, "-p"]
    if allow_bad_res:
        cmd.append("-a")

    print(f"  [准备受体] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if not os.path.exists(output_pdbqt):
        raise RuntimeError(
            f"受体 PDBQT 生成失败:\n{result.stdout}\n{result.stderr}"
        )

    size_kb = os.path.getsize(output_pdbqt) / 1024
    print(f"  ✓ 受体 PDBQT: {output_pdbqt} ({size_kb:.1f} KB)")
    return output_pdbqt


def prepare_ligand_from_smiles(smiles: str, name: str = "ligand",
                                output_dir: str = ".") -> str:
    """
    从 SMILES 生成配体 PDBQT (meeko API)

    Args:
        smiles: 配体 SMILES 字符串
        name: 配体名称
        output_dir: 输出目录

    Returns:
        生成的 .pdbqt 文件路径
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    # 1. RDKit 分子构建
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无效的 SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    # 2. meeko 前处理
    prep = MoleculePreparation(
        merge_these_atom_types=["H"],
        hydrate=False,
        flexible_amides=False,
        charge_model="gasteiger",
    )
    setups = prep.prepare(mol)

    if not setups:
        raise RuntimeError(f"meeko 处理失败: {smiles}")

    pdbqt_str, ok, msg = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        print(f"  ⚠ meeko 警告: {msg}")

    # 3. 写入文件
    output_path = os.path.join(output_dir, f"{name}.pdbqt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pdbqt_str)

    num_torsions = pdbqt_str.count("BRANCH")
    return output_path


def batch_prepare_ligands(smiles_file: str, output_dir: str) -> list[dict]:
    """
    批量处理 SMILES 配体库 → PDBQT

    Args:
        smiles_file: SMILES 文件 (每行: SMILES NAME)
        output_dir: PDBQT 输出目录

    Returns:
        [{"name": ..., "smiles": ..., "pdbqt": ..., "torsions": ...}, ...]
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    with open(smiles_file) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    print(f"\n  批量前处理 {len(lines)} 个配体...")
    for i, line in enumerate(lines):
        parts = line.split()
        smiles = parts[0]
        name = parts[1] if len(parts) > 1 else f"ligand_{i:03d}"

        try:
            pdbqt_path = prepare_ligand_from_smiles(
                smiles, name, output_dir
            )
            results.append({
                "name": name,
                "smiles": smiles,
                "pdbqt": pdbqt_path,
            })
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results.append({
                "name": name,
                "smiles": smiles,
                "pdbqt": None,
                "error": str(e),
            })

    success = sum(1 for r in results if r["pdbqt"])
    print(f"  ✓ 成功: {success}/{len(results)}")
    return results
