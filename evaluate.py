import os
import argparse
import random
import ast
import json

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit import DataStructs


_PROP_ALIASES = {
    "logp": "MolLogP",
    "tpsa": "TPSA",
    "hba": "NumHAcceptors",
    "hbd": "NumHDonors",
    "molwt": "MolWt",
}


def _load_yaml(path: str) -> dict:
    raw = yaml.safe_load(open(path, "r", encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a dict, got {type(raw)}")
    return raw


def _normalize_prop_name(name: str) -> str:
    key = str(name).strip()
    if not key:
        return key
    lower = key.lower()
    return _PROP_ALIASES.get(lower, key)


def _normalize_task_props(taskid_prop: dict[str, list[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for tid, props in taskid_prop.items():
        out[str(tid)] = [_normalize_prop_name(p) for p in props]
    return out


def _parse_smiles_list(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(v) for v in raw if str(v)]
    if raw is None:
        return []
    try:
        if isinstance(raw, float) and pd.isna(raw):
            return []
    except Exception:
        pass
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            val = parser(s)
        except Exception:
            continue
        if isinstance(val, list):
            return [str(v) for v in val if str(v)]
    return []


def _read_pred_series(path: str) -> pd.Series:
    df = pd.read_csv(path)
    if "pred_smiles_list_json" in df.columns:
        return df["pred_smiles_list_json"]
    if "smiles_list" in df.columns:
        return df["smiles_list"]
    if df.shape[1] == 1:
        col = df.columns[0]
        if isinstance(col, str):
            col_str = col.strip()
            if col_str.startswith("[") or col_str.startswith("\"[") or col_str.startswith("'["):
                df = pd.read_csv(path, header=None)
                return df.iloc[:, 0]
        return df.iloc[:, 0]

    df = pd.read_csv(path, header=None)
    if df.shape[1] == 1:
        series = df.iloc[:, 0]
        if len(series) > 0 and str(series.iloc[0]).strip() in ("pred_smiles_list_json", "smiles_list"):
            return series.iloc[1:].reset_index(drop=True)
        return series

    df = pd.read_csv(path, header=None, sep="\t")
    if df.shape[1] == 1:
        series = df.iloc[:, 0]
        if len(series) > 0 and str(series.iloc[0]).strip() in ("pred_smiles_list_json", "smiles_list"):
            return series.iloc[1:].reset_index(drop=True)
        return series

    raise ValueError(f"Unexpected columns in {path}: {list(df.columns)}")


def _resolve_mol_col(df: pd.DataFrame, mol_col: str | None) -> str:
    if mol_col and mol_col in df.columns:
        return mol_col
    for cand in ("mol", "smiles", "start_smiles"):
        if cand in df.columns:
            return cand
    raise KeyError(
        "Cannot find source SMILES column; tried 'mol', 'smiles', 'start_smiles'. "
        "请在测试集 CSV 中提供其中之一，或通过 --mol-col 显式指定。"
    )


def evaluate_molecule_predictions(
    df: pd.DataFrame,
    task_id: str,
    prop_trends: dict[str, str],
    prop_thresholds: dict[str, float],
    mol_col: str | None = None,
) -> tuple[float, float, float, float, float]:
    """
    对某个任务，逐行判断“最佳分子”是否：
      - 成功按正确方向改变性质（loose hit）
      - 严格超过阈值（strict hit）
      - RDKit 合法（valid）
      - scaffold 一致（same scaffold）
      - 与原始分子的相似度（avg similarity，Tanimoto）

    返回四个比例（百分数）及平均相似度：
    (loose_hit_rate, strict_hit_rate, valid_ratio, same_scaffold_ratio, avg_similarity)。
    """
    # 去掉 'x' / 'r' 后缀，映射回基础任务
    base_task_id = task_id.strip("x").strip("r")

    mol_col = _resolve_mol_col(df, mol_col)

    if base_task_id not in taskid_prop:
        raise KeyError(f"Task {task_id} missing in taskid_prop.")

    props = taskid_prop[base_task_id]
    trend_str = prop_trends.get(task_id) or prop_trends.get(base_task_id)
    if trend_str is None:
        raise KeyError(f"Task {task_id} missing in prop_trend.")
    if len(trend_str) < len(props):
        trend_str = trend_str.ljust(len(props), "1")

    total = len(df)
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    loose_cnt = 0
    strict_cnt = 0
    valid_cnt = 0
    scaffold_cnt = 0
    similarity_sum = 0.0

    for _, row in df.iterrows():
        start_smi = str(row[mol_col])
        gene_smi = str(row["gene_mol"])
        scaffold_smi = row.get("scaffold", "")

        # 相似度无论合法与否都计算；非法 SMILES 会得到 0.0
        similarity_sum += similarity_mols(start_smi, gene_smi)

        start_props = _calc_props(start_smi)
        gene_props = _calc_props(gene_smi)

        valid = start_props is not None and gene_props is not None
        if valid:
            valid_cnt += 1

        same_s = False
        if isinstance(scaffold_smi, str) and scaffold_smi:
            same_s = same_scaffold(gene_smi, scaffold_smi)
        if same_s:
            scaffold_cnt += 1

        loose = True
        strict = True
        if not valid:
            loose = False
            strict = False
        else:
            for p, tr in zip(props, trend_str):
                if p not in start_props or p not in gene_props:
                    loose = False
                    strict = False
                    break
                delta = float(gene_props[p] - start_props[p])
                thr = float(prop_thresholds.get(p, 0.0))

                if tr == "1":  # 期望增大
                    if delta <= 0:
                        loose = False
                    if delta <= thr:
                        strict = False
                else:  # 期望减小
                    if delta >= 0:
                        loose = False
                    if delta >= -thr:
                        strict = False

        if loose:
            loose_cnt += 1
        if strict:
            strict_cnt += 1

    loose_hit_rate = 100.0 * loose_cnt / total
    strict_hit_rate = 100.0 * strict_cnt / total
    valid_ratio = 100.0 * valid_cnt / total
    same_scaffold_ratio = 100.0 * scaffold_cnt / total
    avg_similarity = similarity_sum / total

    print(f"\n[Task {task_id}]")
    print(f"  Loose Accuracy: {loose_hit_rate:.2f}%")
    print(f"  Strict Accuracy: {strict_hit_rate:.2f}%")
    print(f"  Valid Ratio: {valid_ratio:.2f}%")
    print(f"  Same Scaffold Ratio: {same_scaffold_ratio:.2f}%")
    print(f"  Avg Similarity: {avg_similarity:.3f}")

    return (
        loose_hit_rate,
        strict_hit_rate,
        valid_ratio,
        same_scaffold_ratio,
        avg_similarity,
    )


def fix_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def _calc_props(smiles: str):
    """
    使用 RDKit 计算若干分子性质，返回字典。
    若 SMILES 非法，则返回 None。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        props = {
            "MolLogP": float(Crippen.MolLogP(mol)),
            "qed": float(QED.qed(mol)),
            "TPSA": float(Descriptors.TPSA(mol)),
            "NumHAcceptors": float(Lipinski.NumHAcceptors(mol)),
            "NumHDonors": float(Lipinski.NumHDonors(mol)),
            "MolWt": float(Descriptors.MolWt(mol)),
        }
    except Exception:
        return None
    return props


def generate_mol_property(df: pd.DataFrame, mol_col: str) -> pd.DataFrame:
    """
    给定 DataFrame 和一个分子列名 mol_col（如 'gene_mol'），
    为该列中的 SMILES 计算性质，并添加到 df：
      gene_MolLogP, gene_qed, gene_TPSA, gene_NumHAcceptors, gene_NumHDonors, gene_MolWt

    若某一行分子非法，则各个 gene_* 列置为 -999。
    """
    prefix = "gene_" if mol_col.startswith("gene_") else f"{mol_col}_"

    col_names = {
        "MolLogP": prefix + "MolLogP",
        "qed": prefix + "qed",
        "TPSA": prefix + "TPSA",
        "NumHAcceptors": prefix + "NumHAcceptors",
        "NumHDonors": prefix + "NumHDonors",
        "MolWt": prefix + "MolWt",
    }

    for v in col_names.values():
        if v not in df.columns:
            df[v] = -999.0

    for idx, smi in df[mol_col].astype(str).items():
        props = _calc_props(smi)
        if props is None:
            continue
        for key, val in props.items():
            df.at[idx, col_names[key]] = val

    return df


def generate_base_property(df: pd.DataFrame, mol_col: str | None = None) -> pd.DataFrame:
    """
    为源分子列 mol_col 计算基础性质，并写入标准列名：
      MolLogP, qed, TPSA, NumHAcceptors, NumHDonors, MolWt

    不依赖原始 CSV 是否已经提供这些列，统一用 RDKit 重算一遍，保证
    与 gene_* 的计算方式一致。
    """
    # 若调用方指定了 mol_col，则优先使用；否则在若干候选列名中自动推断
    if mol_col is not None:
        if mol_col not in df.columns:
            raise KeyError(
                f"指定的 mol 列 '{mol_col}' 不在测试集 CSV 中。"
            )
    else:
        for cand in ("mol", "smiles", "start_smiles"):
            if cand in df.columns:
                mol_col = cand
                break
        if mol_col is None:
            raise KeyError(
                "Cannot find source SMILES column; tried 'mol', 'smiles', 'start_smiles'. "
                "请在测试集 CSV 中提供其中之一，或通过 --mol-col 显式指定。"
            )

    # 统一增加一列标准名称 'mol'，方便后续相似度计算等逻辑复用
    df["mol"] = df[mol_col].astype(str)

    base_cols = ["MolLogP", "qed", "TPSA", "NumHAcceptors", "NumHDonors", "MolWt"]
    for c in base_cols:
        if c not in df.columns:
            df[c] = np.nan

    for idx, smi in df["mol"].astype(str).items():
        props = _calc_props(smi)
        if props is None:
            continue
        for key in base_cols:
            df.at[idx, key] = props.get(key, np.nan)

    return df


def similarity_mols(smi1: str, smi2: str) -> float:
    """
    使用 RDKit Morgan 指纹计算分子相似度（Tanimoto，相似度范围 [0,1]）。
    若任一分子非法，返回 0.0。
    """
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return 0.0
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        return float(sim)
    except Exception:
        return 0.0


def _murcko_scaffold_smiles(smi: str) -> str | None:
    """
    将任意输入安全地转换为 scaffold SMILES。
    若为 NaN / 空字符串 / 非法 SMILES，则返回 None。
    """
    # 处理 NaN / None 等情况
    if smi is None:
        return None
    try:
        # float 且为 NaN
        if isinstance(smi, float) and np.isnan(smi):
            return None
    except TypeError:
        pass

    smi_str = str(smi)
    if not smi_str or smi_str.lower() == "nan":
        return None

    mol = Chem.MolFromSmiles(smi_str)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if core is None or core.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(core)
    except Exception:
        return None


def same_scaffold(mol_smi: str, scaffold_smi: str) -> bool:
    """
    判断生成分子和给定 scaffold 是否具有相同的 Murcko scaffold。
    若任一非法，返回 False。
    """
    core1 = _murcko_scaffold_smiles(mol_smi)
    core2 = _murcko_scaffold_smiles(scaffold_smi)
    if core1 is None or core2 is None:
        return False
    return core1 == core2


def _pick_best_by_delta(
    *,
    start_smi: str,
    smiles_list: list[str],
    task_id: str,
    prop_trends: dict[str, str],
    prop_thresholds: dict[str, float],
) -> str:
    if not smiles_list:
        return ""

    start_props = _calc_props(str(start_smi))
    if start_props is None:
        return find_best_molecule(smiles_list, task_id)

    base_task_id = task_id.strip("x").strip("r")
    props = taskid_prop.get(base_task_id, [])
    trend_str = prop_trends.get(task_id) or prop_trends.get(base_task_id, "")
    if len(trend_str) < len(props):
        trend_str = trend_str.ljust(len(props), "1")

    best_smi = ""
    best_score = (-1, -1, -1, -1.0, -1.0)
    for smi in smiles_list:
        if not isinstance(smi, str) or not smi:
            continue
        gene_props = _calc_props(smi)
        if gene_props is None:
            continue
        strict_all = True
        loose_all = True
        dir_ok_cnt = 0
        margin_sum = 0.0
        dir_sum = 0.0
        for p, tr in zip(props, trend_str):
            if p not in start_props or p not in gene_props:
                strict_all = False
                loose_all = False
                break
            delta = float(gene_props[p] - start_props[p])
            thr = float(prop_thresholds.get(p, 0.0))
            if tr == "1":
                loose_ok = delta > 0
                strict_ok = delta > thr
                margin = delta - thr
                signed = delta
            else:
                loose_ok = delta < 0
                strict_ok = delta < -thr
                margin = (-delta) - thr
                signed = -delta
            if loose_ok:
                dir_ok_cnt += 1
            else:
                loose_all = False
            if not strict_ok:
                strict_all = False
            margin_sum += float(margin)
            dir_sum += float(signed)

        score = (int(strict_all), int(loose_all), int(dir_ok_cnt), float(margin_sum), float(dir_sum))
        if score > best_score:
            best_score = score
            best_smi = smi

    return best_smi


def find_best_molecule(smiles_list: list[str], task_id: str) -> str:
    """
    在给定的一组候选 SMILES 中，基于 RDKit 性质和 task 配置，
    选择一个“最优”的分子并返回其 SMILES。

    简单策略:
      - 对 taskid_prop[task_id] 中列出的性质:
          trend == '1' -> 越大越好 (+value)
          trend == '0' -> 越小越好 (-value)
      - 对每个 valid SMILES 求 score = ∑(±property)；
      - 选取得分最高的一个。
      - 若全部非法，则返回空字符串 ""。
    """
    if not smiles_list:
        return ""

    base_task_id = task_id.strip("x").strip("r")
    props = taskid_prop.get(base_task_id, [])
    trends_str = prop_trend.get(task_id) or prop_trend.get(base_task_id, "")
    if len(trends_str) < len(props):
        trends_str = trends_str.ljust(len(props), "1")

    best_smi = ""
    best_score = -1e9
    for smi in smiles_list:
        if not isinstance(smi, str) or not smi:
            continue
        prop_vals = _calc_props(smi)
        if prop_vals is None:
            continue
        score = 0.0
        for p, tr in zip(props, trends_str):
            val = float(prop_vals.get(p, 0.0))
            if p == "TPSA":
                val /= 10
            elif p == "NumHDonors":
                val *= 2

            if tr == "1":
                score += val
            elif tr == "0" :
                score -= val
        if score > best_score:
            best_score = score
            best_smi = smi

    return best_smi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(
            [str(i) for i in range(101, 109)]
            + [str(i) for i in range(201, 207)]
        ),
        help=(
            "要评估的任务 ID 列表，逗号分隔。"
            "默认: 101-108,201-206 共 16 个任务。"
        ),
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="/inspire/hdd/global_user/yujiajun-240108120114/project/rl4mo/data/test_chatdrug.csv",
        help="测试集 CSV 路径（默认 ./data/test_chatdrug.csv）。",
    )
    parser.add_argument(
        "--result_prefix",
        type=str,
        default="./",
        help="生成结果文件所在目录前缀，默认当前目录，文件名约定为 {task_id}_task.csv。",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./tasks.yaml",
        help="配置文件路径（默认 ./tasks.yaml）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="随机种子（默认 2024）。",
    )
    parser.add_argument(
        "--mol-col",
        type=str,
        default="start_smiles",
        help=(
            "测试集 CSV 中源 SMILES 的列名。"
            "若不指定，则会在 'mol'/'smiles'/'start_smiles' 中自动推断。"
        ),
    )

    args = parser.parse_args()

    fix_seed(args.seed)

    # ==== 加载配置 ====
    config = _load_yaml(args.config_path)

    # 这三个配置既在 main 中用到，也在 find_best_molecule / evaluate_molecule_predictions 中用到
    # 因此声明为全局变量，避免 NameError。
    global taskid_prop, prop_trend, prop_threshold
    taskid_prop_raw = config.get("taskid_prop")
    prop_trend = config.get("prop_trend")
    prop_threshold = config.get("prop_threshold", {})
    if not isinstance(taskid_prop_raw, dict) or not isinstance(prop_trend, dict):
        raise ValueError("config must contain dict keys: taskid_prop and prop_trend")
    if not isinstance(prop_threshold, dict):
        prop_threshold = {}
    taskid_prop = _normalize_task_props(taskid_prop_raw)
    prop_threshold = {str(_normalize_prop_name(k)): float(v) for k, v in prop_threshold.items()}

    # 解析任务列表
    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # 读取测试集（源分子）
    source_df_all = pd.read_csv(args.test_data)
    resolved_mol_col = _resolve_mol_col(source_df_all, args.mol_col)

    metrics_per_task = {}

    for task_id in task_ids:
        result_path = os.path.join(args.result_prefix, f"{task_id}_task.csv")
        if not os.path.exists(result_path):
            print(f"[Warn] 结果文件不存在，跳过任务 {task_id}: {result_path}")
            continue

        # 每个任务都从原始测试集重新拷贝一份，避免串扰
        source_df = source_df_all.copy()

        pred_series = _read_pred_series(result_path)
        pred_lists = [list(_parse_smiles_list(v)) for v in pred_series]
        if len(source_df) != len(pred_lists):
            print(
                f"[Warn] Row count mismatch for task {task_id}: test_data={len(source_df)} "
                f"predictions={len(pred_lists)} ({result_path}); will align by padding/truncation."
            )
            if len(pred_lists) < len(source_df):
                pred_lists.extend([[]] * (len(source_df) - len(pred_lists)))
            else:
                pred_lists = pred_lists[: len(source_df)]

        gene_mol: list[str] = []
        for i in range(len(source_df)):
            start_smi = str(source_df.iloc[i][resolved_mol_col])
            gene_mol.append(
                _pick_best_by_delta(
                    start_smi=start_smi,
                    smiles_list=pred_lists[i],
                    task_id=str(task_id),
                    prop_trends=prop_trend,
                    prop_thresholds=prop_threshold,
                )
            )
        source_df["gene_mol"] = gene_mol

        # 生成 gene_* 的性质列
        df = generate_mol_property(source_df, "gene_mol")

        # 评估该任务的指标
        metrics = evaluate_molecule_predictions(
            df=df,
            task_id=task_id,
            prop_trends=prop_trend,
            prop_thresholds=prop_threshold,
            mol_col=resolved_mol_col,
        )
        metrics_per_task[task_id] = metrics

    if not metrics_per_task:
        print("没有任何任务成功评估，请检查 tasks 列表和结果文件路径。")
        return

    # ==== 汇总所有任务的平均指标 ====
    # 指标顺序: (loose_hit_rate, strict_hit_rate, valid_ratio, same_scaffold_ratio, avg_similarity)
    all_tasks = sorted(metrics_per_task.keys(), key=lambda x: int(x.strip("xr")))
    print("\n===== Summary per task =====")
    for t in all_tasks:
        l, s, v, ss, sim = metrics_per_task[t]
        print(
            f"Task {t}: loose={l:.2f} strict={s:.2f} "
            f"valid={v:.2f} scaffold={ss:.2f} sim={sim:.3f}"
        )

    metrics_array = np.array(list(metrics_per_task.values()), dtype=float)  # [T, 5]
    avg_loose, avg_strict, avg_valid, avg_scaffold, avg_sim = metrics_array.mean(axis=0)

    print("\n===== Average over tasks =====")
    print(f"Avg Loose Accuracy: {avg_loose:.2f}%")
    print(f"Avg Strict Accuracy: {avg_strict:.2f}%")
    print(f"Avg Valid Ratio: {avg_valid:.2f}%")
    print(f"Avg Same Scaffold Ratio: {avg_scaffold:.2f}%")
    print(f"Avg Similarity: {avg_sim:.3f}")


if __name__ == "__main__":
    main()
