import glob
import random
import re
from typing import Dict

import numpy as np
from sklearn.cluster import KMeans
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from config import CONSTRAINT_TYPES


def split_sample_by_blocks(sample_files, train_rate, block_size):
    """Split sample files block-wise into train/valid sets with deterministic shuffling."""
    sample_files = sorted(sample_files, key=lambda x: int(re.search(r'\d+', str(x)).group()))
    sample_files = sample_files[:]
    random.seed(0)
    train_files = []
    valid_files = []

    num_blocks = (len(sample_files) + block_size - 1) // block_size

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, len(sample_files))

        block_files = sample_files[start_idx:end_idx]

        random.shuffle(block_files)
        split_index = int(train_rate * len(block_files))
        train_files.extend(block_files[:split_index])
        valid_files.extend(block_files[split_index:])

    return train_files, valid_files


def get_label_by_kmeans(values, n_type=2):
    """Cluster a 1D list and return (sparse-cluster-id, labels)."""
    X = np.array(values).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_type, random_state=42).fit(X)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_.flatten()
    sparse_cluster = np.argmin(cluster_centers)
    return sparse_cluster, labels


def get_pair(critical_list, labels, critical_num):
    """Build heuristic scores from critical flags and predicted labels."""
    if len(labels) != len(critical_list):
        return
    scores = []
    critical_num = normalize_to_range(critical_num)
    for i in range(len(labels)):
        score = 0
        if labels[i]:
            score += 5
        elif critical_list[i]:
            score += critical_num[i]
        scores.append(score)

    return scores


def focal_loss(pre_cons, labels, weight, alpha=0.75, gamma=2):
    pos_loss = - 2 * alpha * ((1 - pre_cons + 1e-8) ** gamma) * torch.log(pre_cons + 1e-8) * (labels == 1).float()
    neg_loss = - 2 * (1 - alpha) * (pre_cons ** gamma) * torch.log(1 - pre_cons + 1e-8) * (labels == 0).float()

    masked_con_loss = (pos_loss + neg_loss) * weight[:, None]

    return masked_con_loss


def normalize_to_range(data, new_min=0, new_max=2):
    """Normalize a numeric list into [new_min, new_max]."""
    if not data:
        raise ValueError("The input list is empty.")

    old_min = min(data)
    old_max = max(data)

    if old_min == old_max:
        return [new_min] * len(data)

    normalized_data = [
        new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min)
        for x in data
    ]
    return normalized_data


def convert_class_to_labels(class_, n):
    """Convert grouped index classes to a dense label list."""
    labels = [-1] * n
    for class_idx, indices in enumerate(class_):
        for idx in indices:
            labels[idx] = class_idx
    return labels


def save_gap_records(gp_gap_records, test_ins_name, model_name, is_solver=False, model="ps", kc=None, note=None):
    """
    Save gap records to disk and generate a visualization plot.

    Args:
        gp_gap_records: list of tuples (time, gap, best_obj, best_bound)
        test_ins_name: instance name
        model_name: model name in saved file names
        is_solver: whether this comes from a direct solver run
    """
    task = re.search("^(.*?)_(?=\d)", test_ins_name).group()[:-1]
    root_dir = os.path.join(".", "results", task, "gap")
    os.makedirs(root_dir, exist_ok=True)

    if is_solver:
        if model == "scip":
            save_dir = os.path.join(root_dir, "scip")
        elif model == "BKS":
            save_dir = os.path.join(root_dir, "BKS")
        else:
            save_dir = os.path.join(root_dir, "gp")
    else:
        save_dir = os.path.join(root_dir, f"{model}")
    os.makedirs(save_dir, exist_ok=True)

    current_date = datetime.datetime.now()
    date_string = current_date.strftime("%Y%m%d")
    tmp = current_date.strftime("%Y%m%d%H%M")
    test_dir = os.path.join(save_dir, date_string)
    if kc is not None:
        test_dir = test_dir + f"_{kc}"
    if note is not None:
        test_dir = test_dir + note
    os.makedirs(test_dir, exist_ok=True)

    instance_dir = os.path.join(test_dir, test_ins_name)
    os.makedirs(instance_dir, exist_ok=True)
    times = [record[0] for record in gp_gap_records]
    gaps = [record[1] for record in gp_gap_records]
    objs = [record[2] for record in gp_gap_records]
    bounds = [record[3] for record in gp_gap_records]

    df = pd.DataFrame({
        'time': times,
        'gap': gaps,
        'best_obj': objs,
        'best_bound': bounds,
    })

    filename = f"{model_name}_{tmp}"

    csv_path = os.path.join(instance_dir, f"{filename}_gap.csv")
    df.to_csv(csv_path, index=False)
    print(f"Gap records saved to: {csv_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(times, gaps, 'b-', linewidth=2, label=model_name)

    plt.xlabel('Solving Time (s)', fontsize=12)
    plt.ylabel('Relative Gap', fontsize=12)
    plt.title(f'Gap Evolution of {model_name} / {model_name}', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')

    if min(gaps) >= 0:
        plt.ylim(bottom=0)

    plt.xlim(left=0)

    plt_path = os.path.join(instance_dir, f"{filename}_plot.png")
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gap curve saved to: {plt_path}")

    if len(times) <= 1:
        gap_integral = 0.0
    else:
        gap_integral = 0.0
        for i in range(1, len(times)):
            if gaps[i - 1] > 1e+8:
                continue
            dt = times[i] - times[i - 1]
            gap_integral += (gaps[i] + gaps[i - 1]) * dt / 2.0

    integral_path = os.path.join(instance_dir, f"{filename}_integral.txt")
    with open(integral_path, 'w') as f:
        f.write(f"Gap Integral: {gap_integral:.6f}")
    print(f"Gap integral saved to: {integral_path}")

    return gap_integral


def find_type(model, constr):
    """
    Detect the type of a SCIP linear constraint.

    This function normalizes negative binary terms by treating ``-x`` as
    ``(1 - x)`` so structurally equivalent constraints map to the same type.
    """
    try:
        var_coeffs = model.getValsLinear(constr)
        if not var_coeffs:
            return CONSTRAINT_TYPES["General Linear"]

        coefficients = list(var_coeffs.values())
        variables = list(var_coeffs.keys())

        lhs = model.getLhs(constr)
        rhs = model.getRhs(constr)

        if lhs == -float('inf') or lhs == -1e+20:
            sense = '<='
        elif rhs == float('inf') or rhs == 1e+20:
            sense = '>='
        elif lhs == rhs:
            sense = '=='
        else:
            sense = 'range'

        var_types = [var.vtype() for var in variables]

        transformed_coefficients = coefficients.copy()
        rhs_adjustment = 0

        for i, (var, coef, var_type) in enumerate(zip(variables, coefficients, var_types)):
            if var_type == 'BINARY' and coef < 0:
                transformed_coefficients[i] = -coef
                rhs_adjustment += -coef

        if sense == '<=' or sense == '==':
            rhs = rhs + rhs_adjustment
        if sense == '>=' or sense == '==':
            lhs = lhs + rhs_adjustment if lhs != -float('inf') else lhs

        all_positive_coef = all(coef > 0 for coef in transformed_coefficients)

        if len(variables) == 1:
            return CONSTRAINT_TYPES["Singleton"]

        if len(variables) == 2 and sense == '<=' and var_types[0] == var_types[1]:
            if coefficients[0] > 0 and coefficients[1] < 0 and abs(coefficients[0]) == abs(coefficients[1]):
                return CONSTRAINT_TYPES["Precedence"]

        if all(abs(coef) == 1 for coef in transformed_coefficients) and all(vtype == 'BINARY' for vtype in var_types):
            if sense == '==' and rhs == 1:
                return CONSTRAINT_TYPES["SetPartitioning"]
            elif sense == '<=' and rhs == 1:
                return CONSTRAINT_TYPES["SetPacking"]
            elif sense == '>=' and rhs == 1:
                return CONSTRAINT_TYPES["SetCovering"]

        if all(coef == 1 for coef in transformed_coefficients) and all(vtype == 'BINARY' for vtype in var_types):
            if sense == '==' and rhs >= 1 and float(rhs).is_integer():
                return CONSTRAINT_TYPES["Cardinality"]

        if all(vtype == 'BINARY' for vtype in var_types):
            if sense == '<=' and rhs >= 1:
                if all(coef == 1 for coef in transformed_coefficients):
                    return CONSTRAINT_TYPES["Invariant Knapsack"]
                elif all_positive_coef:
                    return CONSTRAINT_TYPES["Knapsack"]
            elif sense == '==' and rhs >= 1:
                if all_positive_coef:
                    return CONSTRAINT_TYPES["Equation Knapsack"]

        if all(vtype == 'INTEGER' for vtype in var_types) and sense == '<=' and rhs >= 1 and float(rhs).is_integer():
            if all_positive_coef:
                return CONSTRAINT_TYPES["Knapsack"]

        if sense == '<=' and any(vtype == 'BINARY' for vtype in var_types):
            binary_indices = [i for i, vtype in enumerate(var_types) if vtype == 'BINARY']
            if len(binary_indices) == 1:
                bin_var_coef = transformed_coefficients[binary_indices[0]]
                if bin_var_coef > 0 and bin_var_coef == rhs and bin_var_coef >= 2:
                    return CONSTRAINT_TYPES["BinPacking"]

        if len(variables) == 2:
            if (var_types[0] == 'BINARY' and var_types[1] != 'BINARY') or \
                    (var_types[1] == 'BINARY' and var_types[0] != 'BINARY'):
                return CONSTRAINT_TYPES["Variable Bound"]

        if sense == '==' and (all_positive_coef or all(coef < 0 for coef in transformed_coefficients)):
            return CONSTRAINT_TYPES["Aggregations"]

        if any(vtype == 'BINARY' for vtype in var_types) and any(vtype == 'CONTINUOUS' for vtype in var_types):
            return CONSTRAINT_TYPES["Mixed Binary"]

        return CONSTRAINT_TYPES["General Linear"]

    except Exception as e:
        print(f"Error analyzing constraint: {e}")
        return CONSTRAINT_TYPES["General Linear"]


def score(n_var: int, con_type: int):
    """Placeholder for a future constraint scoring heuristic."""
    pass


def grb_config(m, TimeLimit=800, Threads=1, presolve=True):
    m.Params.TimeLimit = TimeLimit
    m.Params.Threads = Threads
    m.Params.MIPFocus = 1
    if not presolve:
        m.setParam("Presolve", 0)
