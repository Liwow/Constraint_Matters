import datetime
import pickle
from GCN import postion_get
import config
import utils
import gurobipy
import json
from gurobipy import GRB
import sys
import argparse
import gc
import helper
import gp_tools
import random
import os
import numpy as np
import torch
from time import time
# from get_logits import plot_logits
from helper import get_a_new2, get_bigraph, get_pattern
import logging
from gp_tools import primal_integral_callback, get_gp_best_objective, pred_error

presolve = False
note = "_presolve=" + str(presolve)

DEVICE = config.DEVICE
Threads = config.Threads
TimeLimit = config.TimeLimit
gap_threshold = 0.005
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
parser = argparse.ArgumentParser(description='test')
parser.add_argument('model', type=str, help='taskName', nargs='?', default="CA_anchor5")
parser.add_argument('instance', type=str, help='instanceName', nargs='?', default="CA")
parser.add_argument("-n", '--num', type=int, help='test number', default=100)
parser.add_argument("-s", '--solver', type=bool, help='is solver', default=True)
parser.add_argument("-b", '--bks', type=bool, help='is bks', default=False)

args = parser.parse_args()
acc_10 = 0
acc_20 = 0
acc_50 = 0
acc_100 = 0
tight_num = 0 
constraint_num=0



def recall_k(predicted_indices, actual_indices):
    """
    Compute recall at multiple ratio thresholds for predicted tight constraints.

    Args:
        predicted_indices (list): Predicted constraint indices sorted by confidence (high to low).
        actual_indices (list): Ground-truth tight constraint indices.

    Returns:
        dict: Recall values at different ratios.
    """
    # --- 1. Data preparation and edge cases ---
    
    # Convert to set for O(1) membership checks.
    actual_set = set(actual_indices)
    
    num_actual_tight = len(actual_set)
    
    # Return early when there is no ground-truth tight constraint.
    if num_actual_tight == 0:
        print("Warning: no ground-truth tight constraints found (empty slacks_indices); recall is undefined.")
        return { '10%': 0.0, '20%': 0.0, '50%': 0.0, '100%': 0.0 }

    # --- 2. Ratios to evaluate ---
    percentages_to_check = [0.1, 0.2, 0.5, 1.0] # 10%, 20%, 50%, 100%
    accuracies = {}

    # --- 3. Compute recall at each ratio ---
    for p in percentages_to_check:
        # Determine top-k size based on ratio and ground-truth count.
        k = math.ceil(p * num_actual_tight)
        
        # Get top-k predicted indices
        top_k_predictions = predicted_indices[:k]
        
        # Count matches in top-k
        correct_found = sum(1 for pred_idx in top_k_predictions if pred_idx in actual_set)
        
        accuracy = correct_found / k
        percentage_key = f"{int(p*100)}%"
        accuracies[percentage_key] = accuracy  

    return accuracies

def modify_by_predict(model, predict, k=0, fix=0, th=150, Delta_c=5):
    global acc_10, acc_20, acc_50, acc_100, tight_num,constraint_num
    con_folder = f"./logs/{instanceName}/{TaskName}_GRB_cons"
    if not os.path.exists(con_folder):
        os.makedirs(con_folder)
    con_file = os.path.join(con_folder, f"{test_ins_name}.con")

    min_topk = min(k, predict.size(0))
    topk_indices = torch.topk(predict, min_topk).indices
    critical_mask = torch.zeros_like(predict)
    critical_mask[topk_indices] = 1
    critical_constraints = torch.nonzero(critical_mask == 1, as_tuple=True)[0]
    ct_constraints = critical_constraints
    wrong_indices = []
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] or os.path.exists(con_file):
        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            slacks = {constr: constr.slack for constr in model.getConstrs() if constr.Sense in ['<', '>']}
        elif os.path.exists(con_file):
            with open(con_file, 'rb') as f:
                slack_values = pickle.load(f)
            slack_iter = iter(slack_values)
            slacks = {constr: next(slack_iter) for constr in model.getConstrs() if constr.Sense in ['<', '>']}
        else:
            slacks = {}
        slacks_indices = [i for i, constr in enumerate(model.getConstrs())
                          if constr.Sense in ['<', '>'] and abs(slacks[constr]) <= 1e-8]
        slacks_indices = [
            model_to_filtered_index[i] for i in slacks_indices if i in model_to_filtered_index
        ]
        all_indices = torch.topk(predict, len(slacks_indices)).indices.tolist()
        correct_tight_list = [index for index in all_indices if index in set(slacks_indices)]
        acc_dic = recall_k(all_indices, slacks_indices)
        acc_10 += acc_dic['10%']
        acc_20 += acc_dic['20%'] 
        acc_50 += acc_dic['50%']
        acc_100 += acc_dic['100%'] 
        tight_num += len(slacks_indices)
        constraint_num += len(model.getConstrs())
        wrong_indices = [i for i, value in enumerate(all_indices) if value not in slacks_indices]
        if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and not os.path.exists(con_file):
            with open(con_file, "wb") as f:
                pickle.dump(list(slacks.values()), f)
    if fix == 0:
        print("****** predict do nothing! *********")
        return
    
    m.reset()
    cons = model.getConstrs()
    remove_num = 0
    fixed_constraints = []
    all_tight_constraints = []
    for idx, i in enumerate(all_indices):
        idx_in_model = filtered_index_to_model[i]
        c = cons[idx_in_model]
        all_tight_constraints.append(c.ConstrName)
        if idx <= kc:
            fixed_constraints.append(c.ConstrName)
    
    # ct_constraints = replace_random_elements(ct_constraints, torch.sort(predict)[0].tolist(), 10)

    # for idx, c in enumerate(cons):
    #     if idx in model_to_filtered_index.keys() and model_to_filtered_index[idx] in ct_constraints:
    #         row = model.getRow(c)
    #         coeffs = []
    #         vars = []
    #         for i in range(row.size()):
    #             coeffs.append(row.getCoeff(i))
    #             vars.append(row.getVar(i))
    #         if c.Sense in ['<', '>'] and len(vars) < th:
    #             remove_num += 1
    #             model.remove(c)
    #             model.addConstr(gurobipy.LinExpr(coeffs, vars) == c.RHS, name=f"{c.ConstrName}_tight")
    z_vars = []
    M = 2
    for idx, c in enumerate(cons):
        current_idx = model_to_filtered_index.get(idx, idx) if model_to_filtered_index else idx
        if current_idx in ct_constraints:
            if c.Sense in ['<', '>']:
                remove_num += 1
                lhs_expr = model.getRow(c)
                z = model.addVar(vtype=gurobipy.GRB.BINARY, name=f"z_{c.ConstrName}")
                z_vars.append(z)
                model.addConstr(
                    (z == 0) >> (lhs_expr == c.RHS),
                    name=f"{c.ConstrName}_tight"
                )
            #     z = model.addVar(
            #         vtype=gurobipy.GRB.BINARY,
            #         name=f"z_{c.ConstrName}"
            #     )
            #     z_vars.append(z)
            # if c.Sense == '<':
            #     model.addConstr(
            #         lhs_expr >= c.RHS - M * z,
            #         name=f"{c.ConstrName}_bigM_lower_bound"
            #     )
            # elif c.Sense == '>':
            #     model.addConstr(
            #         lhs_expr <= c.RHS + M * z,
            #         name=f"{c.ConstrName}_bigM_upper_bound"
            #     )
    if z_vars:
        model.addConstr(
        gurobipy.quicksum(z_vars) <= Delta_c,
        name="trust_region_on_constraints"
    )
        
    print("remove_num: ", remove_num, ", threshold:", Delta_c)
    data_to_save = {
        "all_tight_constraints": all_tight_constraints,
        "fixed_constraints_name": fixed_constraints,
        "wrong_indices": wrong_indices[:200]
    }
    if not os.path.exists(f"{results_dir}/{TaskName}_{test_ins_name.split('.')[0]}.json"):
        with open(f"{results_dir}/{TaskName}_{test_ins_name.split('.')[0]}.json", "w") as data_file:
            json.dump(data_to_save, data_file, ensure_ascii=False, indent=4)
    return ct_constraints


fea = False
position = False
is_solver = args.solver
ps_solve = True
ModelName = args.model
TaskName = args.instance.split("_")[0]
# load pretrained model
if ModelName.startswith("IP"):
    # Add position embedding for IP model, due to the strong symmetry
    from model import GNNPolicy_ancon as GNNPolicy
    position = True
    fea = True
else:
    from model import GNNPolicy_ancon as GNNPolicy

model_name = f'{ModelName}.pth'
pathstr = f'./models/{model_name}'
policy = GNNPolicy(TaskName, position=position).to(DEVICE)
state = torch.load(pathstr, map_location=DEVICE)
policy.load_state_dict(state)

policy.eval()

instanceName = args.instance


def test_hyperparam(task):
    """
    set the hyperparams
    k_0, k_1, delta, kc
    """
    if task == "IP":
        return 400, 5, 10, 5,1
    elif task == "IS": 
        return 600, 600, 5, 800,1
    elif task == "IS_hard": 
        return 600, 600, 30, 700,1
    elif task == "WA":  
        return 0, 550, 10, 10000,1
    elif task == "CA":  
        return 1000, 0, 10, 80, 15
    elif task == "MVC":  
        return 600, 200, 20, 1000,1
    elif task == "MMCN2":
        return 5000, 0, 5, 800, 1


def is_consecutive_x_vars(var_names):
    ids = []
    for name in var_names:
        if not name.startswith("x"):
            return False
        try:
            idx = int(name[1:])
            if idx >= 20:
                return False  # only check one-digit suffix
            ids.append(idx)
        except:
            return False
    ids.sort()
    return all(ids[i] + 1 == ids[i + 1] for i in range(len(ids) - 1))


k_0, k_1, delta, kc, delta_c = test_hyperparam(instanceName)

# set log folder
solver = 'GRB'
test_task = f'{instanceName}_{solver}_Predict&Search'
if not os.path.isdir(f'./logs'):
    os.mkdir(f'./logs')
if not os.path.isdir(f'./logs/{instanceName}'):
    os.mkdir(f'./logs/{instanceName}')
log_folder = f'./logs/{instanceName}/{test_task}_con'  # modify log path here
if args.bks:
    TimeLimit = 3600
    ps_solve = False
    is_solver = True
    log_folder = f'./logs/{instanceName}/{test_task}_bks'
if not os.path.isdir(log_folder):
    os.mkdir(log_folder)

results_dir = f"/results/{instanceName}/"
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

sample_names = sorted(os.listdir(f'./instance/test/{instanceName}'))

acc = 0
subop_total = 0
obj_total = 0
solver_obj_total = 0
time_total_ps = 0
time_totol_solver = 0
max_subop = -1
max_time = 0
ps_int_total = 0
gp_int_total = 0
gp_pi = 0
ps_pi = 0
gp_gap_time = []
ps_gap_time = []
acc = 0
acc_local = 0
mse_total = 0
err = 0

ALL_Test = args.num  
epoch = 1
TestNum = round(ALL_Test / epoch)

if not is_solver:
    gp_obj_list = gp_tools.get_gp_best_objective(f'./logs/{instanceName}/{test_task}_bks')
else:
    gp_obj_list = []

for e in range(epoch):
    for ins_num in range(0, 1):
        test_ins_name = sample_names[(0 + e) * TestNum + ins_num]
        ins_name_to_read = f'./instance/test/{instanceName}/{test_ins_name}'
        # get bipartite graph as input
        v_class_name, c_class_name = get_pattern("./task_config.json", TaskName)
        A, v_map, v_nodes, c_nodes, b_vars, v_class, c_class, _ = get_bigraph(ins_name_to_read,
                                                                              fea, v_class_name, c_class_name)
        constraint_features = c_nodes.cpu()
        constraint_features[torch.isnan(constraint_features)] = 1 if not TaskName == "CA" else -1  # remove nan value
        variable_features = v_nodes
        if position:
            variable_features = postion_get(variable_features)
            constraint_features = postion_get(constraint_features)
        edge_indices = A._indices()
        edge_features = A._values().unsqueeze(1)
        # edge_features = torch.ones(edge_features.shape)
        v_class = utils.convert_class_to_labels(v_class, variable_features.shape[0])
        c_class = utils.convert_class_to_labels(c_class, constraint_features.shape[0])

        m = gurobipy.read(ins_name_to_read)
        cons = m.getConstrs()
        c_masks = [1 if con.sense in ['<', '>'] else 0 for con in cons]
        c_masks.append(0)  # add obj
        domain_mask = [1] * (constraint_features.size(0) - 1)
        domain_mask.append(0)
        split_index = None

        for i, c in enumerate(cons):
            conName = c.ConstrName
            if instanceName == "WA" and not conName.startswith("worker_used_ct"):
                domain_mask[i] = 0
            elif instanceName == "IP" and not conName.startswith("deficit_ct"):
                domain_mask[i] = 0
            elif instanceName == "CA":
                if i < 1985:
                    domain_mask[i] = 0
                else:
                    row = m.getRow(c)
                    coeffs = []
                    vars = []
                    for idx in range(row.size()):
                        coeffs.append(row.getCoeff(idx))
                        vars.append(row.getVar(idx))
                    var_names = [var.VarName for var in vars]
                    if split_index is None:
                        if is_consecutive_x_vars(var_names):
                            split_index = i
                    if split_index is not None and i >= split_index:
                        continue
                    else:
                        domain_mask[i] = 0
                domain_mask[i] = 0
            elif instanceName == "CA_hard" and i < 3000:
                domain_mask[i] = 0
            elif instanceName == "MMCN2" and c.Sense == "=":
                domain_mask[i] = 0

        domain_mask = torch.tensor(domain_mask, dtype=torch.int)
        c_masks = torch.tensor(c_masks, dtype=torch.int)
        eq_masks = c_masks
        c_masks = torch.bitwise_and(c_masks, domain_mask)
        is_con = True

        # prediction
        get_logits = False
        import time
        t1 = time.time()
        with torch.no_grad():
            BD = policy(
                constraint_features.to(DEVICE),
                edge_indices.to(DEVICE),
                edge_features.to(DEVICE),
                variable_features.to(DEVICE),
                torch.LongTensor(v_class).to(DEVICE),
                torch.LongTensor(c_class).to(DEVICE),
                get_logits=get_logits,
                con=is_con,
                c_mask=c_masks.to(DEVICE)
            )
        if get_logits:
            variable_features, constraint_features, pre_sols = BD
            # plot_logits(variable_features, constraint_features, v_class, c_class)
        elif not is_con:
            pre_sols = BD.cpu().squeeze()
        else:
            pre_sols, pre_cons = BD
            mask_indices = eq_masks.nonzero(as_tuple=True)[0]
            selected_values = torch.zeros_like(eq_masks[:-1], dtype=pre_cons.dtype, device=DEVICE)
            selected_values[domain_mask[:-1] == 1] = pre_cons
            selected_values = selected_values[eq_masks[:-1] == 1]
            pre_cons = selected_values

            pre_cons = pre_cons.cpu().squeeze()
            pre_sols = pre_sols.cpu().squeeze()

        t = time.time() - t1
        # pre_t = pre_t.cpu().squeeze()
        x_pred = torch.round(pre_sols)

        # align the variable name between the output and the solver
        all_varname = []
        for name in v_map:
            all_varname.append(name)
        binary_name = [all_varname[i] for i in b_vars]
        scores = []  # get a list of (index, VariableName, Prob, -1, type)
        for i in range(len(v_map)):
            type = "C"
            if all_varname[i] in binary_name:
                type = 'BINARY'
            scores.append([i, all_varname[i], pre_sols[i].item(), -1, type])

        scores.sort(key=lambda x: x[2], reverse=True)

        scores = [x for x in scores if x[4] == 'BINARY']  # get binary

        fixer = 0
        # fixing variable picked by confidence scores
        count1 = 0
        for i in range(len(scores)):
            if count1 < k_1:
                scores[i][3] = 1
                count1 += 1
                fixer += 1
        scores.sort(key=lambda x: x[2], reverse=False)
        count0 = 0
        for i in range(len(scores)):
            if count0 < k_0:
                scores[i][3] = 0
                count0 += 1
                fixer += 1

        print(f'instance: {test_ins_name}, '
              f'fix {k_0} 0s and '
              f'fix {k_1} 1s, delta {delta}. ')

        model_to_filtered_index, filtered_index_to_model = helper.map_model_to_filtered_indices(m)
        # Repair predicted initial solution to get an initial feasible solution.
        utils.grb_config(m, TimeLimit, Threads, presolve=presolve)
        gurobipy.setParam('LogToConsole', 1)
        log_file = f'{log_folder}/{test_ins_name}.log'
        m.Params.LogFile = log_file
        bks = gp_obj_list[(0 + e) * TestNum + ins_num] if len(gp_obj_list) > 0 else 1e-8
        t_start_1 = time()
        if is_solver:
            primal_integral_callback.gap_records = []
            primal_integral_callback.gap_threshold = gap_threshold
            primal_integral_callback.point = None
            output_folder = f"./logs/{instanceName}/{TaskName}_GRB_sols"
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"{test_ins_name}.sol")
            print("start solver")
            m.optimize(primal_integral_callback)
            # gp_pi += m.getAttr("PrimalInt")
            gp_bound = m.objBound
            obj = m.objVal
            integer_sols = sorted(
                [(v.varName, v.x) for v in m.getVars() if v.vType in ['I', 'B']],  # variable name and value
                key=lambda x: x[0]  # sort by variable name
            )
            integer_sols = [value for _, value in integer_sols]
            if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and not os.path.exists(output_file):
                with open(output_file, "wb") as f:
                    pickle.dump(integer_sols, f)
            if m.status == GRB.TIME_LIMIT or m.status == GRB.OPTIMAL:
                primal_integral_callback.gap_records.append(
                    (m.Runtime, abs((m.objVal - m.ObjBound) / abs(m.objVal + 1e-16)), m.objVal, m.ObjBound))
            gp_gap_records = primal_integral_callback.gap_records
            gp_gap_time.append(
                round(primal_integral_callback.point[0], 5) if primal_integral_callback.point is not None else -1)
            if TimeLimit == 3600:
                primal_integral = utils.save_gap_records(gp_gap_records, test_ins_name, ModelName, is_solver=True,
                                                         model="BKS", note=note)
            else:
                primal_integral = utils.save_gap_records(gp_gap_records, test_ins_name, ModelName, is_solver=True, note=note)
            gp_int_total += primal_integral
            print("gp_int_total:", gp_int_total)
        else:
            obj = gp_obj_list[(0 + e) * TestNum + ins_num] if len(gp_obj_list) > 0 else 0
        time_totol_solver += (time() - t_start_1)

        error_local, error_all, mse = pred_error(scores, test_ins_name, instanceName, pre_sols[b_vars])
        acc += (1 - error_all / len(scores))
        acc_local += (1 - error_local / (k_0 + k_1)) if (k_0 + k_1) != 0 else 0
        mse_total += mse
        err += error_local
        print(f"gurobi objective: {obj}; pred_error: {error_all}, {error_local}; mse: {mse}")

        # fix modes: 0=no fix, 1=random, 2=sorted, 3=intersection
        # modify(m, n=0, k=100, fix=0)  # if fix=0  do nothing
        if is_con:
            tight_constraints = modify_by_predict(m, pre_cons, k=kc, fix=1, Delta_c=delta_c)

        # trust region method implemented by adding constraints
        # TODO: variable fixing is Gurobi-centric in gp_tools and should be migrated to OptVerse.
        m = gp_tools.search(m, scores, delta)
        # TODO: solving backend should be migrated to OptVerse.
        primal_integral_callback.gap_records = []
        primal_integral_callback.gap_threshold = gap_threshold
        primal_integral_callback.point = None
        m.update()
        utils.grb_config(m, TimeLimit, Threads=1, presolve=presolve)
        if ps_solve:
            t_start_2 = time()
            m.optimize(primal_integral_callback)
            t_ps = time() - t_start_2
            pre_obj = m.objVal
            ps_pi += abs(bks - pre_obj) / bks
        else:
            pre_obj = 0
            t_ps = 0
        if m.status == GRB.TIME_LIMIT or m.status == GRB.OPTIMAL:
            primal_integral_callback.gap_records.append(
                (m.Runtime, abs((m.objVal - m.ObjBound) / abs(m.objVal + 1e-16)), m.objVal, m.ObjBound))
            ps_gap_records = primal_integral_callback.gap_records
            ps_gap_time.append(
                round(primal_integral_callback.point[0], 5) if primal_integral_callback.point is not None else -1)
            primal_integral = utils.save_gap_records(ps_gap_records, test_ins_name, ModelName, is_solver=False,
                                                     model="ps+con", kc=kc, note=note)
            ps_int_total += primal_integral
        print(f"ps_int_total: {ps_int_total}")
        if is_con:
            del pre_cons
        del BD, A, v_nodes, c_nodes, edge_indices, edge_features, b_vars, c_masks, constraint_features, pre_sols, variable_features, x_pred
        torch.cuda.empty_cache()
        gc.collect()

        time_total_ps += t_ps
        obj_total += pre_obj
        solver_obj_total += obj
        if max_time <= t_ps:
            max_time = t_ps
        if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            ps_bound = m.objBound
            pre_obj = m.objVal
            subop = (pre_obj - bks) / (bks + 1e-8) if not instanceName.startswith("CA") and instanceName.startswith(
                "IS") else (bks - pre_obj) / (bks + 1e-8)
            subop_total += subop
            print(
                f"solver objective: {obj}; ps objective: {pre_obj}; subopt: {round(subop / (ins_num + 1), 6)}; pred_error: {round(acc / (ins_num + 1), 4)}")
        else:
            print("infeasible")
        torch.cuda.empty_cache()


acc_10 = acc_10 / TestNum
acc_20 = acc_20 / TestNum   
acc_50 = acc_50 / TestNum
acc_100 = acc_100 / TestNum
tight_proportion = tight_num / constraint_num if constraint_num > 0 else 0
tight_num = tight_num / TestNum
print(f"acc_10: {acc_10}, acc_20: {acc_20}, acc_50: {acc_50}, acc_100: {acc_100}")
print(f"tight_num: {tight_num}, tight_proportion: {tight_proportion}")
total_num = TestNum * epoch
current_date = datetime.datetime.now()
date_string = current_date.strftime("%Y%m%d%H%M")
results = {
    "avg_subopt": round(subop_total / total_num, 6),
    "avg_obj": round(obj_total / total_num, 6),
    "avg_solver_obj": round(solver_obj_total / total_num, 6),
    "mean_time_pred_ps": round(time_total_ps / total_num, 6),
    "mean_time_solver": round(time_totol_solver / total_num, 6),
    "max_time": round(max_time, 6),
    "gurobi_integral": round(gp_int_total / total_num, 6) if is_solver else 0,
    "ps_gap_integral": round(ps_int_total / total_num, 6) if ps_solve else 0,
    "gurobi_survive": gp_gap_time,
    "ps_survive": ps_gap_time,
    "number": total_num,
    "model": f"{ModelName},kc={kc}",
    "thread": f"{Threads}",
    "time": f"{date_string}"
}

with open(results_dir + "results.json", "a") as file:
    json.dump(results, file, indent=4)
    file.write("\n")

print("avg_time_pred_ps: ", results['mean_time_pred_ps'])
print("avg_time_solver: ", results['mean_time_solver'])
print("avg_subopt:", results['avg_subopt'])
print("ps_gap_integral:", results['ps_gap_integral'])
print(f"pred_error_all: {round(acc / total_num, 4)}")
print(f"pred_error_local: {round(acc_local / total_num, 4)}")
print(f"mse: {mse_total / total_num}")
print(f"avg_error_local: {round(err / total_num, 4)}")
