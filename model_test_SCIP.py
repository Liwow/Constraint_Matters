import datetime
import pickle
import sys
import config
import utils
import gurobipy
import json
import pyscipopt as scp
from pyscipopt import SCIP_PARAMSETTING,quicksum
import argparse
import gc
import helper
import gp_tools
import random
import os
import numpy as np
import torch
from time import time
from helper import get_a_new2, get_bigraph, get_pattern
import logging
from gp_tools import get_gp_best_objective

DEVICE = config.DEVICE
Threads = config.Threads
TimeLimit = config.TimeLimit
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
parser = argparse.ArgumentParser(description='test')
parser.add_argument('model', type=str, help='taskName', nargs='?', default="CA")
parser.add_argument('instance', type=str, help='instanceName', nargs='?', default="CA")
parser.add_argument("-n", '--num', type=int, help='test number', default=100)
parser.add_argument("-s", '--solver', type=bool, help='is solver', default=False)

args = parser.parse_args()


def modify_by_predict(model, predict, k=0, fix=0, th=150, Delta_c=5):
    """
    model - SCIP model
    predict - con predict
    k - num of fix
    fix - is fix?
    """
    min_topk = min(k, predict.size(0))
    topk_indices = torch.topk(predict, min_topk).indices
    all_indices = torch.topk(predict, 3 * kc).indices.tolist()
    critical_mask = torch.zeros_like(predict)
    critical_mask[topk_indices] = 1
    critical_constraints = torch.nonzero(critical_mask == 1, as_tuple=True)[0]
    ct_constraints = critical_constraints

    if fix == 0:
        print("****** predict do nothing! *********")
        return

    model.freeTransform()

    cons = model.getConss()
    remove_num = 0
    fixed_constraints = []
    all_tight_constraints = []

    for idx, i in enumerate(all_indices):
        idx_in_model = filtered_index_to_model[i]
        c = cons[idx_in_model]
        all_tight_constraints.append(c.name)
        if idx < kc:
            fixed_constraints.append(c.name)

    var_map = {}
    z_vars = []
    for v in model.getVars():
        var_map[v.name] = v
    for idx, c in enumerate(cons):
        if idx in model_to_filtered_index.keys() and model_to_filtered_index[idx] in ct_constraints:
            rhs = model.getRhs(c)
            lhs = model.getLhs(c)
            coeffs = model.getValsLinear(c)
            expr = [val * var_map[var] for  var, val in coeffs.items()]
            expr = sum(expr)
            is_leq = (lhs == -1e+20 and rhs < 1e+20)
            is_geq = (lhs > -1e+20 and rhs == 1e+20)
            z = model.addVar(vtype="B", name=f"z_{c.name}")
            z_vars.append(z)
            if is_leq:
              model.addConsIndicator(
                cons= -expr <= -rhs,      # activated constraint
                binvar=z,               # trigger binary variable
                activeone = False,
                name=f"{c.name}_upper"
            )
            if is_geq:
              model.addConsIndicator(
                cons= expr <= lhs,      # activated constraint
                binvar=z,               # trigger binary variable
                activeone = False,
                name=f"{c.name}_lower"
            )

            # is_inequality = (lhs == -1e+20 and rhs < 1e+20) or (
            #         lhs > -1e+20 and rhs == 1e+20)
            # if is_inequality:
            #     coeff = model.getValsLinear(c)
            #     if len(coeff) < th:
            #         remove_num += 1
            #         cons_name = c.name
            #         if lhs == -1e+20 and rhs != 1e+20:
            #             model.chgLhs(c, rhs)
            #         elif rhs == 1e+20 and lhs != -1 + 20:
            #             model.chgRhs(c, lhs)
    if z_vars:
      model.addCons(
          quicksum(z_vars) <= Delta_c,
          name="trust_region_on_constraints"
      )
    print("remove_num: ", remove_num, ", threshold:", Delta_c)
    return ct_constraints


fea = False
position = False
is_solver = args.solver
ps_solve = True
ModelName = args.model
TaskName = ModelName.split("_")[0]
# load pretrained model
if ModelName.startswith("IP"):
    # Add position embedding for IP model, due to the strong symmetry
    from GCN import postion_get
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


def is_consecutive_x_vars(var_names):
    ids = []
    for name in var_names:
        if not name.startswith("x"):
            return False
        try:
            idx = int(name[1:])
            if idx >= 20:
                return False
            ids.append(idx)
        except:
            return False
    ids.sort()
    return all(ids[i] + 1 == ids[i + 1] for i in range(len(ids) - 1))


def test_hyperparam(task):
    """
    set the hyperparams
    k_0, k_1, delta, kc
    """
    if task == "IP":
        return 400, 5, 10, 5,1
    elif task == "IS":
        return 600, 600, 5, 500,1
    elif task == "WA": 
        return 0, 550, 5, 10000,1
    elif task == "CA": 
        return 1000, 0, 10, 100, 15
    elif task == "MVC":  
        return 600, 200, 20, 1000, 1
    elif task == "MMCN2": 
        return 5000, 0, 5, 800,1


k_0, k_1, delta, kc, delta_c = test_hyperparam(instanceName)

# set log folder
solver = 'SCIP'
test_task = f'{instanceName}_{solver}_Predict&Search'
if not os.path.isdir(f'./logs'):
    os.mkdir(f'./logs')
if not os.path.isdir(f'./logs/{instanceName}'):
    os.mkdir(f'./logs/{instanceName}')
log_folder = f'./logs/{instanceName}/{test_task}_con'  
if not os.path.isdir(log_folder):
    os.mkdir(log_folder)

# todo 
results_dir = f"/home/ljj/project/predict_and_search/results/{instanceName}/"
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

sample_names = sorted(os.listdir(f'./instance/test/{instanceName}'))

acc = 0
subop_total = 0
obj_total = 0
solver_obj_total = 0
time_total_ps = 0
time_totol_solver = 0
max_time = 0


# ALL_Test = len(sample_names)
ALL_Test = args.num  # 16/30
epoch = 1
TestNum = round(ALL_Test / epoch)

if not is_solver:
    gp_obj_list = get_gp_best_objective(f'./logs/{instanceName}/{instanceName}_GRB_Predict&Search_bks')
else:
    gp_obj_list = []

for e in range(epoch):
    for ins_num in range(0, 5):
        test_ins_name = sample_names[(0 + e) * TestNum + ins_num]
        ins_name_to_read = f'./instance/test/{instanceName}/{test_ins_name}'
        # get bipartite graph as input
        v_class_name, c_class_name = get_pattern("./task_config.json", TaskName)
        A, v_map, v_nodes, c_nodes, b_vars, v_class, c_class, _ = get_bigraph(ins_name_to_read,
                                                                              fea, v_class_name, c_class_name)
        constraint_features = c_nodes.cpu()
        constraint_features[torch.isnan(constraint_features)] = -1
        variable_features = v_nodes
        if instanceName == "IP":
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

        model_to_filtered_index, filtered_index_to_model = helper.map_model_to_filtered_indices(m)
        m.dispose()
        del m

        # prediction
        get_logits = False
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
        if not is_con:
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

        m1 = scp.Model()
        m1.setParam('limits/time', TimeLimit)
        m1.setIntParam("lp/threads", Threads)
        m1.setParam('randomization/randomseedshift', 0)
        m1.setParam('randomization/lpseed', 0)
        m1.setParam('randomization/permutationseed', 0)
        m1.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)  # MIP focus
        m1.setLogfile(f'{log_folder}/{test_ins_name}.log')
        m1.readProblem(ins_name_to_read)

        t_start_1 = time()
        if is_solver:
            print("start solver")
            m1.optimize()

            scip_bound = m1.getDualbound()
            obj = m1.getPrimalbound()

            if m1.getStatus() == 'timelimit' or m1.getStatus() == "optimal":
                time_elapsed = m1.getSolvingTime()
                primal_gap = abs((obj - scip_bound) / abs(obj + 1e-16))
        else:
            obj = 0
        time_totol_solver += (time() - t_start_1)

        if is_con:
            tight_constraints = modify_by_predict(m1, pre_cons, k=kc, fix=1, Delta_c=delta_c)

        # trust region method implemented by adding constraints
        m1 = gp_tools.search_SCIP(m1, scores, delta)

        m1.setParam('limits/time', TimeLimit)
        m1.setIntParam("lp/threads", Threads)
        m1.setParam('randomization/randomseedshift', 0)
        m1.setParam('randomization/lpseed', 0)
        m1.setParam('randomization/permutationseed', 0)
        m1.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)  # MIP focus
        m1.setLogfile(f'{log_folder}/{test_ins_name}.log')

        if ps_solve:
            t_start_2 = time()
            m1.optimize()
            t_ps = time() - t_start_2
            ps_bound = m1.getDualbound()
            pre_obj = m1.getPrimalbound()
            if m1.getStatus() == 'timelimit' or m1.getStatus() == "optimal":
                time_elapsed = m1.getSolvingTime()
                primal_gap = abs((pre_obj - ps_bound) / abs(pre_obj + 1e-16))
   
        else:
            ps_bound = 0
            pre_obj = 0
            t_ps = 0

        if is_con:
            del pre_cons
        del BD, A, v_nodes, c_nodes, edge_indices, edge_features, b_vars, c_masks, constraint_features, pre_sols, variable_features
        torch.cuda.empty_cache()
        gc.collect()

        time_total_ps += t_ps
        obj_total += pre_obj
        solver_obj_total += obj
        if max_time <= t_ps:
            max_time = t_ps
        if m1.getStatus() == 'timelimit' or m1.getStatus() == 'optimal':
            subop = (pre_obj - obj) / (obj + 1e-8) if not instanceName.startswith("CA") and instanceName.startswith(
                "IS") else (obj - pre_obj) / (
                    obj + 1e-8)
            subop_total += subop
        else:
            print("no feasible")
        del m1
        torch.cuda.empty_cache()

