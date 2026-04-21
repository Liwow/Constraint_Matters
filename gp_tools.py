import json
import os
import pickle
import re
import random
from time import time
import gurobipy as gp
import torch
import numpy as np
from gurobipy import GRB

import config
import utils
from helper import get_a_new2, get_pattern, get_bigraph
import pyscipopt as scp

DEVICE = config.DEVICE
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def search(m, scores, delta):
    """

    :param m: GUROBI model
    :param scores:
    :param delta:
    :return:
    """
    instance_variabels = m.getVars()
    instance_variabels.sort(key=lambda v: v.VarName)
    variabels_map = {}
    for v in instance_variabels:  # get a dict (variable map), varname:var clasee
        variabels_map[v.VarName] = v
    alphas = []
    for i in range(len(scores)):
        tar_var = variabels_map[scores[i][1]]  # target variable <-- variable map
        x_star = scores[i][3]  # 1,0,-1, decide whether need to fix
        if x_star < 0:
            continue
        # tmp_var = m1.addVar(f'alp_{tar_var}', 'C')
        tmp_var = m.addVar(name=f'alp_{tar_var}', vtype=GRB.CONTINUOUS)
        alphas.append(tmp_var)
        m.addConstr(tmp_var >= tar_var - x_star, name=f'alpha_up_{i}')
        m.addConstr(tmp_var >= x_star - tar_var, name=f'alpha_dowm_{i}')
    all_tmp = 0
    for tmp in alphas:
        all_tmp += tmp
    m.addConstr(all_tmp <= delta, name="sum_alpha")
    return m


def search_SCIP(m1, scores, delta):
    """

    :param m1:  SCIP MODEL
    :param scores:
    :param delta:
    :return:
    """
    m1_vars = m1.getVars()
    var_map1 = {}
    for v in m1_vars:  # get a dict (variable map), varname:var clasee
        var_map1[v.name] = v
    alphas = []
    for i in range(len(scores)):
        tar_var = var_map1[scores[i][1]]  # target variable <-- variable map
        x_star = scores[i][3]  # 1,0,-1, decide whether to fix
        if x_star < 0:
            continue
        tmp_var = m1.addVar(f'alp_{tar_var}_{i}', 'C')
        alphas.append(tmp_var)
        m1.addCons(tmp_var >= tar_var - x_star, f'alpha_up_{i}')
        m1.addCons(tmp_var >= x_star - tar_var, f'alpha_down_{i}')
    m1.addCons(scp.quicksum(ap for ap in alphas) <= delta, 'sum_alpha')
    return m1


def get_gp_best_objective(log_folder):
    best_objectives = []
    pattern = r"Best objective ([\d.e+-]+)"  
    filenames = sorted(os.listdir(log_folder))
    # 遍历文件夹中的所有文件
    for filename in filenames:
        filepath = os.path.join(log_folder, filename)

        # 确保只处理文件
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                    
                        match = re.search(pattern, line)
                        if match:
                            
                            best_objectives.append(float(match.group(1)))
                            break 
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    return best_objectives


def primal_integral_callback(model, where):
    if where == gp.GRB.Callback.MIP:
        
        time_elapsed = model.cbGet(gp.GRB.Callback.RUNTIME)

        
        ub = model.cbGet(gp.GRB.Callback.MIP_OBJBST)  # Best solution (upper bound)
        lb = model.cbGet(gp.GRB.Callback.MIP_OBJBND)  # Best bound (lower bound)

        
        if lb < gp.GRB.INFINITY and ub < gp.GRB.INFINITY:
            primal_gap = abs(ub - lb) / abs(ub + 1e-16)
           
            if not primal_integral_callback.gap_records:
                primal_integral_callback.gap_records.append((0, primal_gap, ub, lb))
            primal_integral_callback.gap_records.append((time_elapsed, primal_gap, ub, lb))
            if primal_gap <= primal_integral_callback.gap_threshold:
                if primal_integral_callback.point is None:
                    primal_integral_callback.point = (time_elapsed, primal_gap)


from pyscipopt import Eventhdlr, SCIP_RESULT, SCIP_EVENTTYPE


class PrimalIntegralEventhdlr(Eventhdlr):
    def __init__(self, gap_threshold):
        super().__init__()
        self.gap_records = []
        self.gap_threshold = gap_threshold
        self.point = None
        self.start_time = time()

    def eventinit(self):
       
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
        self.model.catchEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)
        self.model.catchEvent(SCIP_EVENTTYPE.OBJCHANGED, self)

    def eventexit(self):
       
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
        self.model.dropEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)
        self.model.dropEvent(SCIP_EVENTTYPE.OBJCHANGED, self)

    def eventexec(self, event):
        
        time_elapsed = time() - self.start_time

        
        if self.model.getStatus() == 'nodelimit' or self.model.getStatus() == 'timelimit' or self.model.getNSols() > 0:
            ub = self.model.getPrimalbound()  
            lb = self.model.getDualbound()  

            
            if abs(lb) < 1e+10 and abs(ub) < 1e+10:
                primal_gap = abs(ub - lb) / abs(ub + 1e-16)
                
                if not self.gap_records:
                    self.gap_records.append((0, primal_gap, ub, lb))
                self.gap_records.append((time_elapsed, primal_gap, ub, lb))
                if primal_gap <= self.gap_threshold and self.point is None:
                    self.point = (time_elapsed, primal_gap)

        return SCIP_RESULT.SUCCESS


def pred_error(scores, test_ins_name, InstanceName, BD=None):
    TaskName = InstanceName.split('_')[0]
    sols_files = f"./logs/{InstanceName}/{TaskName}_GRB_sols"
    sols_file = sols_files + "/" + test_ins_name + ".sol"
    with open(sols_file, 'rb') as f:
        sols = pickle.load(f)

    sorted_scores = sorted(scores, key=lambda x: x[0])

    local_count = 0
    correct_count_local = 0
    err = []
    for score, sol in zip(sorted_scores, sols):
        variable_value = score[3] 
        if variable_value in [0, 1]:
            local_count += 1
            if variable_value == sol:
                correct_count_local += 1
            else:
                err.append(score)

    error_local = local_count - correct_count_local

    if BD is None:
        return error_local
    else:
        pre_sol = torch.round(BD)
        sols_tensor = torch.tensor(sols)
        error_all = sum([1 for px, x in zip(pre_sol, sols_tensor) if px != x])
        mse = torch.mean((BD - sols_tensor) ** 2)
        return error_local, error_all, mse

