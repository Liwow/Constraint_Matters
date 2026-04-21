import json
import torch.nn.functional as F
import os
import re
import sys
import argparse
import pathlib
from typing import Tuple

import numpy as np
import random
import pyscipopt as scp
from pyscipopt.scip import Model
import torch
import torch.nn as nn
import pickle
import gurobipy as gp
from gurobipy import GRB
# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor

device = torch.device("cpu")


def position_get_ordered_flt(variable_features):
    lens = variable_features.shape[0]
    feature_widh = 20  # max length 4095
    sorter = variable_features[:, 1]
    position = torch.argsort(sorter)
    position = position / float(lens)

    position_feature = torch.zeros(lens, feature_widh)

    for row in range(position.shape[0]):
        flt_indx = position[row]
        divider = 1.0
        for ks in range(feature_widh):
            if divider <= flt_indx:
                position_feature[row][ks] = 1
                flt_indx -= divider
            divider /= 2.0
            # print(row,position[row],position_feature[row])
    position_feature = position_feature.to(device)
    variable_features = variable_features.to(device)
    v = torch.concat([variable_features, position_feature], dim=1)
    return v


def get_BG_from_scip(ins_name):
    epsilon = 1e-6

    # vars:  [obj coeff, norm_coeff, degree, max coeff, min coeff, Bin?]
    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(ins_name)

    ncons = m.getNConss()
    nvars = m.getNVars()

    mvars = m.getVars()
    mvars.sort(key=lambda v: v.name)

    v_nodes = []

    b_vars = []

    ori_start = 6
    emb_num = 15

    for i in range(len(mvars)):
        tp = [0] * ori_start
        tp[3] = 0
        tp[4] = 1e+20
        # tp=[0,0,0,0,0]
        if mvars[i].vtype() == 'BINARY' or mvars[i].vtype() == 'INTEGER':
            tp[ori_start - 1] = 1
            b_vars.append(i)

        v_nodes.append(tp)
    v_map = {}

    for indx, v in enumerate(mvars):
        v_map[v.name] = indx

    obj = m.getObjective()
    obj_cons = [0] * (nvars + 2)
    obj_node = [0, 0, 0, 0]
    for e in obj:
        vnm = e.vartuple[0].name
        v = obj[e]
        v_indx = v_map[vnm]
        obj_cons[v_indx] = v
        v_nodes[v_indx][0] = v

        # print(v_indx,float(nvars),v_indx/float(nvars),v_nodes[v_indx][ori_start:ori_start+emb_num])

        obj_node[0] += v
        obj_node[1] += 1
    obj_node[0] /= obj_node[1]
    # quit()

    cons = m.getConss()
    new_cons = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        if len(coeff) == 0:
            # print(coeff,c)
            continue
        new_cons.append(c)
    cons = new_cons
    ncons = len(cons)
    cons_map = [[x, len(m.getValsLinear(x))] for x in cons]
    # for i in range(len(cons_map)):
    #   tmp=0
    #   coeff=m.getValsLinear(cons_map[i][0])
    #    for k in coeff:
    #        tmp+=coeff[k]
    #    cons_map[i].append(tmp)
    cons_map = sorted(cons_map, key=lambda x: [x[1], str(x[0])])
    cons = [x[0] for x in cons_map]
    # print(cons)
    # quit()
    A = []
    for i in range(ncons):
        A.append([])
        for j in range(nvars + 2):
            A[i].append(0)
    A.append(obj_cons)
    lcons = ncons
    c_nodes = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        rhs = m.getRhs(c)
        lhs = m.getLhs(c)
        A[cind][-2] = rhs
        sense = 0

        if rhs == lhs:
            sense = 2
        elif rhs >= 1e+20:
            sense = 1
            rhs = lhs

        summation = 0
        for k in coeff:
            v_indx = v_map[k]
            A[cind][v_indx] = 1
            A[cind][-1] += 1
            v_nodes[v_indx][2] += 1
            v_nodes[v_indx][1] += coeff[k] / lcons
            if v_indx == 1066:
                print(coeff[k], lcons)
            v_nodes[v_indx][3] = max(v_nodes[v_indx][3], coeff[k])
            v_nodes[v_indx][4] = min(v_nodes[v_indx][4], coeff[k])
            # v_nodes[v_indx][3]+=cind*coeff[k]
            summation += coeff[k]
        llc = max(len(coeff), 1)
        c_nodes.append([summation / llc, llc, rhs, sense])
    c_nodes.append(obj_node)
    v_nodes = torch.as_tensor(v_nodes, dtype=torch.float32).to(device)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32).to(device)
    b_vars = torch.as_tensor(b_vars, dtype=torch.int32).to(device)

    A = np.array(A, dtype=np.float32)

    A = A[:, :-2]
    A = torch.as_tensor(A).to(device).to_sparse()
    clip_max = [20000, 1, torch.max(v_nodes, 0)[0][2].item()]
    clip_min = [0, -1, 0]

    v_nodes[:, 0] = torch.clamp(v_nodes[:, 0], clip_min[0], clip_max[0])

    maxs = torch.max(v_nodes, 0)[0]
    mins = torch.min(v_nodes, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    v_nodes = v_nodes - mins
    v_nodes = v_nodes / diff
    v_nodes = torch.clamp(v_nodes, 1e-5, 1)
    # v_nodes=position_get_ordered(v_nodes)
    v_nodes = position_get_ordered_flt(v_nodes)

    maxs = torch.max(c_nodes, 0)[0]
    mins = torch.min(c_nodes, 0)[0]
    diff = maxs - mins
    c_nodes = c_nodes - mins
    c_nodes = c_nodes / diff
    c_nodes = torch.clamp(c_nodes, 1e-5, 1)

    return A, v_map, v_nodes, c_nodes, b_vars


def get_BG_from_GRB(ins_name):
    # vars:  [obj coeff, norm_coeff, degree, max coeff, min coeff, Bin?]

    m = gp.read(ins_name)
    ori_start = 6
    emb_num = 15

    mvars = m.getVars()
    mvars.sort(key=lambda v: v.VarName)

    v_map = {}
    for indx, v in enumerate(mvars):
        v_map[v.VarName] = indx

    nvars = len(mvars)

    v_nodes = []
    b_vars = []
    for i in range(len(mvars)):
        tp = [0] * ori_start
        tp[3] = 0
        tp[4] = 1e+20
        # tp=[0,0,0,0,0]
        if mvars[i].vtype() == 'BINARY' or mvars[i].vtype() == 'INTEGER':
            tp[ori_start - 1] = 1
            b_vars.append(i)

        v_nodes.append(tp)

    obj = m.getObjective()
    obj_cons = [0] * (nvars + 2)
    obj_node = [0, 0, 0, 0]

    nobjs = obj.size()
    for i in range(nobjs):
        vnm = obj.getVar(i).VarName
        v = obj.getCoeff(i)
        v_indx = v_map[vnm]
        obj_cons[v_indx] = v
        v_nodes[v_indx][0] = v
        obj_node[0] += v
        obj_node[1] += 1
    obj_node[0] /= obj_node[1]

    cons = m.getConstrs()
    ncons = len(cons)
    lcons = ncons
    c_nodes = []

    A = []
    for i in range(ncons):
        A.append([])
        for j in range(nvars + 2):
            A[i].append(0)
    A.append(obj_cons)
    for i in range(ncons):
        tmp_v = []
        tmp_c = []

        sense = cons[i].Sense
        rhs = cons[i].RHS
        nzs = 0

        if sense == '<':
            sense = 0
        elif sense == '>':
            sense = 1
        elif sense == '=':
            sense = 2

        tmp_c = [0, 0, rhs, sense]
        summation = 0
        tmp_v = [0, 0, 0, 0, 0]
        for v in mvars:
            v_indx = v_map[v.VarName]
            ce = m.getCoeff(cons[i], v)

            if ce != 0:
                nzs += 1
                summation += ce
                A[i][v_indx] = 1
                A[i][-1] += 1

        if nzs == 0:
            continue
        tmp_c[0] = summation / nzs
        tmp_c[1] = nzs
        c_nodes.append(tmp_c)
        for v in mvars:
            v_indx = v_map[v.VarName]
            ce = m.getCoeff(cons[i], v)

            if ce != 0:
                v_nodes[v_indx][2] += 1
                v_nodes[v_indx][1] += ce / lcons
                v_nodes[v_indx][3] = max(v_nodes[v_indx][3], ce)
                v_nodes[v_indx][4] = min(v_nodes[v_indx][4], ce)

    c_nodes.append(obj_node)
    v_nodes = torch.as_tensor(v_nodes, dtype=torch.float32).to(device)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32).to(device)
    b_vars = torch.as_tensor(b_vars, dtype=torch.int32).to(device)

    A = np.array(A, dtype=np.float32)

    A = A[:, :-2]
    A = torch.as_tensor(A).to(device).to_sparse()
    clip_max = [20000, 1, torch.max(v_nodes, 0)[0][2].item()]
    clip_min = [0, -1, 0]

    v_nodes[:, 0] = torch.clamp(v_nodes[:, 0], clip_min[0], clip_max[0])

    maxs = torch.max(v_nodes, 0)[0]
    mins = torch.min(v_nodes, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    v_nodes = v_nodes - mins
    v_nodes = v_nodes / diff
    v_nodes = torch.clamp(v_nodes, 1e-5, 1)
    # v_nodes=position_get_ordered(v_nodes)
    v_nodes = position_get_ordered_flt(v_nodes)

    maxs = torch.max(c_nodes, 0)[0]
    mins = torch.min(c_nodes, 0)[0]
    diff = maxs - mins
    c_nodes = c_nodes - mins
    c_nodes = c_nodes / diff
    c_nodes = torch.clamp(c_nodes, 1e-5, 1)

    return A, v_map, v_nodes, c_nodes, b_vars


def get_a_new2(ins_name):
    epsilon = 1e-6

    # vars:  [obj coeff, norm_coeff, degree, Bin?]
    m = Model('model')
    m.hideOutput(True)
    m.readProblem(filename=ins_name)

    ncons = m.getNConss()
    nvars = m.getNVars()
    cons = m.getConss()
    new_cons = []

    mvars = m.getVars()
    mvars.sort(key=lambda v: v.name)

    v_nodes = []

    b_vars = []

    ori_start = 6
    emb_num = 15

    for i in range(len(mvars)):
        tp = [0] * ori_start
        tp[3] = 0
        tp[4] = 1e+20
        # tp=[0,0,0,0,0]
        if mvars[i].vtype() == 'BINARY' or mvars[i].vtype() == 'INTEGER':
            tp[ori_start - 1] = 1
            b_vars.append(i)

        v_nodes.append(tp)
    v_map = {}

    for indx, v in enumerate(mvars):
        v_map[v.name] = indx

    obj = m.getObjective()
    obj_cons = [0] * (nvars + 2)
    indices_spr = [[], []]
    values_spr = []
    obj_node = [0, 0, 0, 0]
    for e in obj:
        vnm = e.vartuple[0].name
        v = obj[e]
        v_indx = v_map[vnm]
        obj_cons[v_indx] = v
        if v != 0:
            indices_spr[0].append(ncons)
            indices_spr[1].append(v_indx)
            values_spr.append(v)
            # values_spr.append(1)
        v_nodes[v_indx][0] = v

        # print(v_indx,float(nvars),v_indx/float(nvars),v_nodes[v_indx][ori_start:ori_start+emb_num])

        obj_node[0] += v
        obj_node[1] += 1
    obj_node[0] /= obj_node[1]
    # quit()

    cons_vars = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        cons_vars.append(set(coeff.keys()))
        if len(coeff) == 0:
            # print(coeff,c)
            continue
        # new_cons.append(c)
    # cons = new_cons
    # ncons = len(cons)
    # cons_map = [[x, len(m.getValsLinear(x))] for x in cons]
    # sort
    # cons_map = sorted(cons_map, key=lambda x: [x[1], str(x[0])])
    # cons = [x[0] for x in cons_map]
    # cons_name_map = [c.name for c in cons]

    # **Compute coupling degree for each constraint**

    lcons = ncons
    c_nodes = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        rhs = m.getRhs(c)
        lhs = m.getLhs(c)
        # A[cind][-2]=rhs
        sense = 0
        # 0 leq 1 geq 2 eq
        if rhs == lhs:
            sense = 2
        elif rhs >= 1e+20:
            sense = 1
            rhs = lhs

        summation = 0
        for k in coeff:
            v_indx = v_map[k]
            # A[cind][v_indx]=1
            # A[cind][-1]+=1
            if coeff[k] != 0:
                indices_spr[0].append(cind)
                indices_spr[1].append(v_indx)
                values_spr.append(coeff[k])
            v_nodes[v_indx][2] += 1
            v_nodes[v_indx][1] += coeff[k] / lcons
            v_nodes[v_indx][3] = max(v_nodes[v_indx][3], coeff[k])
            v_nodes[v_indx][4] = min(v_nodes[v_indx][4], coeff[k])
            # v_nodes[v_indx][3]+=cind*coeff[k]
            summation += coeff[k]
        llc = max(len(coeff), 1)
        c_nodes.append([summation / llc, llc, rhs, sense])
    c_nodes.append(obj_node)
    v_nodes = torch.as_tensor(v_nodes, dtype=torch.float32).to(device)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32).to(device)
    b_vars = torch.as_tensor(b_vars, dtype=torch.int32).to(device)

    A = torch.sparse_coo_tensor(indices_spr, values_spr, (ncons + 1, nvars)).to(device)
    clip_max = [20000, 1, torch.max(v_nodes, 0)[0][2].item()]
    clip_min = [0, -1, 0]

    v_nodes[:, 0] = torch.clamp(v_nodes[:, 0], clip_min[0], clip_max[0])

    maxs = torch.max(v_nodes, 0)[0]
    mins = torch.min(v_nodes, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    v_nodes = v_nodes - mins
    v_nodes = v_nodes / diff
    v_nodes = torch.clamp(v_nodes, 1e-5, 1)
    # v_nodes=position_get_ordered(v_nodes)
    # v_nodes=position_get_ordered_flt(v_nodes)

    maxs = torch.max(c_nodes, 0)[0]
    mins = torch.min(c_nodes, 0)[0]
    diff = maxs - mins
    # for ks in range(diff.shape[0]):
    #     if diff[ks] == 0:
    #         diff[ks] = 1
    c_nodes = c_nodes - mins
    c_nodes = c_nodes / diff
    c_nodes = torch.clamp(c_nodes, 1e-5, 1)

    return A, v_map, v_nodes, c_nodes, b_vars


def get_bigraph(ins_name: str, fea: bool, v_class_name=None, c_class_name=None, couple=0):
    epsilon = 1e-6
    TaskName = ins_name.split('/')[-1].split('.')[0]
    # vars:  [obj coeff, norm_coeff, degree, Bin?]
    # cons:[ summation, length, rhs, sense]
    m = Model('model')
    m.hideOutput(True)
    m.readProblem(ins_name)

    ncons = m.getNConss()
    nvars = m.getNVars()
    cons = m.getConss()

    new_cons = []
    mvars = m.getVars()
    mvars.sort(key=lambda v: v.name)

    v_nodes = []
    b_vars = []

    ori_start = 6
    emb_num = 15

    v_map = {}
    v_class = [[] for _ in range(len(v_class_name))]
    c_class = [[] for _ in range(len(c_class_name))]  # c_class_name include obj_node

    for indx, v in enumerate(mvars):
        tp = [0] * ori_start
        tp[3] = 0
        tp[4] = 1e+20

        if v.vtype() == 'BINARY' or v.vtype() == 'INTEGER':
            tp[ori_start - 1] = 1
            b_vars.append(indx)

        # Assign variable to v_class based on prefix
        for i, prefix in enumerate(v_class_name):
            if v.name.startswith(prefix):
                v_class[i].append(indx)
                break

        v_nodes.append(tp)
        v_map[v.name] = indx

    obj = m.getObjective()
    obj_cons = [0] * (nvars + 2)
    indices_spr = [[], []]
    values_spr = []
    obj_node = [0, 0, 0, 0]

    for e in obj:
        vnm = e.vartuple[0].name
        v = obj[e]
        v_indx = v_map[vnm]
        obj_cons[v_indx] = v

        if v != 0:
            indices_spr[0].append(ncons)
            indices_spr[1].append(v_indx)
            values_spr.append(v)

        v_nodes[v_indx][0] = v
        obj_node[0] += v
        obj_node[1] += 1

    obj_node[0] /= obj_node[1]

    # coupling_degrees = [0] * ncons
    coupling_degrees = []
    cons_vars = []

    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        cons_vars.append(set(coeff.keys()))

        if len(coeff) == 0:
            continue

    if couple != 0:
        for i, c in enumerate(cons):
            rhs = m.getRhs(c)
            lhs = m.getLhs(c)
            if rhs != lhs:
                for j in range(i + 1, ncons):
                    shared_vars = cons_vars[i].intersection(cons_vars[j])
                    coupling_degrees[i] += len(shared_vars)
                    coupling_degrees[j] += len(shared_vars)

    lcons = ncons
    c_nodes = []
    split_index = None
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        rhs = m.getRhs(c)
        lhs = m.getLhs(c)
        sense = 0  # leq

        if rhs == lhs:
            sense = 2
        elif rhs >= 1e+20:
            sense = 1  # geq
            rhs = lhs

        summation = 0

        # for v_indx in range(nvars):
        #     if v_indx in coeff:
        #         value = coeff[v_indx]
        #     else:
        #         value = 0
        #     indices_spr[0].append(cind)
        #     indices_spr[1].append(v_indx)
        #     values_spr.append(value)

        for k in coeff:
            v_indx = v_map[k]
            if coeff[k] != 0:
                indices_spr[0].append(cind)
                indices_spr[1].append(v_indx)
                values_spr.append(coeff[k])
            v_nodes[v_indx][2] += 1
            v_nodes[v_indx][1] += coeff[k] / lcons
            v_nodes[v_indx][3] = max(v_nodes[v_indx][3], coeff[k])
            v_nodes[v_indx][4] = min(v_nodes[v_indx][4], coeff[k])
            summation += coeff[k]
        llc = max(len(coeff), 1)
        c_nodes.append([summation / llc, llc, rhs, sense])
        # Assign constraint to c_class based on prefix
        if TaskName.startswith("CA"):
            def is_consecutive_x_vars(var_names):
                ids = []
                for name in var_names:
                    if not name.startswith("x"):
                        return False
                    try:
                        idx = int(name[1:])
                        if idx >= 10:
                            return False  # only check one-digit suffix
                        ids.append(idx)
                    except:
                        return False
                ids.sort()
                return all(ids[i] + 1 == ids[i + 1] for i in range(len(ids) - 1))

            if cind < 1985:
                c_class[0].append(cind)
            else:
                var_names = list(coeff.keys())
                if split_index is None:
                    if is_consecutive_x_vars(var_names):
                        split_index = cind
                if split_index is not None and cind >= split_index:
                    c_class[1].append(cind)
                else:
                    c_class[0].append(cind)
        elif TaskName.startswith("MMCN2"):
            var_names = list(coeff.keys())
            has_x = any(name.startswith("x_") for name in var_names)
            has_v = any(name.startswith("v_") for name in var_names)
            has_z = any(name.startswith("z_") for name in var_names)
            if has_x and not has_v and not has_z and sense == 2:  # x =
                if abs(rhs - 1.0) < 1e-4:
                    c_class[0].append(cind)
            elif has_v and has_z and not has_x and sense == 0:  # vz <
                c_class[1].append(cind)
            elif has_v and has_z and not has_x and sense == 1:  # vz >
                c_class[2].append(cind)
            elif has_x and has_v and not has_z and sense == 2:  # xv =
                c_class[3].append(cind)
            elif has_z and not has_x and not has_v and sense == 0:  # z <
                c_class[4].append(cind)
            elif has_x and has_z and not has_v and sense == 1:  # xz >
                c_class[5].append(cind)
            elif has_x and has_z and not has_v and sense == 0:  # xz <
                c_class[6].append(cind)

        elif TaskName.startswith("MMCN"):
            if sense == 2:
                c_class[0].append(cind)
            else:
                c_class[1].append(cind)
        elif TaskName.startswith("IS") or TaskName.startswith("MVC"):
            c_class[0].append(cind)
        else:
            for i, prefix in enumerate(c_class_name):
                if c.name.startswith(prefix):
                    c_class[i].append(cind)
                    break

    c_nodes.append(obj_node)
    obj_node = torch.as_tensor([obj_node], dtype=torch.float32).to(device)
    c_class[-1].append(len(c_nodes) - 1)
    A = torch.sparse_coo_tensor(indices_spr, values_spr, (ncons + 1, nvars)).to(device)
    if fea:
        constraint_similarities_tensor = get_obj_cos_by_scip(m, A, v_map)
        constraint_tightness_tensor, normalized_duals_tensor = solve_LP_relax_by_gp(m, ins_name)

    v_nodes = torch.as_tensor(v_nodes, dtype=torch.float32).to(device)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32).to(device)
    b_vars = torch.as_tensor(b_vars, dtype=torch.int32).to(device)
    # get new c_nodes
    if fea:
        new_c_nodes = torch.cat([c_nodes[:ncons],
                                 constraint_similarities_tensor,
                                 constraint_tightness_tensor,
                                 normalized_duals_tensor], dim=1)
        obj_node_extended = torch.cat([obj_node, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).to(device)])
        c_nodes = torch.cat([new_c_nodes, obj_node_extended.unsqueeze(0)], dim=0)

    clip_max = [20000, 1, torch.max(v_nodes, 0)[0][2].item()]
    clip_min = [0, -1, 0]
    v_nodes[:, 0] = torch.clamp(v_nodes[:, 0], clip_min[0], clip_max[0])
    maxs = torch.max(v_nodes, 0)[0]
    mins = torch.min(v_nodes, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    v_nodes = v_nodes - mins
    v_nodes = v_nodes / diff
    v_nodes = torch.clamp(v_nodes, 1e-5, 1)

    maxs = torch.max(c_nodes, 0)[0]
    mins = torch.min(c_nodes, 0)[0]
    diff = maxs - mins
    # for ks in range(diff.shape[0]):
    #     if diff[ks] == 0:
    #         diff[ks] = 1
    c_nodes = c_nodes - mins
    c_nodes = c_nodes / diff

    # obj_node = F.normalize(obj_node, p=2, dim=1)
    # c_nodes = torch.cat([c_nodes, obj_node], dim=0)

    c_nodes = torch.clamp(c_nodes, 1e-5, 1)

    return A, v_map, v_nodes, c_nodes, b_vars, v_class, c_class, coupling_degrees


def get_pattern(json_path, task):
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Check if task exists in the JSON
    if "task" not in data or task not in data["task"]:
        print(f"Task '{task}' not found in the JSON file.")
        return None, None

    task_data = data["task"][task]

    # Extract variable type names
    v_class_name = []
    if "variable_type" in task_data:
        for var_name in task_data["variable_type"].keys():
            v_class_name.append(var_name)

    # Extract constraint type names
    c_class_name = []
    if "constraint_type" in task_data:
        for con_name in task_data["constraint_type"].keys():
            c_class_name.append(con_name)

    return v_class_name, c_class_name


def map_model_to_filtered_indices(m):
    # m --->  mask_m (mask "=" con)
    # m gurobi model
    cons = m.getConstrs()  # get all constraints in the model

    # Build mapping: model index -> filtered index
    model_to_filtered_index = {}
    filtered_index_to_model = {}
    filtered_idx = 0  # track filtered index

    for i, c in enumerate(cons):
        if c.Sense in ['<', '>']:
            model_to_filtered_index[i] = filtered_idx
            filtered_index_to_model[filtered_idx] = i
            filtered_idx += 1
        else:
            continue

    return model_to_filtered_index, filtered_index_to_model


def solve_LP_relax_by_gp(m, ins_name: str) -> tuple[Tensor, Tensor]:
    epsilon = 1e-6

    # Load original model
    original_model = gp.read(ins_name)
    relaxed_model = original_model.copy()

    # Convert all integer/binary variables to continuous
    for var in relaxed_model.getVars():
        if var.VType == GRB.BINARY or var.VType == GRB.INTEGER:
            var.VType = GRB.CONTINUOUS

            # Save original number of constraints
    n_cons = len(original_model.getConstrs())

    # Disable output
    relaxed_model.setParam('OutputFlag', 0)

    # Solve relaxed model
    relaxed_model.optimize()

    constraint_tightness = []
    dual_values = []

    # Check whether the solve succeeded
    if relaxed_model.Status == GRB.OPTIMAL:
        # Get all constraints
        constrs = relaxed_model.getConstrs()

        # Ensure we do not exceed original constraint count
        constrs = constrs[:n_cons]

        for c in constrs:
            # Get constraint slack
            slack = relaxed_model.getAttr('Slack', [c])[0]

            # Get RHS value
            rhs = c.RHS

            # Get constraint sense
            sense = c.Sense

            # Check whether the constraint is tight
            is_tight = False
            if sense == '=':  # equality constraints are always tight
                is_tight = True
            elif sense == '<' and abs(slack) < epsilon:
                is_tight = True
            elif sense == '>' and abs(slack) < epsilon:
                is_tight = True

            constraint_tightness.append(float(is_tight))

            # Get dual value (Pi attribute)
            dual = c.Pi
            dual_values.append(dual)

            # Normalize dual values
        if dual_values:
            max_dual = max(abs(d) for d in dual_values)
            if max_dual > epsilon:
                normalized_duals = [d / max_dual for d in dual_values]
            else:
                normalized_duals = [0.0] * len(dual_values)
        else:
            normalized_duals = [0.0] * n_cons
    else:
        # If solve fails, use default values
        print(f"Warning: model solve failed, status: {relaxed_model.Status}")
        constraint_tightness = [0.0] * n_cons
        normalized_duals = [0.0] * n_cons

        # Ensure we have enough constraint features
    if len(constraint_tightness) < n_cons:
        print(f"Warning: found fewer constraints ({len(constraint_tightness)}) than expected ({n_cons})")
        constraint_tightness.extend([0.0] * (n_cons - len(constraint_tightness)))
        normalized_duals.extend([0.0] * (n_cons - len(normalized_duals)))

        # Convert lists to tensors
    normalized_duals_tensor = torch.tensor(normalized_duals[:n_cons], dtype=torch.float32).to(device).unsqueeze(1)
    constraint_tightness_tensor = torch.tensor(constraint_tightness[:n_cons], dtype=torch.float32).to(
        device).unsqueeze(1)

    # Release model resources
    relaxed_model.dispose()
    original_model.dispose()

    return constraint_tightness_tensor, normalized_duals_tensor


def get_obj_cos_by_scip(m, A, v_map):
    n_cons = m.getNConss()
    n_vars = m.getNVars()
    objective_coefficients = np.zeros(n_vars)
    obj = m.getObjective()
    for e in obj:
        vnm = e.vartuple[0].name
        v = obj[e]
        v_indx = v_map[vnm]
        objective_coefficients[v_indx] = v
    from sklearn.metrics.pairwise import cosine_similarity
    A_numpy = A.to_dense().cpu().numpy()[:n_cons]  # exclude last row (objective)
    obj_coeffs_reshaped = objective_coefficients.reshape(1, -1)
    cosine_sims = cosine_similarity(A_numpy, obj_coeffs_reshaped)
    constraint_similarities = [cosine_sims[con_ind, 0] for con_ind in range(n_cons)]

    constraint_similarities_tensor = torch.tensor(constraint_similarities, dtype=torch.float32).to(device).unsqueeze(1)
    return constraint_similarities_tensor
