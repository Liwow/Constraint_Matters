import json
import os
from sklearn.cluster import KMeans
import re
from transformers import T5Tokenizer, T5EncoderModel
import torch
import torch.nn as nn
import torch_geometric
import pickle
import numpy as np

import config
from GCN import getPE, SELayerGraph
import pyscipopt as scip
import utils

DEVICE = config.DEVICE


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self):
        super().__init__("add")
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """

        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        b = torch.cat([self.post_conv_module(output), right_features], dim=-1)
        a = self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        # node_features_i,the node to be aggregated
        # node_features_j,the neighbors of the node i

        # print("node_features_i:",node_features_i.shape)
        # print("node_features_j",node_features_j.shape)
        # print("edge_features:",edge_features.shape)

        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )

        return output


class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
            self,
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,
            v_class,
            c_class,

    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.v_class = v_class
        self.c_class = c_class

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        # elif key == "v_class":
        #     return [self.variable_features.size(0)] * len(self.v_class)
        # elif key == "c_class":
        #     return [self.constraint_features.size(0)] * len(self.c_class)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GNNPolicy_ancon(torch.nn.Module):
    def __init__(self, TaskName, position=False, dropout=0.3):
        super().__init__()
        emb_size = 64
        # 4 16 7 19
        cons_nfeats = 4 if not position else 16
        edge_nfeats = 1
        var_nfeats = 6 if not position else 18
        self.temperature = 0.6
        self.dropout = nn.Dropout(dropout)
        self.se_con = SELayerGraph(emb_size)
        self.se_var = SELayerGraph(emb_size)
        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()

        self.conv_c_to_v3 = BipartiteGraphConvolution()

        self.con_mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

        self.var_mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )
        self.con_mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

        self.anchor_gnn = AnchorGNN(TaskName, emb_size).to(DEVICE)

    def forward(
            self, constraint_features, edge_indices, edge_features, variable_features, v_class, c_class,
            get_logits=False, con=True, c_mask=None
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        constraint_features = self.se_con(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        variable_features = self.se_var(variable_features)

        constraint_features = self.dropout(constraint_features)
        variable_features = self.dropout(variable_features)

        # Two half convolutions
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )

        variable_features, constraint_features = self.anchor_gnn(
            variable_features, constraint_features, v_class, c_class
        )

        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )

        variable_features = self.conv_c_to_v3(
            constraint_features, edge_indices, edge_features, variable_features
        )

        if c_mask is not None:
            mask_constraint_features = constraint_features[c_mask == 1]
        else:
            mask_constraint_features = constraint_features

        con_output = torch.sigmoid(self.con_mlp(mask_constraint_features).squeeze(-1) / self.temperature)
        var_output = torch.sigmoid(self.var_mlp(variable_features).squeeze(-1) / self.temperature)
        if get_logits:
            return variable_features, constraint_features, var_output
        elif con:
            return var_output, con_output
        else:
            return var_output


class GraphDataset_ancon(torch_geometric.data.Dataset):

    def __init__(self, sample_files, task, position=False, method="kmeans"):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.position = position
        self.task = task
        self.method = method

    def len(self):
        return len(self.sample_files)

    def process_sample(self, filepath):
        BGFilepath, solFilePath = filepath
        with open(BGFilepath, "rb") as f:
            bgData = pickle.load(f)
        try:
            with open(solFilePath, 'rb') as f:
                solData = pickle.load(f)
        except Exception as e:
            print(f"Error: {e}, file: {solFilePath}")

        BG = bgData
        varNames = solData['var_names']

        sols = solData['sols'][:50]  # [0:300]
        objs = solData['objs'][:50]  # [0:300]
        slacks = solData['slacks'][:50]
        sols = np.round(sols, 0)
        return BG, sols, objs, varNames, slacks

    def get_critical(self, path, con_num, method="llm", n=15):
        file = os.path.basename(path[0])[:-3]
        task_name = path[0].split('/')[-3]
        instance_path = './instance/train/' + task_name + '/' + file
        model = scip.Model()
        model.setParam('display/verblevel', 0) 
        model.setParam('display/freq', 0)
        model.readProblem(instance_path)
        domain_mask = [1] * con_num
        domain_mask.append(0)
        critical_mask = []
        num_vars_per_constr = []
        if method == "fix":
            for i, constr in enumerate(model.getConss()):
                rhs = model.getRhs(constr)
                lhs = model.getLhs(constr)
                if lhs != rhs:
                    lin_expr = model.getValsLinear(constr)
                    num_vars = len(lin_expr)
                    num_vars_per_constr.append(num_vars)
                    critical_mask.append(1 if num_vars <= n else 0)
        elif method == "kmeans":
            tmp_list = []
            for i, constr in enumerate(model.getConss()):
                rhs = model.getRhs(constr)
                lhs = model.getLhs(constr)
                if lhs != rhs:
                    tmp_list.append(1)
                    lin_expr = model.getValsLinear(constr)
                    n_var_in_expr = len(lin_expr)
                    num_vars_per_constr.append(n_var_in_expr)
                else:
                    tmp_list.append(0)
            sparse_cluster, critical_labels = utils.get_label_by_kmeans(num_vars_per_constr)
            category_iter = iter(critical_labels)
            critical_labels = [next(category_iter) if pos == 1 else 0 for pos in tmp_list]
            critical_mask = [1 if c_label == 1 else 0 for c_label in critical_labels]

        elif method == "llm":
            split_index = None
            for i, constr in enumerate(model.getConss()):
                conName = constr.name
                rhs = model.getRhs(constr)
                lhs = model.getLhs(constr)
                if conName.startswith("workload_ct_") or conName.startswith("worker_capacity_"):
                    critical_mask.append(1)
                elif conName.startswith("deficit_ct") or conName.startswith("supply_ct"):
                    critical_mask.append(1)
                elif bool(re.match(r'^c\d+$', conName)) and self.task.startswith("CA"):
                    # critical_mask.append(1)
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
                    if i < 1985:
                        critical_mask.append(0)
                        domain_mask[i] = 0
                    else:
                        coeff = model.getValsLinear(constr)
                        var_names = list(coeff.keys())
                        if split_index is None:
                            if is_consecutive_x_vars(var_names):
                                split_index = i
                        if split_index is not None and i >= split_index:
                            critical_mask.append(1)
                        else:
                            critical_mask.append(0)
                            domain_mask[i] = 0
                elif bool(re.match(r'^c\d+$', conName)) and not self.task.startswith("CA"):
                    critical_mask.append(1)
                elif self.task.startswith("MMCN2") and lhs != rhs:
                    critical_mask.append(1)
                else:
                    critical_mask.append(0)
                    domain_mask[i] = 0

        elif method == "score":
            score_list = []
            conType_list = []
            for i, constr in enumerate(model.getConss()):
                rhs = model.getRhs(constr)
                lhs = model.getLhs(constr)
                constr_type = utils.find_type(model, constr)
                if lhs != rhs:
                    lin_expr = model.getValsLinear(constr)
                    n_var_in_expr = len(lin_expr)
                    r = utils.score(n_var_in_expr, constr_type)
                else:
                    r = 0
                score_list.append(r)
                conType_list.append(constr_type)
            conType_list.append(99)
            return domain_mask, conType_list
        else:
            print("no select way")
            return None, None
        critical_mask.append(0)
        return domain_mask, critical_mask

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """

        # nbp, sols, objs, varInds, varNames = self.process_sample(self.sample_files[index])
        BG, sols, objs, varNames, slacks = self.process_sample(self.sample_files[index])

        A, v_map, v_nodes, c_nodes, b_vars, v_class, c_class = BG

        n_v_class = len(v_class)
        n_c_class = len(c_class)

        constraint_features = c_nodes
        edge_indices = A._indices()

        variable_features = v_nodes
        edge_features = A._values().unsqueeze(1)

        variable_features = getPE(variable_features, self.position)
        constraint_features = getPE(constraint_features, self.position)

        constraint_features[torch.isnan(constraint_features)] = 1 if not self.task == "CA" else -1

        v_class_list = utils.convert_class_to_labels(v_class, variable_features.shape[0])
        c_class_list = utils.convert_class_to_labels(c_class, constraint_features.shape[0])

        domain_mask, critical_mask = self.get_critical(
            self.sample_files[index], len(slacks[1]) - 1, method=self.method
        )
        critical_mask_tensor = torch.tensor(critical_mask, dtype=torch.int)
        domain_mask_tensor = torch.tensor(domain_mask, dtype=torch.int)
        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features.float()),
            torch.FloatTensor(variable_features),
            torch.LongTensor(v_class_list),
            torch.LongTensor(c_class_list),
        )

        c_mask = []
        c_labels = []
        for slack in slacks:  # slack include objective function
            mask = [1 if con[2] in ['<', '>'] else 0 for con in slack]
            labels = [1 if abs(con[1]) <= 1e-8 and con[2] in ['<', '>'] else 0 for con in slack]
            mask = torch.tensor(mask, dtype=torch.int)
            labels = torch.tensor(labels, dtype=torch.int)
            # sparse_cluster, labels = utils.get_label_by_kmeans(coupling_degrees)
            # critical_list_by_coupling = [1 if label == sparse_cluster else 0 for label in labels]
            # critical_list = critical_list_by_sparse
            # critical_list_tensor = torch.tensor(critical_list, dtype=torch.int)
            if self.method == "score":
                critical_mask_tensor = labels * critical_mask_tensor
                k = min(labels.sum(), int(0.5 * labels.size(0)))
                sort_indices = torch.topk(critical_mask_tensor, k).indices
                labels = [1 if i in sort_indices else 0 for i, label in enumerate(labels)]
            else:
                labels = torch.bitwise_and(labels, critical_mask_tensor)
            mask = torch.bitwise_and(mask, domain_mask_tensor)
            labels = labels[mask == 1]
            c_mask.append(mask)
            c_labels.append(labels.float())
        graph.c_labels = torch.cat(c_labels).reshape(-1)
        graph.c_mask = torch.cat(c_mask).reshape(-1)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.solutions = torch.FloatTensor(sols).reshape(-1)

        graph.objVals = torch.FloatTensor(objs)
        graph.ncons = len(slacks[1])
        graph.nlabels = c_labels[1].size()[0]
        graph.nsols = sols.shape[0]
        graph.ntvars = variable_features.shape[0]
        graph.varNames = varNames
        varname_dict = {}
        varname_map = []
        i = 0
        for iter in varNames:
            varname_dict[iter] = i
            i += 1
        for iter in v_map:
            varname_map.append(varname_dict[iter])

        varname_map = torch.tensor(varname_map)

        graph.varInds = [[varname_map], [b_vars]]

        return graph


def encode_texts(path, text_list, api=False):
    if api:
        import os, openai
        key = ""
        client = openai.OpenAI(
                api_key=key,  
                base_url=""  
                )
        def embed(text: str, dims=None):
            text = text.replace("\n", " ")
            resp = client.embeddings.create(
                input=[text],
                model="text-embedding-3-large",
                dimensions=dims  
            )
            return resp.data[0].embedding
        return torch.stack([torch.tensor(embed(txt, dims=64),
                                    dtype=torch.float32, device=DEVICE)
                       for txt in text_list])
    else:
        feats = []
        tokenizer = T5Tokenizer.from_pretrained(path, legacy=False)
        text_encoder = T5EncoderModel.from_pretrained(path).to(DEVICE)
        for txt in text_list:
            tok = tokenizer(txt, return_tensors="pt").to(DEVICE)
            out = text_encoder(**tok).last_hidden_state[:, -1, :]
            feats.append(out)
        return torch.stack(feats)

def get_by_semantics(task, path, api = False):
    # get var_fea, con_fea, edge, edge_feature
    var_fea = []
    con_fea = []
    edge_idx = []
    edge_features = []
    config_path = './task_config.json'
    with open(config_path, 'r') as f:
        config_json = json.load(f)
    if "task" not in config_json or task not in config_json["task"]:
        raise ValueError(f"Task '{task}' not found in the JSON configuration.")
    task_details = config_json["task"][task]
    task_description = task_details.get("task_description", "No description available")
    task_text = f"task: {task}\ntask_description: {task_description}\n\n"
    var_text = []
    con_text = []
    if "variable_type" in task_details:
        for var_name, var_details in task_details["variable_type"].items():
            var_description = var_details.get("description", "No description available")
            var_type = var_details.get("type", "No type specified")
            var_index = var_details.get("index", "No index specified")
            var_range = var_details.get("range", "No range specified")
            var_constraints = var_details.get("constraints", "No constraints specified")

            var_text.append(
                task_text + f"variable_name: {var_name}, variable_index: {var_index}, variable_type: {var_type}, variable_description: {var_description}, variable_range: {var_range}, variable_constraints: {var_constraints}\n\n")
    if "constraint_type" in task_details:
        for con_name, con_details in task_details["constraint_type"].items():
            con_description = con_details.get("description", "No description available")
            con_type = con_details.get("type", "No type specified")
            con_index = con_details.get("index", "No index specified")
            con_expression = con_details.get("expression", "No expression specified")
            con_constraints = con_details.get("constraints", "No constraints specified")

            con_text.append(
                task_text + f"constraint_name: {con_name}, constraint_index: {con_index}, constraint_type: {con_type}, constraint_description: {con_description}, constraint_expression: {con_expression}, constraint_constraints: {con_constraints}\n\n")
        
    var_fea = encode_texts(path,var_text, api=api) 
    con_fea = encode_texts(path,con_text, api=api)

    edges = task_details["edges"]
    for edge in edges:
        source = int(edge["source"])
        target = int(edge["target"])
        edge_feature = edge["feature"]
        edge_idx.append([source, target])
        edge_features.append(edge_feature)

    edge_idx = torch.tensor(edge_idx).t()
    edge_features = torch.tensor(edge_features).unsqueeze(0)

    return var_fea, con_fea, edge_idx, edge_features

    


class AnchorGNN(torch.nn.Module):
    def __init__(self, task, emb_size=64):
        super().__init__()
        api = True
        self.emb_size = emb_size
        self.layer_norm = nn.LayerNorm(self.emb_size)
        path = "../../local_models/t5-base"
        var_fea, con_fea, edge, edge_feature = get_by_semantics(task, path,api=api)
        self.v_sem_fea = var_fea
        self.v_n_class = var_fea.shape[0]
        self.c_sem_fea = con_fea
        self.c_n_class = con_fea.shape[0]
        self.edge_idx = edge
        self.edge_fea = edge_feature

        self.self_att = torch.nn.MultiheadAttention(self.emb_size, num_heads=4, batch_first=False)

        text_dim = 64 if api else 768
        
        self.proj_var = torch.nn.Sequential(
            torch.nn.Linear(text_dim, 2 * emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_size, emb_size),
        )

        self.proj_con = torch.nn.Sequential(
            torch.nn.Linear(text_dim, 2 * emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_size, emb_size),
        )
        self.dropout = nn.Dropout(0.3)


        self.anchor = Anchor(self.v_n_class, self.c_n_class, emb_size)

    def forward(self, v, c, v_class, c_class, mpnn=False):
        # v_class: [[indices],...,[indices]]   get batch
        v_class = v_class.to(v.device)
        c_class = c_class.to(c.device)
        v = self.layer_norm(v)
        c = self.layer_norm(c)
        v_sem_fea = self.proj_var(self.v_sem_fea)
        c_sem_fea = self.proj_con(self.c_sem_fea)

        if not mpnn:
            fea = torch.concat([v_sem_fea, c_sem_fea], dim=0)
            fea_sem = self.self_att(fea, fea, fea)[0]
            fea_sem = self.layer_norm(fea_sem + fea).squeeze(1)
            v_sem = fea_sem[:self.v_n_class]
            c_sem = fea_sem[-self.c_n_class:]

        else:
            reversed_edge_indices = torch.stack([self.edge_idx[1], self.edge_idx[0]], dim=0)
            c_sem = self.conv_high_v_to_c(v_sem_fea, reversed_edge_indices, self.edge_fea, c_sem_fea)
            v_sem = self.conv_high_c_to_v(c_sem_fea, reversed_edge_indices, self.edge_fea, v_sem_fea)

        v_new, c_new = self.anchor(v, c, v_sem, c_sem, v_class, c_class)
        # v_new = self.se_var(v_new)
        # c_new = self.se_con(c_new)

        return v_new, c_new


class Anchor(nn.Module):
    def __init__(self, v_n, c_n, emb_size=64):
        super().__init__()
        self.emb_size = emb_size
        self.v_n = v_n
        self.c_n = c_n
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(self.emb_size)
        self.cross_att_var = nn.MultiheadAttention(self.emb_size, num_heads=4, batch_first=False)
        self.cross_att_con = nn.MultiheadAttention(self.emb_size, num_heads=4, batch_first=False)

        self.send_var = nn.Linear(emb_size, emb_size)
        self.send_con = nn.Linear(emb_size, emb_size)

        self.rec_var = nn.Linear(2 * emb_size, emb_size)
        self.rec_con = nn.Linear(2 * emb_size, emb_size)

        self.gate_v = nn.Linear(2 * emb_size, emb_size)
        self.gate_c = nn.Linear(2 * emb_size, emb_size)

        self.norm = nn.LayerNorm(emb_size)

    def forward(self, v, c, v_sem, c_sem, v_class, c_class):


        v_s = self.send_var(v)  # [num_vars, emb_size]
        c_s = self.send_con(c)  # [num_cons, emb_size]

        # 2) 初始化更新容器
        v_updates = torch.zeros_like(v_s)  # [num_vars, emb_size]
        c_updates = torch.zeros_like(c_s)  # [num_cons, emb_size]

        # ================ variable node ================
        for v_i in range(self.v_n):
            v_i_indices = torch.nonzero(v_class == v_i, as_tuple=False).squeeze(1)
            if len(v_i_indices) == 0:
                continue

 
            v_i_fea = v_s[v_i_indices]

            v_i_fea_for_attn = v_i_fea.unsqueeze(1)  # shape = [X, 1, emb_size]


            v_i_sem = v_sem[v_i]  # [emb_size]

            v_i_sem_for_attn = v_i_sem.unsqueeze(0).unsqueeze(1)  # [1, 1, emb_size]

            # 3) Cross-attention:
            #   query = v_i_sem_for_attn, key = v_i_fea_for_attn, value = v_i_fea_for_attn
            v_i_final = self.cross_att_var(
                v_i_sem_for_attn, v_i_fea_for_attn, v_i_fea_for_attn
            )[0]  # shape [1, 1, emb_size]


            v_i_final = v_i_final.squeeze(1).squeeze(0)

            # 4)
            cat_fea = torch.cat([v_i_sem, v_i_final], dim=-1)  # [2*emb_size]
            new_fea = self.rec_var(cat_fea)  # [emb_size]


            old_fea = v_i_fea.mean(dim=0)  # shape = [emb_size]

            concat_gate = torch.cat([old_fea, new_fea], dim=-1)  # [2*emb_size]
            gate_raw = self.gate_v(concat_gate)  # [emb_size]
            gate = torch.sigmoid(gate_raw)  # [emb_size], each dimension in [0,1]
            # 融合
            fused = gate * old_fea + (1 - gate) * new_fea
            fused = self.norm(fused) 


            v_updates[v_i_indices] = fused * v[v_i_indices]

        # ================ constraint node ================
        for c_i in range(self.c_n):
            c_i_indices = torch.nonzero(c_class == c_i, as_tuple=False).squeeze(1)
            if len(c_i_indices) == 0:
                continue

            c_i_fea = c_s[c_i_indices]
            c_i_fea_for_attn = c_i_fea.unsqueeze(1)
            c_i_sem = c_sem[c_i]
            c_i_sem_for_attn = c_i_sem.unsqueeze(0).unsqueeze(1)

            c_i_final = self.cross_att_con(
                c_i_sem_for_attn, c_i_fea_for_attn, c_i_fea_for_attn
            )[0].squeeze(1).squeeze(0)

            cat_fea = torch.cat([c_i_sem, c_i_final], dim=-1)
            new_fea = self.rec_con(cat_fea)

            old_fea = c_i_fea.mean(dim=0)  # [emb_size]
            concat_gate = torch.cat([old_fea, new_fea], dim=-1)
            gate_raw = self.gate_c(concat_gate)
            gate = torch.sigmoid(gate_raw)

            fused = gate * old_fea + (1 - gate) * new_fea
            fused = self.norm(fused)

            c_updates[c_i_indices] = fused * c[c_i_indices]

        return v_updates, c_updates
