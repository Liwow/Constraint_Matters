import torch


CONSTRAINT_TYPES = {
    "Singleton": 6,  # with a single variable
    "Aggregations": 4,  # of the type ax+by=c
    "Precedence": 7,  #   of the type ax−ay≤b where x and y must have the same type
    "Variable Bound": 9,  # of the form  ax+by≤c,x∈{0,1}
    "SetPartitioning": 1,  #  of the form ∑xi=1,xi∈{0,1}∀i
    "SetPacking": 11,  # of the form  ∑xi≤1,xi∈{0,1}∀i
    "SetCovering": 14,  # of the form  ∑xi≥1,xi∈{0,1}∀i
    "Cardinality": 2,  # of the form  ∑xi=k,xi∈{0,1}∀i,k≥2
    "Invariant Knapsack":12,  # of the form  ∑xi≤b,xi∈{0,1}∀i,b∈N≥2
    "Equation Knapsack": 3,  # of the form  ∑ai*xi=b,xi∈{0,1}∀i,b∈N≥2
    "BinPacking": 10,  # of the form  ∑ai*xi+ax≤a,x,xi∈{0,1}∀i,a∈N≥2
    "Knapsack": 13,  #  of the form  ∑ak*xk≤b,xi∈{0,1}∀i,b∈N≥2
    "Mixed Binary": 8,  #  of the form ∑akxk+∑pjsj {≤,=} b,xi∈{0,1}∀i,sj cont. ∀j
    "General Linear": 5,  # with no special structure
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TimeLimit = 800
Threads = 1

