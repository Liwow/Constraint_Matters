import os
import random
import time
from multiprocessing import Process, Queue
import networkx as nx
import random
from pyscipopt import Model as ScipModel
import ecole
import os

os.environ["LD_DEBUG"] = "libs"

# prefix = '../../../../project/predict_and_search/'
prefix = './'


# prefix  = ''
import networkx as nx
import ecole.scip


class MVCGenerator:
    """Custom Minimum Vertex Cover (MVC) generator."""
    def __init__(self, num_nodes=6000, target_edges=None, barabasi_m=None):
        self.num_nodes = num_nodes
        self.target_edges = target_edges

        if target_edges is not None:
            estimated_m = max(1, int(target_edges / num_nodes))
            self.barabasi_m = estimated_m
        else:
            self.barabasi_m = barabasi_m if barabasi_m is not None else 5

        self._base_seed = None

    def seed(self, seed):
        self._base_seed = seed
        random.seed(seed)

    def __next__(self):
        if self._base_seed is None:
            raise ValueError("Please call generator.seed(...) to set the random seed first.")

            # Generate a BA graph
        G = nx.barabasi_albert_graph(
            n=self.num_nodes,
            m=self.barabasi_m,
            seed=self._base_seed
        )
        self._base_seed += 1

        # If target edge count is set and current graph is too dense, randomly remove edges.
        current_edges = G.number_of_edges()
        if self.target_edges is not None and current_edges > self.target_edges:
            edges_to_remove = random.sample(list(G.edges()), current_edges - self.target_edges)
            G.remove_edges_from(edges_to_remove)

            # Print actual edge count (for debugging)
        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

        # Build model using PySCIPOpt
        model = ScipModel("mvc")

        # Add variables
        x_vars = {}
        for node in G.nodes():
            x_vars[node] = model.addVar(f"x_{node}", vtype="B")

            # Set objective
        model.setObjective(sum(x_vars.values()), "minimize")

        # Add constraints
        for u, v in G.edges():
            model.addCons(x_vars[u] + x_vars[v] >= 1)

            # Save model to a temporary file
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.lp', delete=False)
        model.writeProblem(temp_file.name)
        temp_file.close()

        # Load model with Ecole
        ecole_model = ecole.scip.Model.from_file(temp_file.name)

        # Remove temporary file
        import os
        os.unlink(temp_file.name)

        return ecole_model

    def __iter__(self):
        return self

def generate_single_instance(n, queue, istrain, size, generator, seed):
    generator.seed(seed)
    while True:
        i = queue.get()
        if i is None:
            break  # No more tasks

        instance = next(generator)
        instance_dir = prefix + f"instance/{istrain}/{size}"
        os.makedirs(instance_dir, exist_ok=True)
        instance_path = os.path.join(instance_dir, f"{size}_{n + i}.lp")
        instance.write_problem(instance_path)
        print(f"Generated instance #{n + i}: {instance_path}")


def generate_instances(num_instances, istrain, size, epoch=0):
    if size == "CF":
        generator = ecole.instance.CapacitatedFacilityLocationGenerator(50, 100)
    elif size == "IS": # 4000
        generator = ecole.instance.IndependentSetGenerator(6000)
    elif size == "IS_hard": # 6000
        generator = ecole.instance.IndependentSetGenerator(9000)
    elif size == "CA":
        if epoch == 0:
            generator = ecole.instance.CombinatorialAuctionGenerator(300, 1500)
        elif epoch == 1:
            generator = ecole.instance.CombinatorialAuctionGenerator(2000, 4000)
    elif size == "CA_hard":
        if epoch == 0:
            generator = ecole.instance.CombinatorialAuctionGenerator(600, 3000)
        elif epoch == 1:
            generator = ecole.instance.CombinatorialAuctionGenerator(3000, 6000)
    elif size == "SC":
        generator = ecole.instance.SetCoverGenerator(3000, 5000)
    elif size == "MVC":
        # MVC: medium difficulty, default barabasi_m=5
        generator = MVCGenerator(num_nodes=4000)
    elif size == "MVC_hard":
        # MVC_hard: 6000 nodes
        generator = MVCGenerator(num_nodes=6000)
    else:
        raise ValueError("Invalid type")
    base_seed = random.randint(0, 2 ** 16 - 1)
    # seed = int(time.time())
    observation_function = ecole.observation.MilpBipartite()

    # Create a queue to hold tasks
    task_queue = Queue()
    # n = epoch * num_instances
    n = 0
    # Add tasks to queue
    for i in range(num_instances):
        task_queue.put(i)

    # Number of worker processes
    num_workers = 4

    # Create worker processes
    workers = []
    for worker_id in range(num_workers):
        seed = base_seed + worker_id
        worker = Process(target=generate_single_instance,
                         args=(n, task_queue, istrain, size, generator, seed))
        workers.append(worker)
        worker.start()

    # Add None to the queue to signal workers to exit
    for _ in range(num_workers):
        task_queue.put(None)

    # Wait for all worker processes to finish
    for worker in workers:
        worker.join()


if __name__ == '__main__':
    mix = False
    if mix:
        for i in range(3):
            generate_instances(100, "test", "CA", epoch=i)
    else:
        generate_instances(4, "test", "SC", epoch=1)
