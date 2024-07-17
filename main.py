import tools
import os


figures_path = "./figures"
if not os.path.exists(figures_path):
    os.makedirs(figures_path)

top_30_EC_nodes_path = "./top_30_EC_nodes"
if not os.path.exists(top_30_EC_nodes_path):
    os.makedirs(top_30_EC_nodes_path)

results_path = "./results"
if not os.path.exists(results_path):
    os.makedirs(results_path)

data_dir = "./data"

# Generate Graphs
tools.GenGraph(data_dir)

# Generate Key Nodes
tools.GenKeyNodes(data_dir)

# Detect Community
tools.GenCommunity(data_dir)

# Generate Monotonicity
tools.GenMonCentrality(data_dir)

# Generate Topological Structure of KPNs
tools.GenTopoStructure(data_dir)

# Degree Distribution
tools.DegreeDistr(data_dir)

# ER Random Model.
tools.GenER(data_dir, network_seed=1e1, numpy_seed=1e1)

print("end")
