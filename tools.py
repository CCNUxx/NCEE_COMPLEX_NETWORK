import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from pylab import mpl
import time

# Specify the default font: Solve the problem of not being able to display Chinese characters.
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

# Solve the problem of the negative sign '-' being displayed as a square when saving images.
mpl.rcParams["axes.unicode_minus"] = False


def get_res(tmp, res, idx_, idy_):
    for i in range(len(tmp)):
        ttmmpp = tmp.copy()
        del ttmmpp[i]
        tmp_i = np.where(idx_ == tmp[i][1])
        for j in range(len(ttmmpp)):
            tmp_j = np.where(idy_ == ttmmpp[j][1])
            res[tmp_i, tmp_j] = res[tmp_i, tmp_j] + 1


def dataRd(name):

    Data = pd.read_excel(name + ".xlsx")
    KnowledgePoints = Data.iloc[:, 0:2]
    KnowledgePoints = np.array(KnowledgePoints)

    jiedian_idx = np.where(KnowledgePoints[:, 1] == "节点")
    KnowledgePoints = np.delete(KnowledgePoints, jiedian_idx, axis=0)
    nan_idx = []
    for i in range(len(KnowledgePoints)):
        if type(KnowledgePoints[:, 1][i]) != str and np.isnan(KnowledgePoints[:, 1][i]):
            nan_idx.append(i)
    KnowledgePoints = np.delete(KnowledgePoints, nan_idx, axis=0)

    for i in range(len(KnowledgePoints) - 1):
        if (
            type(KnowledgePoints[i + 1, 0]) != int
            and type(KnowledgePoints[i + 1, 0]) != str
        ):
            KnowledgePoints[i + 1, 0] = KnowledgePoints[i, 0]
        if type(KnowledgePoints[i, 0]) == str:
            tmp = KnowledgePoints[i, 0][0:2]
            tmp = int(tmp)
            KnowledgePoints[i, 0] = tmp

    idx = []
    idy = []
    idx.append(KnowledgePoints[0, 1])
    idy.append(KnowledgePoints[0, 1])
    for i in range(1, len(KnowledgePoints)):
        if KnowledgePoints[i, 1] not in idx:
            idx.append(KnowledgePoints[i, 1])
        if KnowledgePoints[i, 1] not in idy:
            idy.append(KnowledgePoints[i, 1])

    res_ = []
    i = 0
    while i < len(KnowledgePoints):
        tmp = []
        tmp.append(KnowledgePoints[i])
        if i >= len(KnowledgePoints) - 1:
            res_.append(tmp)
            break
        j = 1
        while KnowledgePoints[i, 0] == KnowledgePoints[i + j, 0]:
            tmp.append(KnowledgePoints[i + j])
            j = j + 1
            if i + j > len(KnowledgePoints) - 1:
                break
        res_.append(tmp)
        i = i + j

    res = np.zeros((len(idx), len(idy)))
    idx_ = np.array(idx)
    idy_ = np.array(idy)

    for p in res_:
        get_res(p, res, idx_, idy_)

    result = pd.DataFrame(res, columns=idy, index=idx)

    return res, idx, idy, result


def GenGraph(data_dir):
    """Network Graph Generation

    Args:
        data_dir: Data Path.
    """

    from random import randint

    name_list = os.listdir(data_dir)

    for i in name_list:

        print(i.split(".xlsx")[0])

        name = data_dir + "/" + i.split(".xlsx")[0]
        res, _, idy, _ = dataRd(name)

        node_label = list(range(0, len(idy)))
        labels = dict(zip(node_label, idy))

        G = nx.from_numpy_array(res)

        # Rearrange the Graph
        pos = nx.spring_layout(G, iterations=15, seed=2500)

        label_propagation_communities = nx.community.label_propagation_communities(G)

        colors = ["" for x in range(G.number_of_nodes())]  # initialize colors list

        counter = 0  # Number of community

        for com in label_propagation_communities:
            color = "#%06X" % randint(0, 0xFFFFFF)  # Creates random RGB color
            counter += 1
            for node in list(
                com
            ):  # Fill colors list with the particular color for the community nodes
                colors[node] = color

        fig, ax = plt.subplots(figsize=(15, 9))
        nx.draw(
            G, node_size=300, labels=labels, width=2, node_color=colors, pos=pos, ax=ax
        )
        fig.savefig("./figures/{}.png".format(name.split("/")[-1]))
        plt.close()


def GenKeyNodes(data_dir):
    """Key Node Identification

    Args:
        data_dir: Data Path.
    """

    name_list = os.listdir(data_dir)

    for i in name_list:

        print(i.split(".xlsx")[0])

        name = data_dir + "/" + i.split(".xlsx")[0]

        res, _, idy, _ = dataRd(name)

        G = nx.from_numpy_array(res)

        # Extract the top 15 nodes with the highest eigenvector centrality
        data = dict(nx.eigenvector_centrality(G, max_iter=2000, weight="weight"))

        df = pd.DataFrame(
            {"node": list(data.keys()), "degree": list(data.values()), "label": idy}
        )

        top_30_EC_nodes = df.sort_values(
            by=["degree", "node"], ascending=[False, True]
        ).head(30)

        top_30_EC_nodes.to_csv(
            "./top_30_EC_nodes/top_30_EC_nodes_{}.csv".format(name.split("/")[-1]),
            index=False,
            encoding="utf-8-sig",
        )


def Community(data_dir):

    name_list = os.listdir(data_dir)

    commu_L_total = []
    lp_total = []
    counter_total = []
    bl_tatal = []

    for i in name_list:

        print(i.split(".xlsx")[0])

        name = data_dir + "/" + i.split(".xlsx")[0]

        res, _, _, _ = dataRd(name)

        G = nx.from_numpy_array(res)

        # Modularity
        commu_L = nx.community.modularity(
            G, nx.community.label_propagation_communities(G), weight="weight"
        )
        print("L-Modularity:", commu_L)

        # Use semi-synchronous label propagation method to detect communities.
        label_propagation_communities = nx.community.label_propagation_communities(G)

        a = list(label_propagation_communities)
        b = sorted(a, key=len)
        lp = []
        for j in range(len(b)):
            lp.append(len(sorted(b[j])))
        print("Community structure:", lp)

        bl = sum(1 for x in lp if x >= 3)
        print("Number of blocks:", bl)

        # Number of community
        counter = len(label_propagation_communities)
        print("Number of community:", counter)

        commu_L_total.append(commu_L)
        lp_total.append(lp)
        counter_total.append(counter)
        bl_tatal.append(bl)

    return commu_L_total, lp_total, counter_total, bl_tatal


def M_cal(a):
    # Monotonicity calculation

    N = len(a)
    b = np.array(a)
    c = b[:, 1]
    count = []
    d = [c[0]]
    for i in c:
        if i not in d:
            d.append(i)
    for j in range(len(d)):
        tmp = 0
        for ii in range(len(c)):
            if d[j] == c[ii]:
                tmp += 1
        count.append(tmp)

    tmp = 0
    for i in range(len(count)):
        tmp += count[i] * (count[i] - 1)
    tmp = tmp / (N * (N - 1))
    M = np.power(1 - tmp, 2)
    return M


def MonCentrality(data_dir):

    name_list = os.listdir(data_dir)

    DC_total = []
    EC_total = []
    CC_total = []
    BC_total = []

    for i in name_list:

        print(i.split(".xlsx")[0])

        name = data_dir + "/" + i.split(".xlsx")[0]
        res, _, _, _ = dataRd(name)

        G = nx.from_numpy_array(res)

        DC = M_cal(
            sorted(nx.degree_centrality(G).items(), key=lambda x: x[1], reverse=True)
        )
        EC = M_cal(
            sorted(
                nx.eigenvector_centrality(G, max_iter=2000, weight="weight").items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        CC = M_cal(
            sorted(nx.closeness_centrality(G).items(), key=lambda x: x[1], reverse=True)
        )
        BC = M_cal(
            sorted(
                nx.betweenness_centrality(G, weight="weight").items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        print("DC-M:", DC)
        print("EC-M:", EC)
        print("CC-M:", CC)
        print("BC-M:", BC)

        DC_total.append(DC)
        EC_total.append(EC)
        CC_total.append(CC)
        BC_total.append(BC)

    return DC_total, EC_total, CC_total, BC_total


def TopoStructure(data_dir):

    name_list = os.listdir(data_dir)

    Nodes_total = []
    Edges_total = []
    Diameter_total = []
    Density_total = []
    Assortativity_total = []
    Transitivity_total = []
    AvgDegree_total = []
    AvgShortPathLen_total = []
    AvgCluCoeffi_total = []
    Fd_total = []

    for i in name_list:

        print(i.split(".xlsx")[0])

        name = data_dir + "/" + i.split(".xlsx")[0]
        res, _, _, _ = dataRd(name)

        G = nx.from_numpy_array(res)

        n = G.number_of_nodes()
        m = G.number_of_edges()

        print("Number of Nodes:", n)

        print("Number of Edges:", m)

        Avg_degree = 2 * m / n
        print("Average degree:", Avg_degree)

        Density = nx.density(G)
        print("Density:", Density)

        Transitivity = nx.transitivity(G)
        print("Transitivity:", Transitivity)

        ACC = nx.average_clustering(G, weight="weight", count_zeros=True)
        print("Average clustering coefficient:", ACC)

        Fd = Avg_degree * Density * Transitivity * ACC * 10
        print("Comprehensive difficulty coefficient:", Fd)

        Assortativity = nx.degree_assortativity_coefficient(G, weight="weight")
        print("Assortativity:", Assortativity)

        # Connected subgraphs in G
        components = nx.connected_components(G)

        # Max connected subgraph
        max_component = max(components, key=len)

        mc = list(max_component)
        i, j = np.ix_(mc, mc)
        d_res = res[i, j]

        G0 = nx.from_numpy_array(d_res)

        Diameter = nx.diameter(G0)
        print("Diameter:", Diameter)

        ASPL = nx.average_shortest_path_length(G0, weight="weight")
        print("Average shortest path length:", ASPL)

        Nodes_total.append(n)
        Edges_total.append(m)
        Diameter_total.append(Diameter)
        Density_total.append(Density)
        Assortativity_total.append(Assortativity)
        Transitivity_total.append(Transitivity)
        AvgDegree_total.append(Avg_degree)
        AvgShortPathLen_total.append(ASPL)
        AvgCluCoeffi_total.append(ACC)
        Fd_total.append(Fd)

    return (
        Nodes_total,
        Edges_total,
        Diameter_total,
        Density_total,
        Assortativity_total,
        Transitivity_total,
        AvgDegree_total,
        AvgShortPathLen_total,
        AvgCluCoeffi_total,
        Fd_total,
    )


def DegreeDistr(data_dir):
    """Degree Distribution Calculation

    Args:
        data_dir: Data Path
    """

    name_list = os.listdir(data_dir)

    file_path = "./results/DegreeDistr.xlsx"

    if os.path.exists(file_path):
        os.remove(file_path)

    with pd.ExcelWriter("./results/DegreeDistr.xlsx") as writer:

        for ii in name_list:

            print(ii.split(".xlsx")[0])

            name = data_dir + "/" + ii.split(".xlsx")[0]
            res, _, _, _ = dataRd(name)

            G = nx.from_numpy_array(res)

            n = len(G.nodes)
            d = dict(nx.degree(G, weight="weight"))

            dd = list(d.values())
            tmp_value = []
            tmp_count = []
            for i in dd:
                if i not in tmp_value:
                    tmp_value.append(i)

            tmp_value.sort()

            for j in tmp_value:
                count = 0
                for jj in dd:
                    if j == jj:
                        count += 1
                tmp_count.append(count)

            tmp_count_frq = np.array(tmp_count) / n

            x = np.array(tmp_value)
            y = tmp_count_frq

            result = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))

            result = pd.DataFrame(result, columns=["x", "y"])

            result.to_excel(writer, sheet_name=ii.split(".xlsx")[0], index=None)


def GenCommunity(data_dir):
    """Network Community Structure Detection

    Args:
        data_dir: Data Path
    """

    commu_L_total, lp_total, counter_total, bl_total = Community(data_dir)

    name_list = os.listdir(data_dir)

    for k in range(len(name_list)):
        name_list[k] = name_list[k].replace(".xlsx", "")

    commu_L_total = np.array(commu_L_total)
    counter_total = np.array(counter_total)
    bl_total = np.array(bl_total)

    for l in range(len(lp_total)):
        lp_total[l] = str(lp_total[l]).replace("[", "").replace("]", "")
    lp_total = np.array(lp_total)

    result = np.vstack((commu_L_total, counter_total, lp_total, bl_total))

    result = pd.DataFrame(
        result,
        columns=name_list,
        index=[
            "L-Modularity",
            "Number of community",
            "Community structure",
            "Number of blocks",
        ],
    )

    result.to_excel("./results/Community_results.xlsx")


def GenMonCentrality(data_dir):
    """Calculation of Monotonicity for Centrality Measures

    Args:
        data_dir: Data Path
    """

    DC_total, EC_total, CC_total, BC_total = MonCentrality(data_dir)

    name_list = os.listdir(data_dir)

    DC_total = np.array(DC_total)
    EC_total = np.array(EC_total)
    CC_total = np.array(CC_total)
    BC_total = np.array(BC_total)

    for k in range(len(name_list)):
        name_list[k] = name_list[k].replace(".xlsx", "")

    result = np.vstack((DC_total, EC_total, CC_total, BC_total))

    result = pd.DataFrame(
        result,
        columns=name_list,
        index=["DC", "EC", "CC", "BC"],
    )

    result.to_excel("./results/MonCentrality_results.xlsx")


def GenTopoStructure(data_dir):
    """Calculation of Basic Topological Parameters of Networks

    Args:
        data_dir: Data Path
    """
    (
        Nodes_total,
        Edges_total,
        Diameter_total,
        Density_total,
        Assortativity_total,
        Transitivity_total,
        AvgDegree_total,
        AvgShortPathLen_total,
        AvgCluCoeffi_total,
        Fd_total,
    ) = TopoStructure(data_dir)

    name_list = os.listdir(data_dir)

    Nodes_total = np.array(Nodes_total)
    Edges_total = np.array(Edges_total)
    Diameter_total = np.array(Diameter_total)
    Density_total = np.array(Density_total)
    Assortativity_total = np.array(Assortativity_total)
    Transitivity_total = np.array(Transitivity_total)
    AvgDegree_total = np.array(AvgDegree_total)
    AvgShortPathLen_total = np.array(AvgShortPathLen_total)
    AvgCluCoeffi_total = np.array(AvgCluCoeffi_total)
    Fd_total = np.array(Fd_total)

    for k in range(len(name_list)):
        name_list[k] = name_list[k].replace(".xlsx", "")

    result = np.vstack(
        (
            Nodes_total,
            Edges_total,
            Diameter_total,
            Density_total,
            Assortativity_total,
            Transitivity_total,
            AvgDegree_total,
            AvgShortPathLen_total,
            AvgCluCoeffi_total,
            Fd_total,
        )
    )

    result = pd.DataFrame(
        result,
        columns=name_list,
        index=[
            "Number of Nodes",
            "Number of Edges",
            "Diameter",
            "Density",
            "Assortativity",
            "Transitivity",
            "Average degree",
            "Average shortest path length",
            "Average clustering coefficient",
            "Comprehensive difficulty coefficient",
        ],
    )

    result.to_excel("./results/Topological_Structure_of_KPNs.xlsx")


def CalER(name, network_seed, numpy_seed):

    res, _, _, _ = dataRd(name)

    G = nx.from_numpy_array(res)
    components = nx.connected_components(G)
    max_component = max(components, key=len)

    mc = list(max_component)
    i, j = np.ix_(mc, mc)
    d_res = res[i, j]

    # Maximum connected subgraph
    G0 = nx.from_numpy_array(d_res)

    print("G0_Number of nodes:", G0.number_of_nodes())
    print("G0_Number of edges:", G0.number_of_edges())
    print("G0_Diameter:", nx.diameter(G0))
    print(
        "G0_Average shortest path length:",
        nx.average_shortest_path_length(G0, weight="weight"),
    )
    print(
        "G0_Average clustering coefficient:",
        nx.average_clustering(G0, weight="weight", count_zeros=False),
    )

    # Count the proportion of edges in the network with weight greater than 1
    total_edges = G0.number_of_edges()
    weighted_edges = [(u, v) for u, v, d in G0.edges(data=True) if d["weight"] > 1]
    num_weighted_edges = len(weighted_edges)
    weighted_edge_probability = (
        num_weighted_edges / total_edges if total_edges > 0 else 0.0
    )

    tic = time.time()

    # Constructing ER network
    n = G0.number_of_nodes()
    m = G0.number_of_edges()

    network_seed_list = list(range(1, int(network_seed + 1)))
    numpy_seed_list = list(range(1, int(numpy_seed + 1)))

    diameters = []
    avg_shortest_paths = []
    avg_clusterings = []
    weighted_edge_probability = num_weighted_edges / total_edges

    for i in range(len(numpy_seed_list)):

        for j in range(len(network_seed_list)):

            ER = nx.erdos_renyi_graph(n, 2.0 * m / (n * (n - 1)), seed=j)

            np.random.seed(i)
            for u, v in ER.edges():
                ER[u][v]["weight"] = (
                    2 if np.random.rand() < weighted_edge_probability else 1
                )

            if nx.is_connected(ER):
                diameter = nx.diameter(ER)
                avg_shortest_path = nx.average_shortest_path_length(ER, weight="weight")

                clustering_coeffs = nx.clustering(ER, weight="weight").values()
                non_zero_clustering_coeffs = [
                    coeff for coeff in clustering_coeffs if coeff > 0
                ]

                if non_zero_clustering_coeffs:
                    avg_clustering = sum(non_zero_clustering_coeffs) / len(
                        non_zero_clustering_coeffs
                    )
                else:
                    avg_clustering = 0

                diameters.append(diameter)
                avg_shortest_paths.append(avg_shortest_path)
                avg_clusterings.append(avg_clustering)

    if diameters:
        avg_diameter = np.mean(diameters)
        std_diameter = np.std(diameters)
        avg_avg_shortest_path = np.mean(avg_shortest_paths)
        std_avg_shortest_path = np.std(avg_shortest_paths)
        avg_avg_clustering = np.mean(avg_clusterings)
        std_avg_clustering = np.std(avg_clusterings)

        print(f"ER_Diameter: {avg_diameter} ± {std_diameter}")
        print(
            f"ER_Average shortest path length: {avg_avg_shortest_path} ± {std_avg_shortest_path}"
        )
        print(
            f"ER_Average clustering coefficient: {avg_avg_clustering} ± {std_avg_clustering}"
        )

        # small-coefficient σ
        C_g = (
            nx.average_clustering(G0, weight="weight", count_zeros=False)
            / avg_avg_clustering
        )
        L_g = (
            nx.average_shortest_path_length(G0, weight="weight") / avg_avg_shortest_path
        )
        small_coffi = C_g / L_g

        print(f"C_g: {C_g}")
        print(f"L_g: {L_g}")

        if C_g > 1 and L_g >= 1 and small_coffi > 1:
            print(
                f"There is a small-world effect, and the small-world coefficient is: {small_coffi}"
            )
        else:
            print(
                f"No small world effect, and the coefficient result is: {small_coffi}"
            )
    else:
        print("No connected ER graph generated, please check the parameters.")

    print("time used %g sec(s)." % (time.time() - tic))

    return (
        C_g,
        L_g,
        small_coffi,
        avg_diameter,
        avg_avg_shortest_path,
        avg_avg_clustering,
    )


def GenER(data_dir, network_seed=1e2, numpy_seed=1e2):
    """Generate ER Random Model

    Args:
        data_dir: Data Path
        network_seed: Random Graph Seed. Defaults to 1e2.
        numpy_seed: Edges Random Weights Seed. Defaults to 1e2.
    """

    name_list = os.listdir(data_dir)

    obj = np.arange(2006, 2021, 1).astype(str)

    name_list_for_ER = []
    for k in range(len(name_list)):
        tmp = name_list[k].split("-")[0]
        if tmp in obj:
            name_list_for_ER.append(name_list[k].replace(".xlsx", ""))

    res_C_g = []
    res_L_g = []
    res_small_coffi = []
    res_avg_diameter = []
    res_avg_avg_shortest_path = []
    res_avg_avg_clustering = []

    for i in name_list_for_ER:

        name_tmp = data_dir + "/" + i
        (
            C_g,
            L_g,
            small_coffi,
            avg_diameter,
            avg_avg_shortest_path,
            avg_avg_clustering,
        ) = CalER(name_tmp, network_seed, numpy_seed)

        res_C_g.append(C_g)
        res_L_g.append(L_g)
        res_small_coffi.append(small_coffi)
        res_avg_diameter.append(avg_diameter)
        res_avg_avg_shortest_path.append(avg_avg_shortest_path)
        res_avg_avg_clustering.append(avg_avg_clustering)

    res_C_g = np.array(res_C_g)
    res_L_g = np.array(res_L_g)
    res_small_coffi = np.array(res_small_coffi)
    res_avg_diameter = np.array(res_avg_diameter)
    res_avg_avg_shortest_path = np.array(res_avg_avg_shortest_path)
    res_avg_avg_clustering = np.array(res_avg_avg_clustering)

    res = np.vstack(
        [
            res_avg_diameter,
            res_avg_avg_shortest_path,
            res_avg_avg_clustering,
            res_C_g,
            res_L_g,
            res_small_coffi,
        ]
    )

    result = pd.DataFrame(res, index=None, columns=name_list_for_ER)

    result.to_excel(
        "./results/ER_result_{}.xlsx".format(int(network_seed * numpy_seed)), index=None
    )
