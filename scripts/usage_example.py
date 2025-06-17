from q_analysis.examples.scale_free_configurational import generate_networks
import numpy as np
import os
from itertools import product
from q_analysis.simplicial_complex import SimplicialComplex
import pandas as pd
from matplotlib import pyplot as plt
from q_analysis.viz import plot_q_analysis_vectors
from scipy import stats
from q_analysis.stat import consensus_statistic, calculate_consensus_adjacency_matrix
from q_analysis.transformers import GradedParametersTransformer
from q_analysis.simplicial_complex import GradedParameters
import seaborn as sns

os.makedirs("plots", exist_ok=True)

N_SAMPLES, N_NODES, M_PARAMETER = 100, 100, 8
scale_free_networks, configurational_networks = generate_networks(
    N_NODES,
    M_PARAMETER,
    N_SAMPLES,
)
networks = np.concatenate([scale_free_networks, configurational_networks])


index = product(["Scale free", "Configurational"], range(N_SAMPLES))
simplicial_complex_metrics = [
    SimplicialComplex.from_adjacency_matrix(network)
    .graded_parameters()
    .to_dataframe()
    .assign(Network=net_type, Sample=sample_id)
    for network, (net_type, sample_id) in zip(networks, index)
]

structure_vectors_df = pd.concat(simplicial_complex_metrics, ignore_index=True)
plot_q_analysis_vectors(
    structure_vectors_df, hue="Network", height=3, col_wrap=2, legend_out=False
)
plt.savefig("plots/scale_free_configurational.pdf")
plt.close()

consensus_scale_free_vectors, consensus_configurational_vectors = (
    GradedParametersTransformer().fit_transform(
        [
            calculate_consensus_adjacency_matrix(scale_free_networks),
            calculate_consensus_adjacency_matrix(configurational_networks),
        ],
    )
)
max_order = len(consensus_scale_free_vectors)
stats_res = stats.permutation_test(
    [scale_free_networks, configurational_networks],
    statistic=lambda a, b, axis: consensus_statistic(
        a, b, max_order=max_order, edge_inclusion_threshold=0.95
    ),
    n_resamples=10000,
    vectorized=True,
    batch=100,
    axis=1,
)
p_value_df = GradedParameters.from_numpy(stats_res.pvalue).to_dataframe()
consensus_vectors_df = pd.concat(
    [
        GradedParameters.from_numpy(consensus_vector)
        .to_dataframe()
        .assign(Network=network)
        for network, consensus_vector in zip(
            ["Scale free", "Configurational"],
            [consensus_scale_free_vectors, consensus_configurational_vectors],
        )
    ],
    ignore_index=True,
)
plot_q_analysis_vectors(
    consensus_vectors_df,
    pvalues_df=p_value_df,
    hue="Network",
    height=3,
    col_wrap=2,
    legend_out=False,
)
plt.savefig("plots/scale_free_configurational_consensus_vectors.pdf")
plt.close()

simplicial_complexes_topological_dimensionality = [
    SimplicialComplex.from_adjacency_matrix(network)
    .topological_dimensionality()
    .to_dataframe()
    .assign(Network=net_type)
    .set_index(["Network", "Node"])
    for network, net_type in zip(
        [scale_free_networks[0], configurational_networks[0]],
        ["Scale free", "Configurational"],
    )
]
topological_dimensionality_df = pd.concat(
    simplicial_complexes_topological_dimensionality,
)
ax = sns.barplot(
    data=topological_dimensionality_df,
    x="Node",
    y="Topological Dimensionality",
    hue="Network",
)
plt.xticks(plt.xticks()[0][::10])
plt.savefig("plots/scale_free_configurational_topological_dimensionality.pdf")
