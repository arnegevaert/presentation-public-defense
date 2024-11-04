from itertools import combinations

from manim import *
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def paragraph(*strs, alignment=LEFT, direction=DOWN, **kwargs):
    texts = VGroup(*[Text(s, **kwargs) for s in strs]).arrange(direction)

    if len(strs) > 1:
        for text in texts[1:]:
            text.align_to(texts[0], direction=alignment)

    return texts


def example_function_1():
    def fun(x):
        return 0.5 * x - 1

    return fun, "f({}) = 0.5 * {} - 1"


def linreg_univariate(x_points, y_points):
    lm = linear_model.LinearRegression()
    lm.fit(x_points.reshape(-1, 1), y_points)
    return lm


def linreg_multivariate(x_points, y_points):
    lm = linear_model.LinearRegression()
    lm.fit(x_points, y_points)
    return lm


def parabolic_reg(x_points, y_points):
    feat = PolynomialFeatures(degree=2)
    lm = Pipeline(
        [
            ("preproc", PolynomialFeatures(degree=2)),
            ("model", linear_model.LinearRegression()),
        ]
    )
    lm.fit(x_points.reshape(-1, 1), y_points)
    return lm


def nn_reg(x_points, y_points):
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "predict",
                MLPRegressor(
                    alpha=1e-4, tol=1e-6, max_iter=int(1e6), hidden_layer_sizes=(25,)
                ),
            ),
        ]
    )
    model.fit(x_points.reshape(-1, 1), y_points)
    return model


def bin_to_int(b):
    return int(b, 2)


def int_to_bin(n, length):
    b = bin(n)[2:]
    padding = length - len(b)
    return "0" * padding + b


def make_hasse(vertex_config=None, edge_config=None, labels=None):
    edges = []
    partitions = []
    n_elements = 3

    partitions = []
    vertices = []
    edges = []

    for i in range(n_elements + 1):
        # Generate vertices
        partition = []
        combs = combinations(range(n_elements), i)
        for comb in combs:
            bin = "0" * n_elements
            for element in comb:
                bin = list(bin)
                bin[element] = "1"
                bin = "".join(bin)
            partition.append(bin_to_int(bin))
            vertices.append(bin_to_int(bin))
        partitions.append(sorted(partition))

        # Generate edges
        if i < n_elements:
            for node in partition:
                binary_rep = int_to_bin(node, n_elements)

                for j in range(len(binary_rep)):
                    char = binary_rep[j]
                    if char == "0":
                        other_node = list(binary_rep)
                        other_node[j] = "1"
                        other_node = bin_to_int("".join(other_node))
                        edges.append((node, other_node))

    graph = Graph(
        vertices[::-1],
        edges=edges,
        layout="partite",
        layout_config={"align": "horizontal"},
        partitions=partitions,
        layout_scale=(3, 3),
        labels=labels,  # Set to True to debug
        vertex_config=vertex_config,
        edge_config=edge_config,
    )
    return graph

