from manim import *
from sklearn import linear_model


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