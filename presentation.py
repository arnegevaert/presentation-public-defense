from manim import *
from manim_slides import ThreeDSlide
from sklearn.metrics import root_mean_squared_error

from util import (
    example_function_1,
    linreg_multivariate,
    linreg_univariate,
    make_hasse,
    nn_reg,
    parabolic_reg,
    paragraph,
)

config.background_color = "#262626ff"


class PixelsAsSquares(VGroup):
    def __init__(self, image_mobject, **kwargs):
        VGroup.__init__(self, **kwargs)
        for row in image_mobject.pixel_array:
            for val in row:
                square = Square(
                    stroke_width=1,
                    stroke_color=WHITE,
                    stroke_opacity=0.5,
                    fill_opacity=val[0] / 255.0,
                    fill_color=WHITE,
                )
                self.add(square)
        self.arrange_in_grid(*image_mobject.pixel_array.shape[:2], buff=0)
        self.replace(image_mobject)


class PixelsAsCircles(VGroup):
    def __init__(self, v_img, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.neurons = []
        for pixel in v_img:
            neuron = Circle(
                radius=pixel.width / 2,
                stroke_color=WHITE,
                stroke_width=1,
                fill_color=WHITE,
                fill_opacity=pixel.fill_opacity,
            )
            neuron.rotate(3 * np.pi / 4)
            neuron.move_to(pixel)
            self.add(neuron)
            self.neurons.append(neuron)
        self.space_out_submobjects(1.2)

    def get_values(self):
        nums = []
        for neuron in self.neurons:
            o = neuron.fill_opacity
            num = DecimalNumber(o, num_decimal_places=1)
            num.set(width=0.7 * neuron.width)
            num.move_to(neuron)
            if o > 0.8:
                num.set_fill(BLACK)
            nums.append(num)
        return nums


class Presentation(ThreeDSlide):
    def cleanup_slide(self):
        self.next_slide()
        self.play(*[FadeOut(obj) for obj in self.mobjects_without_canvas])

    def construct_toc_slide(self, chapter):
        contents_title = Text("Contents", color=BLACK, font_size=32).to_corner(UL)
        contents = (
            paragraph(
                f"1. Introduction",
                f"2. Feature Attribution Benchmark",
                f"3. Removal-Based Attribution Methods",
                f"4. Functional Decomposition",
                f"5. PDD-SHAP",
                f"6. Conclusion",
                color=WHITE,
                font_size=24,
            )
            .align_to(contents_title, LEFT)
            .shift(RIGHT)
        )

        self.play(FadeIn(contents))
        self.next_slide()
        cur = contents[chapter - 1]

        cur.generate_target()
        cur.target.set_fill(color=BLUE)
        cur.target.scale(1.2)

        self.play(
            FadeOut(contents[: chapter - 1]),
            FadeOut(contents[chapter:]),
            MoveToTarget(cur),
        )
        self.cleanup_slide()

    def construct_titleslide(self):
        title1 = Text(
            "Unifying Attribution-Based Explanation Methods",
            color=WHITE,
            font_size=40,
        )
        title2 = Text(
            "in Machine Learning",
            color=WHITE,
            font_size=40,
        ).next_to(title1, DOWN)
        author_date = Text(
            "Arne Gevaert - November 7th 2024", color=WHITE, font_size=24
        ).next_to(title2, DOWN)

        self.next_slide()
        self.play(FadeIn(title1), FadeIn(title2))
        self.play(FadeIn(author_date))
        self.cleanup_slide()

    def construct_chapter1_1(self):
        # GENERAL FUNCTION
        rect = Rectangle(width=5, height=3)
        f = MathTex("f", font_size=100)
        self.next_slide()

        self.play(Create(rect), Write(f))

        self.next_slide()

        in_arrow = Arrow(start=LEFT, end=RIGHT).next_to(rect, direction=LEFT)
        out_arrow = Arrow(start=LEFT, end=RIGHT).next_to(rect, direction=RIGHT)
        x = MathTex("x", font_size=100).next_to(in_arrow, direction=LEFT)
        y = MathTex("y", font_size=100).next_to(out_arrow, direction=RIGHT)
        self.play(DrawBorderThenFill(in_arrow, run_time=1), Write(x, run_time=1))
        self.play(DrawBorderThenFill(out_arrow, run_time=1), Write(y, run_time=1))

        self.next_slide()

        # EXAMPLE FUNCTION
        fun, fmt_str = example_function_1()
        ex_f = MathTex(fmt_str.format("x", "x"))
        self.play(Transform(f, ex_f))

        self.next_slide()

        input_values = (6, 4)
        for value in input_values:
            in_tex = MathTex(str(value), font_size=100).next_to(
                in_arrow, direction=LEFT
            )
            out_val = fun(value)
            assert out_val == int(out_val), "Output of function is not an integer"
            out_tex = MathTex(str(int(out_val)), font_size=100).next_to(
                out_arrow, direction=RIGHT
            )
            ex_f_transformed = MathTex(fmt_str.format(str(value), str(value)))
            self.play(
                Transform(x, in_tex),
                Transform(y, out_tex),
                Transform(f, ex_f_transformed),
            )
            self.next_slide()

        self.cleanup_slide()

    def construct_chapter1_2(self):
        self.next_slide()
        # GRAPH
        axis_config = {"include_ticks": True, "include_numbers": True}
        plane = NumberPlane(x_axis_config=axis_config, y_axis_config=axis_config)
        self.play(Write(plane))

        self.next_slide()

        fun, fmt_str = example_function_1()

        red_dots = []
        for x in [6, 4]:
            dot = Dot(color=RED).move_to([x, 0, 0])
            dot.generate_target()
            out = fun(x)
            dot.target.shift(out * UP)
            red_dots.append(dot)

        self.play(*[DrawBorderThenFill(dot, run_time=0.5) for dot in red_dots])

        self.next_slide()
        self.play(*[MoveToTarget(dot) for dot in red_dots])

        self.next_slide()

        dots = []
        for x in range(-35, 35, 1):
            x = x / 10 * 2
            dot = Dot(color=WHITE).move_to([x, 0, 0]).scale(0.5)
            dot.generate_target()
            dot.target.shift(fun(x) * UP)
            dots.append(dot)

        self.next_slide()

        self.play(*[DrawBorderThenFill(dot, run_time=0.5) for dot in dots])

        self.next_slide()
        self.play(LaggedStart(*[MoveToTarget(dot) for dot in dots], lag_ratio=0.01))

        self.next_slide()
        graph = plane.plot(fun, color=WHITE)
        self.play(
            Create(graph),
            *[FadeOut(dot) for dot in dots],
            *[FadeOut(dot) for dot in red_dots],
        )
        self.cleanup_slide()

    def construct_chapter1_3(self):
        self.next_slide()
        # Look at equation parameters
        eq = MathTex(
            "f(x) = 0.5 * x - 1", font_size=100, substrings_to_isolate=["0.5", "1"]
        )
        self.play(Write(eq))
        self.next_slide()

        params = ["0.5", "1"]

        for param in params:
            p = eq.get_parts_by_tex(param)
            p.generate_target()
            p.target.set_color(YELLOW)
            self.play(Circumscribe(p), MoveToTarget(p))
            self.next_slide()

        # Show influence on line
        axis_config = {"include_ticks": True, "include_numbers": True}
        plane = NumberPlane(
            x_axis_config=axis_config,
            y_axis_config=axis_config,
            background_line_style={"stroke_width": 4, "stroke_opacity": 0.2},
        )
        eq.generate_target()
        eq.target.to_corner(UL).shift(2 * LEFT).scale(0.5)
        graph = plane.plot(lambda x: 0.5 * x - 1, color=WHITE)
        self.play(Write(plane), MoveToTarget(eq))
        self.play(Create(graph))

        self.next_slide()

        graph2 = plane.plot(lambda x: x - 1, color=WHITE)
        eq2 = (
            MathTex("f(x) = 1 * x - 1", font_size=100, substrings_to_isolate=["1"])
            .to_corner(UL)
            .shift(2 * LEFT)
            .scale(0.5)
        )
        eq2.set_color_by_tex("1", YELLOW)
        self.play(Transform(graph, graph2), Transform(eq, eq2))

        self.next_slide()

        graph3 = plane.plot(lambda x: 0.5 * x - 1, color=WHITE)
        eq3 = (
            MathTex(
                "f(x) = 0.5 * x - 1", font_size=100, substrings_to_isolate=["0.5", "1"]
            )
            .to_corner(UL)
            .shift(2 * LEFT)
            .scale(0.5)
        )
        eq3.set_color_by_tex("1", YELLOW)
        eq3.set_color_by_tex("0.5", YELLOW)
        self.play(Transform(graph, graph3), Transform(eq, eq3))

        self.next_slide()

        graph4 = plane.plot(lambda x: 0.5 * x - 2, color=WHITE)
        eq4 = (
            MathTex(
                "f(x) = 0.5 * x - 2", font_size=100, substrings_to_isolate=["0.5", "2"]
            )
            .to_corner(UL)
            .shift(2 * LEFT)
            .scale(0.5)
        )
        eq4.set_color_by_tex("2", YELLOW)
        eq4.set_color_by_tex("0.5", YELLOW)
        self.play(Transform(graph, graph4), Transform(eq, eq4))

        self.next_slide()

        graph5 = plane.plot(lambda x: 0.5 * x - 1, color=WHITE)
        eq5 = (
            MathTex(
                "f(x) = 0.5 * x - 1", font_size=100, substrings_to_isolate=["0.5", "1"]
            )
            .to_corner(UL)
            .shift(2 * LEFT)
            .scale(0.5)
        )
        eq5.set_color_by_tex("1", YELLOW)
        eq5.set_color_by_tex("0.5", YELLOW)
        self.play(Transform(graph, graph5), Transform(eq, eq5))

        self.cleanup_slide()

    def construct_chapter1_4(self):
        self.next_slide()
        # Draw axes
        axis_config = {
            "include_ticks": True,
            "include_numbers": True,
        }
        plane = NumberPlane(
            x_axis_config=axis_config,
            y_axis_config=axis_config,
            background_line_style={"stroke_width": 4, "stroke_opacity": 0.2},
            x_range=(-6, 22, 2),
            x_length=config.frame_width,
            y_range=(-1, 7),
            y_length=config.frame_height,
        )
        self.play(Write(plane))

        self.next_slide()

        # Generate data
        def true_fn(x):
            return 5 - x / 4

        x_points = np.arange(-3, 19, 2)
        rng = np.random.default_rng(seed=0)
        noise = rng.normal(0, 0.5, len(x_points))
        y_points = true_fn(x_points) + noise

        dots = []
        for x, y in zip(x_points, y_points):
            dot = Dot().move_to(plane.coords_to_point(x, y, 0))
            dots.append(dot)

        self.play(Create(dot) for dot in dots)

        self.next_slide()

        # Random line
        # TODO add a little equation on the top right of the screen
        # TODO add error function
        rand_line = plane.plot(lambda x: 0.1 * x + 0.5)
        self.play(Create(rand_line))

        self.next_slide()

        # TODO split animation: first fit slope, then intercept
        lm = linreg_univariate(x_points, y_points)
        reg_line = plane.plot(lambda x: lm.predict(np.array(x).reshape(-1, 1))[0])
        self.play(Transform(rand_line, reg_line))

        self.cleanup_slide()

    def construct_chapter1_5(self):
        self.next_slide()
        axes = ThreeDAxes()
        self.set_camera_orientation(theta=-45 * DEGREES, phi=75 * DEGREES)
        self.begin_ambient_camera_rotation(rate=75 * DEGREES / 4)

        rng = np.random.default_rng(seed=0)
        n_points = 10
        data = rng.uniform(low=-3, high=3, size=(n_points, 2))

        dots = [
            Dot3D(color=RED, point=axes.coords_to_point(row[0], row[1], 0))
            for row in data
        ]
        self.play(Write(axes), *[FadeIn(dot) for dot in dots])

        def outcome(x):
            return 0.5 * x[:, 0] - 0.25 * x[:, 1]

        rng = np.random.default_rng(seed=0)
        noise = rng.normal(0, 0.5, n_points)
        y = outcome(data) + noise

        self.play(Wait(run_time=3))

        for dot, y_val in zip(dots, y):
            dot.generate_target()
            dot.target.shift(axes.coords_to_point(0, 0, y_val))
        self.play(*[MoveToTarget(dot) for dot in dots])

        self.play(Wait(run_time=3))

        surface = Surface(
            lambda u, v: axes.c2p(u, v, 0.5 * u - 0.25 * v),
            u_range=[-7, 7],
            v_range=[-7, 7],
            resolution=8,
            fill_opacity=0.5,
        )
        self.play(DrawBorderThenFill(surface))

        self.play(Wait(run_time=3))

        self.cleanup_slide()

        self.set_camera_orientation(theta=-90 * DEGREES, phi=0)
        self.stop_ambient_camera_rotation()

    def construct_chapter1_6(self):
        rows = [Text(f"Day {i+1}") for i in range(6)]
        cols = [
            Tex(label)
            for label in [
                "$x_1$: Temp (C)",
                "$x_1$: People",
                "$x_1$: Sun (\%)",
                "$y$: Energy",
            ]
        ]
        data = [
            [14, 3, 70],
            [17, 4, 80],
            [13, 3, 75],
            [5, 2, 65],
            [8, 3, 70],
            [10, 4, 80],
        ]

        def outcome(x):
            rng = np.random.default_rng(seed=0)
            noise = rng.normal(0, 0.5, x.shape[0])
            return 2 * x[:, 0] + 3 * x[:, 1] - 0.25 * x[:, 2] + 5 + noise

        y = list(outcome(np.array(data)).reshape(-1))

        self.next_slide()

        table = (
            Table(
                [
                    [str(num) for num in entry] + [f"{out:.1f}"]
                    for entry, out in zip(data, y)
                ],
                row_labels=rows,
                col_labels=cols,
            )
            .scale(0.4)
            .set_column_colors(WHITE, WHITE, WHITE, WHITE, RED)
            .to_corner(DR)
        )
        self.play(table.create())

        self.next_slide()

        params = ["5.1", "1.3", "5.4", "2.8"]
        pred_before = [x[0] * 5.1 + x[1] * 1.3 - x[2] * 5.4 + 2.8 for x in data]
        mse_before = root_mean_squared_error(y, pred_before)

        eqn_before = MathTex(
            "f(x_1, x_2, x_3) = 5.1 * x_1 + 1.3 * x_2 - 5.4 * x_3 + 2.8",
            substrings_to_isolate=params,
        ).to_corner(UL)
        for p in params:
            eqn_before.set_color_by_tex(p, YELLOW)
        error_before = (
            MathTex(f"E = {mse_before:.2f}", color=RED).to_corner(UL).shift(DOWN)
        )

        self.play(Write(eqn_before))
        self.play(Write(error_before))

        self.next_slide()

        lm = linreg_multivariate(np.array(data), outcome(np.array(data)))
        params = [
            f"{lm.coef_[0]:.1f}",
            f"{lm.coef_[1]:.1f}",
            f"{abs(lm.coef_[2]):.1f}",
            f"{lm.intercept_:.1f}",
        ]
        eqn_after = MathTex(
            f"f(x_1, x_2, x_3) = {lm.coef_[0]:.1f} * x_1 + {lm.coef_[1]:.1f} * x_2 - {abs(lm.coef_[2]):.1f} * x_3 + {lm.intercept_:.1f}",
            substrings_to_isolate=params,
        ).to_corner(UL)
        for p in params:
            eqn_after.set_color_by_tex(p, YELLOW)

        mse_after = root_mean_squared_error(y, lm.predict(np.array(data)))
        mse_after = f"{mse_after:.2f}".zfill(6)
        error_after = MathTex(f"E = {mse_after}", color=RED).to_corner(UL).shift(DOWN)

        self.play(
            Transform(eqn_before, eqn_after), Transform(error_before, error_after)
        )

        self.cleanup_slide()

    def construct_chapter1_7(self):
        # ORIGINAL SETUP
        # Draw axes
        axis_config = {
            "include_ticks": True,
            "include_numbers": True,
        }
        plane = NumberPlane(
            x_axis_config=axis_config,
            y_axis_config=axis_config,
            background_line_style={"stroke_width": 4, "stroke_opacity": 0.2},
            x_range=(-6, 22, 2),
            x_length=config.frame_width,
            y_range=(-1, 7),
            y_length=config.frame_height,
        )

        self.next_slide()

        # Generate data
        def true_fn(x):
            return 5 - x / 4

        x_points = np.arange(-3, 19, 2)
        rng = np.random.default_rng(seed=0)
        noise = rng.normal(0, 0.5, len(x_points))
        y_points = true_fn(x_points) + noise

        dots = []
        for x, y in zip(x_points, y_points):
            dot = Dot().move_to(plane.coords_to_point(x, y, 0))
            dots.append(dot)

        lm = linreg_univariate(x_points, y_points)
        reg_line = plane.plot(lambda x: lm.predict(np.array(x).reshape(-1, 1))[0])

        self.play(Write(plane), *[Create(dot) for dot in dots], Create(reg_line))

        self.next_slide()

        # CHANGE TO PARABOLIC DATA
        def parabola(x):
            return x**2 / 15 - x + 4

        rng = np.random.default_rng(seed=0)
        noise = rng.normal(0, 0.5, len(x_points))
        y_parabola = parabola(x_points) + noise

        for i in range(len(dots)):
            dot = dots[i]
            dot.generate_target()
            dot.target.move_to(plane.coords_to_point(x_points[i], y_parabola[i], 0))

        self.play(MoveToTarget(dot) for dot in dots)

        self.next_slide()

        lm_para = parabolic_reg(x_points, y_parabola)
        para_line = plane.plot(lambda x: lm_para.predict(np.array(x).reshape(-1, 1))[0])
        self.play(Transform(reg_line, para_line))

        # TODO add equation showing added parameter

        # CHANGE TO MORE COMPLEX DATA

        self.next_slide()

        self.play(FadeOut(reg_line))

        def noise_data(x):
            rng = np.random.default_rng(seed=0)
            noise = rng.normal(0, 2, len(x))
            return noise + 3.5

        y_noise = noise_data(x_points)

        self.next_slide()

        for i in range(len(dots)):
            dot = dots[i]
            dot.generate_target()
            dot.target.move_to(plane.coords_to_point(x_points[i], y_noise[i], 0))
        self.play(MoveToTarget(dot) for dot in dots)

        self.next_slide()

        nn = nn_reg(x_points, y_noise)
        nn_line = plane.plot(lambda x: nn.predict(np.array(x).reshape(-1, 1))[0])
        self.play(Write(nn_line))

        self.cleanup_slide()

    def construct_chapter1_8(self):
        self.next_slide()

        x_points = np.arange(-3, 19, 2)

        def noise_data(x):
            rng = np.random.default_rng(seed=0)
            noise = rng.normal(0, 2, len(x))
            return noise + 3.5

        y_noise = noise_data(x_points)
        nn = nn_reg(x_points, y_noise)

        nn_string = f"f(x) &= {nn['predict'].intercepts_[1][0]:.2f}"
        l1_coefs = nn["predict"].coefs_[0].reshape(-1)
        l2_coefs = nn["predict"].coefs_[1].reshape(-1)
        l1_intercepts = nn["predict"].intercepts_[0].reshape(-1)

        for i in range(25):
            substring = "{:.4f} * \\max(0, {:.4f} * x {} {:.4f})".format(
                l2_coefs[i],
                l1_coefs[i],
                "+" if l1_intercepts[i] > 0 else "",
                l1_intercepts[i],
            )
            if i % 2 == 0:
                substring = substring + "\\\\ &"
            if l2_coefs[i] > 0:
                nn_string += "+"
            nn_string += substring

        formula = MathTex(nn_string).scale(1.5)
        self.play(Write(formula))
        self.next_slide()

        formula.generate_target()
        formula.target.scale(0.5)

        self.play(MoveToTarget(formula, run_time=5))

        self.cleanup_slide()

    def construct_chapter1_9(self):
        self.next_slide()
        title_text = Text("ChaptGPT (GPT-3.5):").shift(2 * UP)
        self.play(Write(title_text))

        num_text = Text("175", font_size=125)
        self.play(Write(num_text))

        param_text = Text("parameters").next_to(num_text, 2 * DOWN)
        self.play(Write(param_text))

        self.next_slide()

        num_text.generate_target()
        num_text.target.shift(3 * LEFT)
        self.play(MoveToTarget(num_text))
        billion = Text("billion", font_size=125, color=RED).next_to(num_text, 2 * RIGHT)
        self.play(Write(billion))

        self.next_slide()
        cur_text = Text("ChatGPT-4: 1.75 trillion (estimated)").next_to(
            param_text, DOWN
        )
        self.play(Write(cur_text))

        self.next_slide()
        self.play(*[FadeOut(obj) for obj in self.mobjects_without_canvas])

        map_img = ImageMobject("distance.png").scale(0.4)
        dist_text = Text("687.8 km").next_to(map_img, DOWN)
        self.play(FadeIn(map_img), Write(dist_text))

        self.next_slide()

        time_text = (
            Text("Writing at 1 parameter per second: 55,492 years")
            .scale(0.8)
            .next_to(dist_text, DOWN)
        )
        self.play(Write(time_text))

        self.cleanup_slide()

    def construct_chapter1_10(self):
        self.next_slide()
        rect = Rectangle(width=5, height=3)
        f = MathTex("f", font_size=100)
        self.play(Create(rect), Write(f))

        self.next_slide()

        in_arrows, in_vars = [], []
        for i, direction in enumerate((UL, LEFT, DL)):
            arr = Arrow(start=direction, end=RIGHT).next_to(rect, direction=direction)
            in_arrows.append(arr)
            in_vars.append(
                MathTex(f"x_{i+1}", font_size=100).next_to(arr, direction=direction)
            )

        out_arrow = Arrow(start=LEFT, end=RIGHT).next_to(rect, direction=RIGHT)

        y = MathTex("y", font_size=100).next_to(out_arrow, direction=RIGHT)

        self.play(
            *[DrawBorderThenFill(in_arrow, run_time=1) for in_arrow in in_arrows],
            *[Write(x, run_time=1) for x in in_vars],
        )
        self.play(DrawBorderThenFill(out_arrow, run_time=1), Write(y, run_time=1))

        self.next_slide()

        params = [(0, 1.1, 0.01), (1, 2, 0.06), (2, 1.3, 0.02)]

        for i, scale, rot in params:
            self.play(Circumscribe(in_vars[i]))
            self.play(
                Wiggle(in_vars[i]),
                Wiggle(y, scale_value=scale, rotation_angle=rot * TAU),
            )

            self.next_slide()

        self.cleanup_slide()

        bar = BarChart(
            values=[0.1, 0.5, 0.2], bar_names=[f"$x_{i+1}$" for i in range(3)]
        ).scale(1.5)
        self.play(Write(bar), run_time=2)

        self.cleanup_slide()

    def construct_chapter2_1(self):
        self.next_slide()
        X = np.loadtxt("digits.csv", delimiter=",")
        y = np.loadtxt("labels.csv", delimiter=",")

        def get_img(idx):
            img = ImageMobject(X[idx, :].reshape(28, 28)).scale(15)
            img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            label = Text(f"Label: {int(y[idx])}")
            return img, label

        imgs, labels = [], []
        for i in range(3):
            img, label = get_img(i)
            imgs.append(img.shift((-4 + i * 4) * RIGHT))
            labels.append(label.next_to(img, DOWN))

        self.play(FadeIn(img) for img in imgs)
        self.play(LaggedStart(*[Write(label) for label in labels], lag_ratio=0.25))

        self.next_slide()

        self.play(
            *[FadeOut(img) for img in (imgs[0], imgs[2])],
            *[FadeOut(label) for label in labels],
        )

        img = imgs[1]
        img.generate_target()
        img.target.center()
        img.target.scale(1.8)
        self.play(MoveToTarget(img))

        self.next_slide()

        v_img = PixelsAsSquares(img)
        self.play(FadeOut(img), FadeIn(v_img))

        circles = PixelsAsCircles(v_img)
        values = circles.get_values()
        self.play(ReplacementTransform(v_img, circles))
        self.play(*[FadeIn(value) for value in values])

        self.next_slide()

        self.play(*[FadeOut(value) for value in values])

        rows = VGroup(
            *[VGroup(*circles.neurons[28 * i : 28 * (i + 1)]) for i in range(28)]
        )

        self.play(rows.animate.space_out_submobjects(1.2))
        self.play(rows.animate.arrange(RIGHT, buff=SMALL_BUFF), run_time=2)

        summarized = (
            VGroup(
                *[Circle(stroke_color=WHITE, radius=0.25) for _ in range(7)],
                MathTex(r"\dots", font_size=100),
                *[Circle(stroke_color=WHITE, radius=0.25) for _ in range(3)],
            )
            .arrange(RIGHT)
            .scale(20)
        )
        summarized.generate_target()
        summarized.target.scale(0.05)
        self.play(
            LaggedStart(
                ShrinkToCenter(rows),
                AnimationGroup(FadeIn(summarized), MoveToTarget(summarized)),
                lag_ratio=0.5,
            )
        )

        self.next_slide()

        summarized.generate_target()
        summarized.target.rotate(-90 * DEGREES)
        summarized.target.scale(0.7)
        summarized.target.to_edge(LEFT).shift(RIGHT)

        self.play(MoveToTarget(summarized))

        rect = Rectangle(width=5, height=3)
        f = MathTex("f", font_size=100)
        self.play(Create(rect), Write(f))

        arrows = []
        for obj in summarized:
            if isinstance(obj, Circle):
                arr = Line(
                    start=obj.get_edge_center(RIGHT) + 0.2 * RIGHT,
                    end=rect.get_edge_center(LEFT),
                )
                arrows.append(arr)
        self.play(LaggedStart(*[Write(arr) for arr in arrows]))

        out_arrow = Arrow(start=LEFT, end=RIGHT).next_to(rect, direction=RIGHT)
        y = MathTex("y", font_size=100).next_to(out_arrow, direction=RIGHT)
        self.play(
            LaggedStart(
                DrawBorderThenFill(out_arrow, run_time=1),
                Write(y, run_time=1),
                lag_ratio=0.5,
            )
        )

        self.cleanup_slide()

    def construct_chapter2_2(self):
        self.next_slide()
        idx = 1
        X = np.loadtxt("digits.csv", delimiter=",")
        y = np.loadtxt("labels.csv", delimiter=",")

        img = ImageMobject(X[idx, :].reshape(28, 28)).scale(25)
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        img.to_edge(LEFT).shift(RIGHT)
        self.play(FadeIn(img))

        self.next_slide()

        attrs = X[idx, :].reshape(28, 28)
        img_binarized = np.zeros((28, 28))
        img_binarized[attrs > 128] = 1

        rng = np.random.default_rng(seed=0)
        neg_noise = img_binarized * np.clip(
            rng.normal(loc=50, scale=25, size=(28, 28)), 0, 255
        )
        attrs -= neg_noise

        pos_noise = (1 - img_binarized) * rng.uniform(low=0, high=30, size=(28, 28))
        attrs += pos_noise

        attrs[13:17, 9:20] += rng.normal(loc=70, scale=20, size=(4, 11))
        attrs[14:16, 9:20] += rng.normal(loc=70, scale=20, size=(2, 11))

        attr_img = ImageMobject(attrs).scale(25)
        attr_img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        attr_img.to_edge(RIGHT).shift(LEFT)
        self.play(FadeIn(attr_img))

        self.cleanup_slide()

    def construct_chapter2_3(self):
        self.next_slide()
        lesion_img = ImageMobject("lesion.png").scale(0.8).shift(LEFT * 5)
        self.play(FadeIn(lesion_img))

        # GENERAL FUNCTION
        rect = Rectangle(width=3, height=2)
        f = MathTex("f", font_size=100)

        self.next_slide()

        in_arrow = Arrow(
            start=lesion_img.get_edge_center(RIGHT), end=rect.get_edge_center(LEFT)
        )
        out_arrow = Arrow(start=LEFT, end=RIGHT).next_to(rect, direction=RIGHT)
        y = Text("malignant?", font_size=35).next_to(out_arrow, direction=RIGHT)
        self.play(
            LaggedStart(
                DrawBorderThenFill(in_arrow, run_time=1),
                AnimationGroup(Create(rect), Write(f)),
                DrawBorderThenFill(out_arrow, run_time=1),
                Write(y, run_time=1),
                lag_ratio=0.5,
            )
        )

        self.cleanup_slide()

    def construct_chapter2_4(self):
        self.next_slide()
        lesion_img = ImageMobject("lesions.png").scale(0.8).shift(UP)
        self.play(FadeIn(lesion_img))

        self.next_slide()
        markings_img = ImageMobject("markings.png").scale(0.8).shift(DOWN * 2)
        self.play(FadeIn(markings_img))

        self.cleanup_slide()

    def construct_chapter2_5(self):
        self.next_slide()
        imgs = []

        coords = [
            (-3, -2),
            (-3, 0),
            (-3, 2),
            (3, -2),
            (3, 0),
            (3, 2),
            (0, -3),
            (0, 0),
            (0, 2),
        ]

        for i in range(9):
            x, y = coords[i]
            img = ImageMobject(
                f"method{i+1}.png",
            ).move_to((x, y, 0))
            img.width = 6
            imgs.append(img)
        self.play(LaggedStart(*[FadeIn(img) for img in imgs], lag_ratio=0.8))

        self.cleanup_slide()

        imgs = []
        for i in range(9):
            x, y = coords[i]
            img = ImageMobject(
                f"metric{i+1}.png",
            ).move_to((x, y, 0))
            img.width = 6
            imgs.append(img)
        self.play(LaggedStart(*[FadeIn(img) for img in imgs], lag_ratio=0.8))

        self.cleanup_slide()

    def construct_chapter2_6(self):
        self.next_slide()
        scale = 0.275
        imgs = [
            ImageMobject("MNIST_default.png").scale(scale).shift(4.5 * LEFT),
            ImageMobject("CIFAR10_default.png").scale(scale),
            ImageMobject("ImageNet_default.png").scale(scale).shift(4.5 * RIGHT),
        ]
        self.play(FadeIn(img) for img in imgs)

        self.next_slide()
        self.play(*[FadeOut(obj) for obj in self.mobjects_without_canvas])

        text = Text("There is no universal measure of quality!", font_size=45)
        self.play(Write(text))

        self.cleanup_slide()

    def construct_chapter3_1(self):
        self.next_slide()
        variables = (
            VGroup(
                *[
                    Text(name, font_size=24)
                    for name in [
                        "Length",
                        "Weight",
                        "Age",
                        "Blood pressure",
                        "Cholesterol",
                    ]
                ]
            )
            .arrange(DOWN, buff=0.5)
            .to_edge(LEFT)
            .shift(0.25 * RIGHT)
        )
        for var in variables[1:]:
            var.align_to(variables[0], direction=RIGHT)

        self.play(Write(variables))

        # GENERAL FUNCTION
        rect = Rectangle(width=5, height=3)
        f = MathTex("f", font_size=100)

        arrows = []
        for obj in variables:
            arr = Line(
                start=obj.get_edge_center(RIGHT) + 0.2 * RIGHT,
                end=rect.get_edge_center(LEFT),
            )
            arrows.append(arr)

        out_arrow = Arrow(start=LEFT, end=0.5 * RIGHT).next_to(rect, direction=RIGHT)
        y = Text("Diabetes risk", font_size=35).next_to(out_arrow, direction=RIGHT)
        self.play(
            LaggedStart(
                *[Write(arr) for arr in arrows],
                AnimationGroup(Create(rect), Write(f)),
                DrawBorderThenFill(out_arrow, run_time=1),
                Write(y, run_time=1),
            )
        )

        self.next_slide()

        variables_new = (
            VGroup(
                *[
                    MathTex(name, font_size=44)
                    for name in ["L=195", "W=85", "A=45", "B=135", "C=110"]
                ]
            )
            .arrange(DOWN, buff=0.5)
            .to_edge(LEFT)
            .shift(0.5 * RIGHT)
        )

        y_new = Text("High risk", font_size=35).next_to(out_arrow, direction=RIGHT)

        self.play(
            ReplacementTransform(variables, variables_new),
            ReplacementTransform(y, y_new),
        )

        self.next_slide()

        variables_new[0].generate_target()
        variables_new[0].target.color = BLUE
        self.play(Circumscribe(variables_new[0]), MoveToTarget(variables_new[0]))

        self.next_slide()

        length = variables_new[0]
        no_length = MathTex(r"L = ???", font_size=44, color=BLUE).move_to(length)
        no_output = Text("???", font_size=35).next_to(out_arrow, direction=RIGHT)
        self.play(Transform(length, no_length), Transform(y_new, no_output))

        self.next_slide()

        zero_length = MathTex("L = 0", font_size=44, color=BLUE).move_to(length)
        y_zero = Text("Low risk", font_size=35).next_to(out_arrow, direction=RIGHT)
        self.play(Transform(length, zero_length), Transform(y_new, y_zero))

        self.next_slide()

        values = [9, 41, 24, 38, "..."]
        outputs = ["Low risk", "High risk", "Low risk", "High risk", "..."]
        for value, output in zip(values, outputs):
            new_length = MathTex(f"L = {value}", font_size=44, color=BLUE).move_to(
                length
            )
            new_output = Text(output, font_size=35).next_to(out_arrow, direction=RIGHT)
            self.play(Transform(length, new_length), Transform(y_new, new_output))

        self.next_slide()

        orig_length = MathTex(f"L = 195", font_size=44).move_to(length)
        orig_out = Text("High risk", font_size=35).next_to(out_arrow, direction=RIGHT)
        self.play(Transform(length, orig_length), Transform(y_new, orig_out))

        self.next_slide()

        new_f = Tex(
            r"""
        IF $B > 130$ OR $C > 100$\\
        THEN High risk\\ELSE Low risk
        """,
            font_size=40,
        )

        self.play(Transform(f, new_f))

        self.next_slide()

        bp_values = [120, 110, 140, 130, "..."]
        c_values = [100, 105, 115, 90, "..."]

        bp = variables_new[3]
        c = variables_new[4]
        for value in bp_values:
            new_bp = MathTex(f"B = {value}", font_size=44, color=BLUE).move_to(bp)
            self.play(
                Transform(bp, new_bp),
            )
        orig_bp = MathTex(f"B = 135", font_size=44).move_to(bp)
        self.play(Transform(bp, orig_bp))

        for value in c_values:
            new_c = MathTex(f"C = {value}", font_size=44, color=BLUE).move_to(c)
            self.play(
                Transform(c, new_c),
            )
        orig_c = MathTex(f"C = 110", font_size=44).move_to(c)
        self.play(Transform(c, orig_c))

        self.cleanup_slide()

    def construct_chapter3_2(self):
        self.next_slide()
        title = Text("Three ingredients:", font_size=50).shift(2 * UP)
        p = (
            paragraph(
                "1. Target: what to explain",
                "2. Removal: how to remove features",
                "3. Aggregation: how to summarize effects",
                t2c={"1. Target:": BLUE, "2. Removal:": BLUE, "3. Aggregation:": BLUE},
            )
            .to_edge(LEFT)
            .shift(0.5 * RIGHT)
        )

        self.play(Write(title))
        self.play(Write(p), run_time=6)

        self.cleanup_slide()

    def construct_chapter3_3(self):
        self.next_slide()
        eqn = VGroup(
            *[
                Tex(r"$m(f,i,\mathbf{x})$", font_size=70),
                Tex(
                    r"$= \sum_{S \subseteq [d]} \alpha_{S}\Phi(P_{S}(f))(\mathbf{x})$",
                    font_size=70,
                ),
            ]
        ).arrange(RIGHT)
        eqn[1].shift(0.075 * DOWN)
        self.play(Write(eqn))

        self.next_slide()

        lhs, rhs = eqn
        self.play(FadeOut(rhs))
        orig_point = lhs.get_center()
        self.play(lhs.animate.center())

        self.next_slide()

        rhss = [
            r"$= f(\mathbf{x})$",
            r"$= P_{S}(f)(\mathbf{x})$",
            r"$= \Phi(P_{S}(f))(\mathbf{x})$",
            r"$= \alpha_{S}\Phi(P_{S}(f))(\mathbf{x})$",
            r"$= \sum_{S \subseteq [d]} \alpha_{S}\Phi(P_{S}(f))(\mathbf{x})$",
        ]

        rhs1 = Tex(r"$=f$", font_size=70).move_to(rhs.get_center())
        self.play(lhs.animate.move_to(orig_point))
        self.play(FadeIn(rhs1))

        for rhsi in rhss[1:]:
            self.next_slide()
            rhs_tex = Tex(rhsi, font_size=70).move_to(rhs.get_center())
            self.play(Transform(rhs1, rhs_tex))
        self.cleanup_slide()

    def construct_chapter3_4(self):
        self.next_slide()
        eqns = (
            VGroup(
                *[
                    MathTex(r"\Phi(f)(\mathbf{x}) = f(\mathbf{x})", font_size=70),
                    Tex(
                        r"$P_S(f)(\mathbf{x}) = $ average output over dataset",
                        font_size=70,
                    ),
                    MathTex(
                        r"""
            \alpha_S = \begin{cases}
                1 & \mbox{if } S = \emptyset \\
                -1 & \mbox{if } S = \{i\}\\
                0 & \mbox{otherwise.}
            \end{cases}""",
                        font_size=70,
                    ),
                ]
            )
            .arrange(DOWN)
            .to_edge(DOWN)
        )
        for eqn in eqns:
            self.next_slide()
            self.play(Write(eqn))
        self.cleanup_slide()
        self.next_slide()

        full_eqn = Tex(
            r"$m(f,i,\mathbf{x})= \sum_{S \subseteq [d]} \alpha_{S}\Phi(P_{S}(f))(\mathbf{x})$",
            font_size=70,
        )

        self.play(Write(full_eqn))
        self.cleanup_slide()

    def construct_chapter4_1(self):
        self.next_slide()
        child_a = VGroup(
            *[
                SVGMobject("childa.svg").scale(2),
                Text("A", font_size=75, fill_color=BLACK).shift(0.2 * DOWN),
            ]
        )
        child_b = VGroup(
            *[
                SVGMobject("childb.svg").scale(2),
                Text("B", font_size=75, fill_color=BLACK).shift(0.2 * DOWN),
            ]
        )
        children = VGroup(*[child_a, child_b]).arrange(RIGHT)
        self.play(Write(children))

        self.next_slide()

        eqns = VGroup(
            *[
                MathTex(r"b(\{\}) = 0"),
                MathTex(r"b(\{A\}) = 25"),
                MathTex(r"b(\{B\}) = 20"),
                MathTex(r"b(\{A,B\}) = 5"),
                MathTex(r"b(\{A,B\}) < b(\{A\}) + b(\{B\})"),
                MathTex(r"i(\{A,B\}) &= b(\{A,B\}) - (b(\{A\}) + b(\{B\}))"),
            ]
        ).arrange(DOWN)

        self.play(children.animate.to_corner(UL).shift(LEFT).scale(0.5))

        for eqn in eqns[:-1]:
            self.next_slide()
            self.play(Write(eqn))

        self.next_slide()
        self.play(Transform(eqns[-2], MathTex(r"5 < 25 + 20").move_to(eqns[-2])))

        self.next_slide()
        self.play(Write(eqns[-1]))

        self.next_slide()
        self.play(
            Transform(
                eqns[-1], MathTex(r"i(\{A,B\}) = 5 - (25 + 20)").move_to(eqns[-1])
            )
        )

        self.next_slide()
        self.play(Transform(eqns[-1], MathTex(r"i(\{A,B\}) = -40").move_to(eqns[-1])))

        self.cleanup_slide()

    def construct_chapter4_2(self):
        equations = MathTex(
            r"f(\mathbf{x}) &= 2x_1 + 4x_2 - 3x_1x_2 + x_3 + 5\\",
            r"P_{\{1,2,3\}}(f)(\mathbf{x})",
            r"&= f(0,0,0)\\",
            r"&= 2*0 + 4*0 - 3*0*0 + 0 + 5\\",
            r"&= 5\\",
            r"P_{\{2,3\}}(f)(\mathbf{x})",
            r"&= f(x_1,0,0) \\",
            r"&= 2x_1 + 4*0 - 3*x_1*0 + 0 + 5 \\",
            r"&= 2x_1 + 5\\",
            r"e_1(x_1)",
            r"&= P_{\{2,3\}}(f)(\mathbf{x}) - P_{\{1,2,3\}}(f)(\mathbf{x}) \\",
            r"&= 2x_1\\",
            r"e_2(x_2) &= 4x_2\\",
            r"e_3(x_3) &= x_3\\",
            r"e_{1,2}(x_1,x_2)",
            r"&= P_{\{3\}}(f)(\mathbf{x}) - P_{\{1,2,3\}}(f)(\mathbf{x})\\",
            r"&= (2x_1 + 4x_2 - 3x_1x_2 + 0 + 5) - 5\\",
            r"&= 2x_1 + 4x_2 - 3x_1x_2\\",
            r"i_{1,2}(x_1,x_2)",
            r"&= e_{1,2}(x_1,x_2) - (e_1(x_1) + e_2(x_2))\\",
            r"&= (2x_1 + 4x_2 - 3x_1x_2) - (2x_1 + 4x_2)\\",
            r"&= -3x_1x_2",
        ).shift(4 * DOWN)

        self.next_slide()
        self.play(Write(equations[0]))

        self.next_slide()
        for eqn in equations[1:5]:
            self.play(Write(eqn))

        self.next_slide()
        self.play(FadeOut(eqn) for eqn in equations[2:4])
        self.play(
            equations[4].animate.move_to(equations[2]).align_to(equations[2], LEFT)
        )

        self.next_slide()
        for eqn in equations[5:9]:
            self.play(Write(eqn.shift(1.5 * UP)))

        self.next_slide()
        self.play(FadeOut(eqn) for eqn in equations[6:8])
        self.play(
            equations[8].animate.move_to(equations[6]).align_to(equations[6], LEFT)
        )

        self.next_slide()
        self.play(Write(equations[9].shift(3 * UP)))
        self.play(Write(equations[10].shift(3 * UP)))

        self.next_slide()
        self.play(Write(equations[11].shift(3 * UP)))

        self.next_slide()
        self.play(FadeOut(equations[10]))
        self.play(
            equations[11].animate.move_to(equations[10]).align_to(equations[10], LEFT)
        )

        self.next_slide()
        self.play(Write(equations[12].next_to(equations[11], RIGHT).shift(0.5 * RIGHT)))
        self.play(Write(equations[13].next_to(equations[12], RIGHT).shift(0.5 * RIGHT)))

        self.next_slide()
        for eqn in equations[14:18]:
            self.play(Write(eqn.shift(5.25 * UP)))

        self.next_slide()
        self.play(FadeOut(eqn) for eqn in equations[15:17])
        self.play(
            equations[17].animate.move_to(equations[15]).align_to(equations[15], LEFT)
        )

        self.next_slide()
        for eqn in equations[18:22]:
            self.play(Write(eqn.shift(6.75 * UP)))

        self.next_slide()
        self.play(FadeOut(eqn) for eqn in equations[19:21])
        self.play(
            equations[21].animate.move_to(equations[19]).align_to(equations[19], LEFT)
        )

        self.play(Wait())
        self.cleanup_slide()

    def construct_chapter4_3(self):
        self.next_slide()
        equations = MathTex(
            r"f(\mathbf{x}) &= 2x_1 + 4x_2 - 3x_1x_2 + x_3 + 5\\",
            r"&= e_1(x_1) + e_2(x_2) + i_{1,2}(x_1,x_2) + e_3(x_3) + 5",
        )
        self.play(Write(equations))
        self.cleanup_slide()

    def construct_chapter4_4(self):
        self.next_slide()
        eqn = MathTex(r"""
        f &= e_\emptyset\\
        &+ e_1 + e_2 + e_3 \\
        &+ i_{1,2} + i_{1,3} + i_{2,3} \\
        &+ i_{1,2,3} \\
        """).to_edge(LEFT).shift(RIGHT)
        self.play(Write(eqn))
        
        graph = make_hasse(
            vertex_config={"radius": 0.3},
            edge_config={"stroke_width": 10},
            labels={
                0: MathTex(r"e_\emptyset", color=BLACK),
                1: MathTex(r"e_1", color=BLACK),
                2: MathTex(r"e_2", color=BLACK),
                3: MathTex(r"i_{1,2}", color=BLACK, font_size=35),
                4: MathTex(r"e_3", color=BLACK),
                5: MathTex(r"i_{1,3}", color=BLACK, font_size=35),
                6: MathTex(r"i_{2,3}", color=BLACK, font_size=35),
                7: MathTex(r"i_{1,2,3}", color=BLACK, font_size=25)
            }
        ).to_edge(RIGHT).shift(LEFT)
        self.play(Write(graph))

        self.cleanup_slide()

    def construct_chapter4_6(self):
        self.next_slide()
        title = Text("Three ingredients:", font_size=50).shift(2 * UP)
        p = paragraph(
            "1. Target: what to explain",
            "2. Removal: how to remove features",
            "3. Aggregation: how to summarize effects",
            t2c={
                "1. Target:": BLUE,
                "2. Removal:": BLUE,
                "3. Aggregation:": BLUE
            }, font_size=35
        ).to_edge(LEFT).shift(0.5 * RIGHT)

        self.play(Write(title))
        self.play(Write(p), run_time=4)

        self.next_slide()
        decomp = Text("2. Decomposition: how to decompose the function",t2c={
                "2. Decomposition:": BLUE,
            }, font_size=35).move_to(p[1]).align_to(p[0], direction=LEFT)
        self.play(Transform(p[1], decomp))

        self.cleanup_slide()


    def construct(self):
        self.construct_titleslide()
        self.construct_toc_slide(chapter=1)

        self.construct_chapter1_1()
        self.construct_chapter1_2()
        self.construct_chapter1_3()
        self.construct_chapter1_4()
        self.construct_chapter1_5()  # 3D SCENE
        self.construct_chapter1_6()
        self.construct_chapter1_7()
        self.construct_chapter1_8()
        self.construct_chapter1_9()
        self.construct_chapter1_10()

        self.construct_toc_slide(chapter=2)
        self.construct_chapter2_1()
        self.construct_chapter2_2()
        self.construct_chapter2_3()
        self.construct_chapter2_4()
        self.construct_chapter2_5()
        self.construct_chapter2_6()

        self.construct_toc_slide(chapter=3)
        self.construct_chapter3_1()
        self.construct_chapter3_2()
        self.construct_chapter3_3()
        self.construct_chapter3_4()

        self.construct_toc_slide(chapter=4)
        self.construct_chapter4_1()
        self.construct_chapter4_2()
        self.construct_chapter4_3()
        self.construct_chapter4_4()
        self.construct_chapter4_5()
