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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.default_run_time = 0.8
        self.long_run_time = 2 * self.default_run_time

        self.subtitles = None

    def set_subtitles(self, *texts):
        result = []
        if len(texts) == 0:
            if len(self.subtitles) > 0:
                result.append(FadeOut(self.subtitles, run_time=0.5))
            self.subtitles = None
        else:
            if self.subtitles is not None:
                result.append(FadeOut(self.subtitles, run_time=0.5))
            subs = (
                VGroup(*[Tex(s, font_size=30) for s in texts])
                .arrange(DOWN, buff=0.1)
                .to_edge(UP, buff=0.2)
            ).set_z_index(2)
            padding = 0.2
            square = (
                Rectangle(
                    stroke_opacity=0,
                    fill_color=config.background_color,
                    fill_opacity=0.7,
                    width=subs.width + padding,
                    height=subs.height + padding,
                )
                .move_to(subs.get_center())
                .set_z_index(subs.z_index - 1)
            )
            self.subtitles = VGroup(subs, square)
            self.add_fixed_in_frame_mobjects(self.subtitles)
            result.append(FadeIn(self.subtitles, run_time=0.5))
        return result

    def cleanup_slide(self):
        self.next_slide()
        self.play(
            *[FadeOut(obj) for obj in self.mobjects_without_canvas],
            run_time=self.default_run_time,
        )
        self.subtitles = None

    def construct_toc_slide(self, chapter):
        contents_title = Tex("Contents", color=WHITE, font_size=60).to_corner(UL)

        contents = Tex(
            r"\item Introduction\vspace{-0.5em}",
            r"\item Benchmarking Attribution Methods in the Image Domain\vspace{-0.5em}",
            r"\item Removal-Based Attribution Methods\vspace{-0.5em}",
            r"\item Functional Decomposition\vspace{-0.5em}",
            r"\item PDD-SHAP\vspace{-0.5em}",
            r"\item Conclusion",
            font_size=40, color=WHITE, tex_environment="enumerate"
        )
        contents = contents.align_to(contents_title, LEFT).shift(RIGHT)

        self.play(
            FadeIn(contents), FadeIn(contents_title), run_time=self.default_run_time
        )
        self.next_slide()
        cur = contents[chapter - 1]

        cur.generate_target()
        cur.target.set_fill(color=BLUE)
        cur.target.scale(1.2)

        self.play(
            FadeOut(contents[:chapter -1]),
            FadeOut(contents[chapter:]),
            MoveToTarget(cur),
            run_time=self.default_run_time,
        )
        self.cleanup_slide()

    def construct_titleslide(self):
        self.next_slide(loop=True)
        g = make_hasse(
            vertex_config={"radius": 0.3, "color": GREY},
            edge_config={"stroke_width": 10, "color": GREY},
        )
        self.play(Write(g), run_time=3)
        self.play(Wait())
        self.play(FadeOut(g))

        self.next_slide()
        self.play(FadeOut(g))

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

        self.play(FadeIn(title1), FadeIn(title2))
        self.play(FadeIn(author_date))
        self.cleanup_slide()

        rutte = ImageMobject("rutte.png").scale(0.55).shift(DOWN * 0.5)
        self.play(
            *self.set_subtitles(
                """
            On January 15, 2021, the Dutch third Rutte cabinet resigned.
            The main reason was the so-called ``toeslagenaffaire'', or childcare benefits scandal.
            """
            ),
            FadeIn(rutte, run_time=self.default_run_time),
        )

        self.next_slide()
        self.play(
            *self.set_subtitles(
                """
            In this scandal, thousands of people were falsely accused of fraud by an AI system.
            This AI system turned out to be biased in important ways.
            """
            )
        )

        self.next_slide()
        self.play(
            *self.set_subtitles(
                """
            This illustrates the need for Explainable AI, or XAI.
            In this presentation, I will focus on ``attribution-based'' explanations specifically.
            """
            )
        )
        self.cleanup_slide()

    def construct_chapter1_1(self):
        # GENERAL FUNCTION
        rect = Rectangle(width=5, height=3)
        f = MathTex("f", font_size=100)
        self.next_slide()

        self.play(
            Create(rect),
            Write(f),
            *self.set_subtitles(
                "Before we begin, we take a look at a function. A function is like a machine: it takes an input, and produces an output."
            ),
            run_time=self.default_run_time,
        )

        self.next_slide()

        in_arrow = Arrow(start=LEFT, end=RIGHT).next_to(rect, direction=LEFT)
        out_arrow = Arrow(start=LEFT, end=RIGHT).next_to(rect, direction=RIGHT)
        x = MathTex("x", font_size=100).next_to(in_arrow, direction=LEFT)
        y = MathTex("y", font_size=100).next_to(out_arrow, direction=RIGHT)
        self.play(
            DrawBorderThenFill(in_arrow), Write(x), run_time=self.default_run_time
        )

        self.play(
            DrawBorderThenFill(out_arrow), Write(y), run_time=self.default_run_time
        )

        self.next_slide()

        # EXAMPLE FUNCTION
        fun, fmt_str = example_function_1()
        ex_f = MathTex(fmt_str.format("x", "x"))
        self.play(
            Transform(f, ex_f, run_time=self.default_run_time),
            *self.set_subtitles(
                "The ``instructions'' of the machine can be expressed in an equation."
            ),
        )

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
                run_time=self.default_run_time,
            )
            self.next_slide()

        self.cleanup_slide()

    def construct_chapter1_2(self):
        self.next_slide()
        # GRAPH
        axis_config = {"include_ticks": True, "include_numbers": True}
        plane = NumberPlane(x_axis_config=axis_config, y_axis_config=axis_config)
        self.play(
            Write(plane),
            *self.set_subtitles(
                "We can show this visually by drawing a dot for each input-output pair."
            ),
            run_time=3 * self.default_run_time,
        )

        self.next_slide()

        fun, fmt_str = example_function_1()

        red_dots = []
        for x in [6, 4]:
            dot = Dot(color=RED).move_to([x, 0, 0])
            dot.generate_target()
            out = fun(x)
            dot.target.shift(out * UP)
            red_dots.append(dot)

        self.play(
            *[
                DrawBorderThenFill(dot, run_time=self.default_run_time)
                for dot in red_dots
            ]
        )

        self.next_slide()
        self.play(
            *[MoveToTarget(dot, run_time=self.default_run_time) for dot in red_dots]
        )

        dots = []
        for x in range(-35, 35, 1):
            x = x / 10 * 2
            dot = Dot(color=WHITE).move_to([x, 0, 0]).scale(0.5)
            dot.generate_target()
            dot.target.shift(fun(x) * UP)
            dots.append(dot)

        self.next_slide()

        self.play(
            *[DrawBorderThenFill(dot, run_time=self.default_run_time) for dot in dots],
            *self.set_subtitles("Repeating this for all inputs, we get a line."),
        )

        self.next_slide()
        self.play(LaggedStart(*[MoveToTarget(dot) for dot in dots], lag_ratio=0.01))

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
            "f(x) = 0.5 * x - 1", font_size=100, substrings_to_isolate=["0.5", "- 1"]
        )
        self.play(
            Write(eq),
            *self.set_subtitles(
                "Let's take another look at the function. We see that it has 2 parameters."
            ),
            run_time=self.long_run_time,
        )
        self.next_slide()

        params = ["0.5", "- 1"]

        for param in params:
            p = eq.get_parts_by_tex(param)
            p.generate_target()
            p.target.set_color(YELLOW)
            self.play(Circumscribe(p), MoveToTarget(p), run_time=self.default_run_time)
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
        self.play(Write(plane), MoveToTarget(eq), run_time=self.long_run_time)
        self.play(Create(graph), run_time=self.default_run_time)

        self.next_slide()

        graph2 = plane.plot(lambda x: x - 1, color=WHITE)
        eq2 = (
            MathTex("f(x) = 1 * x - 1", font_size=100, substrings_to_isolate=["1"])
            .to_corner(UL)
            .shift(2 * LEFT)
            .scale(0.5)
        )
        eq2.set_color_by_tex("1", YELLOW)
        self.play(
            Transform(graph, graph2),
            Transform(eq, eq2),
            *self.set_subtitles(
                "Manipulating the first parameter changes the slope of the function."
            ),
            run_time=self.default_run_time,
        )

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
        self.play(
            Transform(graph, graph3), Transform(eq, eq3), run_time=self.default_run_time
        )

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
        self.play(
            Transform(graph, graph4),
            Transform(eq, eq4),
            *self.set_subtitles(
                "Manipulating the second parameter moves the function up and down."
            ),
            run_time=self.default_run_time,
        )
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
        self.play(
            Transform(graph, graph5), Transform(eq, eq5), run_time=self.default_run_time
        )

        self.next_slide()

        self.play(
            *self.set_subtitles(
                """Machine learning is simply letting the computer choose parameter values
                  such that the resulting function is somehow useful.
                  """
            )
        )

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
        x_label = MathTex("T", font_size=40).to_corner(DR, buff=0.3)
        y_label = MathTex("E", font_size=40).to_edge(UP, buff=0.3).shift(4.5 * LEFT)
        self.play(
            Write(plane),
            *self.set_subtitles(
                r"Imagine we want to predict energy usage in our home\\based on the temperature outside."
            ),
            run_time=self.long_run_time,
        )
        self.play(Write(x_label), Write(y_label))

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

        self.play(
            *[Create(dot) for dot in dots],
            *self.set_subtitles(
                "We measure the temperature and resulting energy usage on a few days."
            ),
            run_time=self.default_run_time,
        )

        self.next_slide()

        self.play(
            *self.set_subtitles(
                "If we want to predict the energy usage when the temperature is 12 degrees, we can draw a line through the data and measure its height when $T=12$."
            )
        )

        self.next_slide()

        # Random line
        rand_line = plane.plot(lambda x: 0.1 * x + 0.5)
        rand_mse = root_mean_squared_error(y_points, 0.1 * x_points + 0.5)
        rand_eqn = (
            MathTex("f(x) &=", f"0.10", "* x", r"+0.50\\", "L &=", f"{rand_mse:.2f}")
        ).to_corner(DL, buff=0.2).set_z_index(2)
        rand_eqn[1].color = YELLOW
        rand_eqn[3].color = YELLOW
        rand_eqn[4].color = RED
        rand_eqn[5].color = RED
        padding = 0.2
        bg_rect = (
            Rectangle(
                stroke_opacity=0,
                fill_color=config.background_color,
                fill_opacity=0.7,
                width=rand_eqn.width + padding,
                height=rand_eqn.height + padding,
            )
            .move_to(rand_eqn.get_center())
            .set_z_index(rand_eqn.z_index - 1)
        )

        self.play(
            Create(rand_line),
            Write(rand_eqn),
            FadeIn(bg_rect),
            *self.set_subtitles(
                "First we choose random values for the parameters. This line obviously does not describe the data very well."
            ),
            run_time=self.default_run_time,
        )

        self.next_slide()

        self.play(
            *self.set_subtitles(
                "The ``loss'' function $L$ describes the quality of the line. The higher the value for $L$, the worse the line."
            )
        )

        self.next_slide()

        lm = linreg_univariate(x_points, y_points)
        reg_line = plane.plot(lambda x: lm.predict(np.array(x).reshape(-1, 1))[0])
        reg_mse = root_mean_squared_error(y_points, lm.predict(np.array(x_points).reshape(-1, 1)))
        reg_eqn = (
            MathTex("f(x) &=", f"{lm.coef_[0]:.2f}", "* x", f"+{lm.intercept_:.2f}\\\\", "L &=", f"{reg_mse:.2f}")
        ).to_corner(DL, buff=0.2).set_z_index(2)
        reg_eqn[1].color = YELLOW
        reg_eqn[3].color = YELLOW
        reg_eqn[4].color = RED
        reg_eqn[5].color = RED
        padding = 0.2
        reg_bg_rect = (
            Rectangle(
                stroke_opacity=0,
                fill_color=config.background_color,
                fill_opacity=0.7,
                width=reg_eqn.width + padding,
                height=reg_eqn.height + padding,
            )
            .move_to(reg_eqn.get_center())
            .set_z_index(reg_eqn.z_index - 1)
        )
        self.play(
            Transform(rand_line, reg_line),
            Transform(rand_eqn, reg_eqn),
            Transform(bg_rect, reg_bg_rect),
            *self.set_subtitles(
                "The computer searches for parameter values that minimize $L$, and now the line nicely goes through the data. This process is also called ``training''."
            ),
            run_time=self.default_run_time,
        )
        self.cleanup_slide()

    def construct_chapter1_5(self):
        self.next_slide()

        self.set_camera_orientation(theta=-45 * DEGREES, phi=75 * DEGREES)
        self.play(
            *self.set_subtitles(
                """Now assume we also measure the number of people in the house.
            Now there are 2 input variables.
            """
            )
        )

        axes = ThreeDAxes()

        rng = np.random.default_rng(seed=0)
        n_points = 10
        data = rng.uniform(low=-3, high=3, size=(n_points, 2))

        dots = [
            Dot3D(color=RED, point=axes.coords_to_point(row[0], row[1], 0))
            for row in data
        ]

        self.next_slide()
        self.play(
            Write(axes),
            *[FadeIn(dot) for dot in dots],
            *self.set_subtitles(
                "For each of the 2 input variables we need an axis, so now we get a 3D plot."
            ),
            run_time=self.long_run_time,
        )

        self.next_slide()

        self.begin_ambient_camera_rotation(rate=75 * DEGREES / 4)

        def outcome(x):
            return 0.5 * x[:, 0] - 0.25 * x[:, 1]

        rng = np.random.default_rng(seed=0)
        noise = rng.normal(0, 0.5, n_points)
        y = outcome(data) + noise

        for dot, y_val in zip(dots, y):
            dot.generate_target()
            dot.target.shift(axes.coords_to_point(0, 0, y_val))
        self.play(
            *[MoveToTarget(dot) for dot in dots],
            *self.set_subtitles(
                "Each dot represents a value for temperature and number of people. The height of each dot is the resulting energy usage."
            ),
            run_time=self.default_run_time,
        )

        self.play(Wait(run_time=4))

        surface = Surface(
            lambda u, v: axes.c2p(u, v, 0.5 * u - 0.25 * v),
            u_range=[-7, 7],
            v_range=[-7, 7],
            resolution=8,
            fill_opacity=0.5,
        )
        self.play(
            DrawBorderThenFill(surface),
            *self.set_subtitles(
                "Instead of a straight line, our model now becomes a flat plane."
            ),
            run_time=self.long_run_time,
        )

        self.play(Wait(run_time=4))

        self.cleanup_slide()

        self.set_camera_orientation(theta=-90 * DEGREES, phi=0)
        self.stop_ambient_camera_rotation()

    def construct_chapter1_6(self):
        rows = [Text(f"Day {i+1}") for i in range(6)]
        cols = [
            Tex(label)
            for label in [
                "$x_1$: Temp (C)",
                "$x_2$: People",
                "$x_3$: Sun (\%)",
                "$y$: Energy",
            ]
        ]
        data = [
            [14, 3, 45],
            [17, 4, 55],
            [13, 3, 50],
            [9, 2, 45],
            [11, 3, 60],
            [10, 4, 40],
        ]

        def outcome(x):
            rng = np.random.default_rng(seed=0)
            noise = rng.normal(0, 0.5, x.shape[0])
            return -0.1 * x[:, 0] + 3 * x[:, 1] - 0.1 * x[:, 2] + 9 + noise

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
        self.play(
            table.create(run_time=self.long_run_time),
            *self.set_subtitles(
                "Assume we also measure the percentage of sunlight in the day. Now we have 3 input variables."
            ),
        )

        self.next_slide()

        pred_before = [x[0] * 5.1 + x[1] * 1.3 - x[2] * 5.4 + 2.8 for x in data]
        mse_before = root_mean_squared_error(y, pred_before)

        eqn_before = (
            MathTex(
                "f(x_1, x_2, x_3) =",
                "5.1",
                "* x_1",
                "+ 1.3",
                "* x_2",
                "- 5.4",
                "* x_3",
                "+ 2.8",
            )
            .to_corner(UL)
            .shift(1 * DOWN)
        )
        for idx in [1, 3, 5, 7]:
            eqn_before[idx].color = YELLOW

        error_before = (
            MathTex(f"L = {mse_before:.2f}", color=RED)
            .next_to(eqn_before, DOWN)
            .align_to(eqn_before, LEFT)
        )

        self.play(
            Write(eqn_before),
            *self.set_subtitles(
                "We cannot visualize this situation anymore in a plot."
            ),
            run_time=self.default_run_time,
        )
        self.play(Write(error_before), run_time=self.default_run_time)

        self.next_slide()

        lm = linreg_multivariate(np.array(data), outcome(np.array(data)))
        eqn_after = (
            MathTex(
                "f(x_1, x_2, x_3) =",
                f"{lm.coef_[0]:.1f}",
                "* x_1",
                f"+{lm.coef_[1]:.1f}",
                "* x_2",
                f"{lm.coef_[2]:.1f}",
                "* x_3",
                f"+{lm.intercept_:.1f}",
            )
            .move_to(eqn_before)
            .align_to(eqn_before, LEFT)
        )
        for idx in [1, 3, 5, 7]:
            eqn_after[idx].color = YELLOW

        mse_after = root_mean_squared_error(y, lm.predict(np.array(data)))
        mse_after = f"{mse_after:.2f}".zfill(6)
        error_after = (
            MathTex(f"L = {mse_after}", color=RED)
            .move_to(error_before)
            .align_to(error_before, LEFT)
        )

        self.play(
            Transform(eqn_before, eqn_after),
            Transform(error_before, error_after),
            *self.set_subtitles(
                "However, this is not a problem for the computer: it simply minimizes the loss function $L$."
            ),
            run_time=self.default_run_time,
        )

        self.next_slide()

        self.play(
            *self.set_subtitles(
                """We can look at the equation to see what the model does:
            temperature and sunlight have a negative influence on energy usage,
            number of people has a positive influence.
        """
            )
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
        reg_eqn = (
            MathTex("f(x) =", f"{lm.coef_[0]:.2f}", "* x", f"+{lm.intercept_:.2f}")
            .to_corner(DL, buff=0.2)
            .set_z_index(2)
        )

        padding = 0.2
        bg_rect = (
            Rectangle(
                stroke_opacity=0,
                fill_color=config.background_color,
                fill_opacity=0.7,
                width=reg_eqn.width + padding,
                height=reg_eqn.height + padding,
            )
            .move_to(reg_eqn.get_center())
            .set_z_index(reg_eqn.z_index - 1)
        )

        reg_eqn[1].color = YELLOW
        reg_eqn[3].color = YELLOW
        reg_line = plane.plot(lambda x: lm.predict(np.array(x).reshape(-1, 1))[0])

        self.play(
            Write(plane),
            *[Create(dot) for dot in dots],
            Create(reg_line),
            Write(reg_eqn),
            FadeIn(bg_rect),
            *self.set_subtitles(
                "Let's take another look at the single-variable model. This model can only draw straight lines."
            ),
            run_time=self.long_run_time,
        )

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

        self.play(
            *[MoveToTarget(dot) for dot in dots],
            *self.set_subtitles("This becomes a problem if the data looks like this."),
            run_time=self.default_run_time,
        )

        self.next_slide()

        lm_para = parabolic_reg(x_points, y_parabola)

        para_model = lm_para["model"]
        para_eqn = (
            MathTex(
                "f(x) =",
                f"{para_model.coef_[1]:.2f}",
                "* x^2",
                f"{para_model.coef_[0]:.2f}",
                "* x",
                f"+{para_model.intercept_:.2f}",
            )
            .to_corner(DL, buff=0.2)
            .set_z_index(2)
        )

        padding = 0.2
        para_bg_rect = (
            Rectangle(
                stroke_opacity=0,
                fill_color=config.background_color,
                fill_opacity=0.7,
                width=para_eqn.width + padding,
                height=para_eqn.height + padding,
            )
            .move_to(para_eqn.get_center())
            .set_z_index(para_eqn.z_index - 1)
        )

        para_eqn[1].color = YELLOW
        para_eqn[3].color = YELLOW
        para_eqn[5].color = YELLOW

        para_line = plane.plot(lambda x: lm_para.predict(np.array(x).reshape(-1, 1))[0])
        self.play(
            Transform(reg_line, para_line),
            Transform(reg_eqn, para_eqn),
            Transform(bg_rect, para_bg_rect),
            *self.set_subtitles(
                "However, by adding a parameter, we can let the model learn this kind of pattern as well."
            ),
            run_time=self.default_run_time,
        )

        # CHANGE TO MORE COMPLEX DATA

        self.next_slide()

        self.play(FadeOut(reg_line), FadeOut(reg_eqn), FadeOut(bg_rect))

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
        self.play(
            *[MoveToTarget(dot) for dot in dots],
            *self.set_subtitles("Now what if the data looks like this?"),
            run_time=self.default_run_time,
        )

        self.next_slide()

        nn = nn_reg(x_points, y_noise)
        nn_line = plane.plot(lambda x: nn.predict(np.array(x).reshape(-1, 1))[0])
        self.play(
            Write(nn_line),
            *self.set_subtitles(
                "No problem: we can still learn a model for this data."
            ),
        )

        self.cleanup_slide()

    def construct_chapter1_8(self):
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

        formula = MathTex(nn_string).scale(2)
        self.play(Write(formula))
        self.next_slide()

        formula.generate_target()
        formula.target.scale(0.3)
        formula.target.to_edge(DOWN)

        self.play(MoveToTarget(formula, run_time=5))
        self.play(
            *self.set_subtitles(
                """
        This is a neural network. This particular one has 100 parameters... It is basically impossible to interpret."""
            )
        )

        self.cleanup_slide()

    def construct_chapter1_9(self):
        self.next_slide()
        title_text = Text("ChatGPT (GPT-3.5):").shift(2 * UP)
        self.play(Write(title_text))

        self.next_slide()
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
        self.play(
            FadeIn(map_img),
            Write(dist_text),
            *self.set_subtitles(
                "If we would write all of those parameters on standard A4 paper, the stack would be about 700km high, which would cover the distance from here to Berlin."
            ),
        )

        self.next_slide()

        self.play(
            *self.set_subtitles(
                """If you write at 1 parameter per second, this would take you 55,492 years.
        Obviously, we can't interpret this model by looking at the formula."""
            )
        )

        self.cleanup_slide()

    def construct_chapter1_10(self):
        self.next_slide()
        rect = Rectangle(width=5, height=3).shift(0.4 * DOWN)
        f = MathTex("f", font_size=100).shift(0.4 * DOWN)
        self.play(
            Create(rect),
            Write(f),
            *self.set_subtitles(
                "However, this giant model is still just a function. We can play with the inputs and look at the effect."
            ),
        )

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
            *[
                DrawBorderThenFill(in_arrow, run_time=self.default_run_time)
                for in_arrow in in_arrows
            ],
            *[Write(x, run_time=1) for x in in_vars],
        )
        self.play(
            DrawBorderThenFill(out_arrow, run_time=self.default_run_time),
            Write(y, run_time=self.default_run_time),
        )

        self.next_slide()

        params = [(0, 1.1, 0.01), (1, 2, 0.05), (2, 1.2, 0.015)]

        self.play(
            *self.set_subtitles(
                """
        If changing a variable has a strong impact on the output, then this variable is probably important.
        Otherwise, the variable is probably not very important.
        """
            )
        )
        for i, scale, rot in params:
            self.play(Circumscribe(in_vars[i]), run_time=self.default_run_time)
            self.play(
                Wiggle(in_vars[i], run_time=self.long_run_time),
                Wiggle(
                    y,
                    scale_value=scale,
                    rotation_angle=rot * TAU,
                    run_time=self.long_run_time,
                ),
            )

            self.next_slide()

        self.cleanup_slide()

        bar = (
            BarChart(values=[0.1, 0.5, 0.2], bar_names=[f"$x_{i+1}$" for i in range(3)])
            .scale(1.5)
            .to_edge(DOWN, buff=0.1)
        )
        self.play(
            Write(bar, run_time=2),
            *self.set_subtitles(
                "If we quantify this in a score per variable, we get an attribution-based explanation."
            ),
        )

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

        self.play(
            *[FadeIn(img) for img in imgs],
            *self.set_subtitles(
                "In this part, we focus on the setting of image classification."
            ),
            run_time=self.default_run_time,
        )
        self.play(LaggedStart(*[Write(label) for label in labels], lag_ratio=0.25))

        self.next_slide()

        self.play(
            *[
                FadeOut(img, run_time=self.default_run_time)
                for img in (imgs[0], imgs[2])
            ],
            *[FadeOut(label, run_time=self.default_run_time) for label in labels],
        )

        img = imgs[1]
        img.generate_target()
        img.target.center()
        img.target.scale(1.8)
        self.play(MoveToTarget(img, run_time=self.default_run_time))

        self.next_slide()

        v_img = PixelsAsSquares(img)
        self.play(FadeOut(img), FadeIn(v_img))

        circles = PixelsAsCircles(v_img)
        values = circles.get_values()
        self.play(
            ReplacementTransform(v_img, circles, run_time=self.default_run_time),
            *self.set_subtitles("Every pixel in the image is just a number."),
        )
        self.play(*[FadeIn(value) for value in values])

        self.next_slide()

        self.play(*[FadeOut(value) for value in values])

        rows = VGroup(
            *[VGroup(*circles.neurons[28 * i : 28 * (i + 1)]) for i in range(28)]
        )

        self.play(
            rows.animate.space_out_submobjects(1.2),
            *self.set_subtitles(
                "Each of these numbers becomes an input variable for the model."
            ),
        )
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

        self.play(MoveToTarget(summarized), run_time=self.default_run_time)

        rect = Rectangle(width=5, height=3)
        f = MathTex("f", font_size=100)
        self.play(Create(rect), Write(f), run_time=self.default_run_time)

        arrows = []
        for obj in summarized:
            if isinstance(obj, Circle):
                arr = Line(
                    start=obj.get_edge_center(RIGHT) + 0.2 * RIGHT,
                    end=rect.get_edge_center(LEFT),
                )
                arrows.append(arr)
        self.play(
            LaggedStart(*[Write(arr) for arr in arrows]), run_time=self.long_run_time
        )

        out_arrow = Arrow(start=LEFT, end=RIGHT).next_to(rect, direction=RIGHT)
        y = MathTex("y", font_size=100).next_to(out_arrow, direction=RIGHT)
        self.play(
            LaggedStart(
                DrawBorderThenFill(out_arrow, run_time=self.default_run_time),
                Write(y, run_time=self.default_run_time),
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
        self.play(
            FadeIn(img),
            *self.set_subtitles(
                "Attributions are very useful in this setting because they can easily be visualized as a new image."
            ),
            run_time=self.default_run_time,
        )

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
        self.play(
            FadeIn(attr_img),
            *self.set_subtitles(
                "In this case, the pixels inside the number zero are also important: if they were different, the number might have been an eight."
            ),
            run_time=self.default_run_time,
        )

        self.next_slide()

        self.play(
            *self.set_subtitles(
                r"""
        Note: these attributions are for a \textit{single} image. These are \textit{local} attributions.
        If the attributions pertain to the model in general, they are \textit{global}.
        """
            )
        )

        self.cleanup_slide()

    def construct_chapter2_3(self):
        self.next_slide()
        self.play(
            *self.set_subtitles(
                "Attributions are useful because they help us see what the model has learned."
            )
        )
        self.next_slide()
        lesion_img = ImageMobject("lesion.png").scale(0.8).shift(LEFT * 5)
        self.play(
            FadeIn(lesion_img),
            *self.set_subtitles(
                "An example: if we could automatically classify lesions as malignant or benign, this would be very useful for early diagnosis."
            ),
            run_time=self.default_run_time,
        )

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
            *self.set_subtitles("So researchers trained a model to do exactly this."),
            LaggedStart(
                DrawBorderThenFill(in_arrow, run_time=self.default_run_time),
                AnimationGroup(Create(rect), Write(f), run_time=self.default_run_time),
                DrawBorderThenFill(out_arrow, run_time=self.default_run_time),
                Write(y, run_time=self.default_run_time),
                lag_ratio=0.5,
            ),
        )

        self.next_slide()

        self.play(
            *self.set_subtitles(
                "The model seemed to work during testing, but it didn't work anymore when the model was actually deployed in the real world."
            )
        )

        self.cleanup_slide()

    def construct_chapter2_4(self):
        self.next_slide()
        lesion_img = ImageMobject("lesions.png").scale(0.8).shift(UP)
        self.play(
            FadeIn(lesion_img),
            *self.set_subtitles(
                "When a doctor diagnoses a lesion as malignant, they add markings for the surgeons to remove it."
            ),
            run_time=self.default_run_time,
        )

        self.next_slide()
        markings_img = ImageMobject("markings.png").scale(0.8).shift(DOWN * 2)
        self.play(
            FadeIn(markings_img),
            *self.set_subtitles(
                "When the researchers looked at attributions, the problem became clear: the model simply learned to detect the markings."
            ),
            run_time=self.default_run_time,
        )

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
        self.play(
            LaggedStart(*[FadeIn(img) for img in imgs], lag_ratio=0.8),
            *self.set_subtitles(
                "Many attribution methods have been introduced in recent years. If we would have a way to measure the quality of these methods, it would help us choose the right one."
            ),
        )

        self.cleanup_slide()

        imgs = []
        for i in range(9):
            x, y = coords[i]
            img = ImageMobject(
                f"metric{i+1}.png",
            ).move_to((x, y, 0))
            img.width = 6
            imgs.append(img)
        self.play(
            LaggedStart(*[FadeIn(img) for img in imgs], lag_ratio=0.8),
            *self.set_subtitles(
                r"Many quality metrics have been introduced in recent years. Nearly all of them are designed to measure the same thing: the \textit{correctness} of the attributions."
            ),
        )

        self.cleanup_slide()

    def construct_chapter2_6(self):
        self.next_slide()
        self.play(
            *self.set_subtitles(
                "The experiment was simple: let's gather some methods, metrics, and datasets, and compare the quality scores of each method, according to each metric, on each dataset."
            )
        )
        self.next_slide()
        scale = 0.14
        imgs = [
            ImageMobject("MNIST_default.png").scale(scale),
            ImageMobject("CIFAR10_default.png").scale(scale),
            ImageMobject("ImageNet_default.png").scale(scale),
        ]
        imgs[-1].set(height=imgs[0].height)
        imgs_group = Group(*imgs).arrange(RIGHT)
        self.play(
            FadeIn(imgs_group),
            *self.set_subtitles("Here are some of the results."),
            run_time=self.default_run_time,
        )

        self.next_slide()
        self.play(
            *self.set_subtitles(
                "We see that the ``quality'' of the methods depends on the dataset, but also on the metric."
            )
        )
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

        self.play(
            Write(variables, run_time=self.long_run_time),
            *self.set_subtitles(
                "Assume we have a model that predicts diabetes risk based on length, weight, age, blood pressure and cholesterol. Let's design an attribution method for this model."
            ),
        )

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
            ),
        )

        self.next_slide()

        self.play(
            *self.set_subtitles(
                "The first choice we need to make is: exactly what do we want to explain? A single prediction? The model behaviour in general? The model performance?"
            )
        )

        self.next_slide()

        self.play(
            *self.set_subtitles(
                r"This is the first ingredient of an explanation: \textbf{the target.}"
            )
        )

        self.next_slide()

        variables_new = (
            VGroup(
                *[
                    MathTex(*name, font_size=44)
                    for name in [
                        ["L=195"],
                        ["W=85"],
                        ["A=", "45"],
                        ["B=", "135"],
                        ["C=", "110"],
                    ]
                ]
            )
            .arrange(DOWN, buff=0.5)
            .to_edge(LEFT)
            .shift(0.5 * RIGHT)
        )

        y_new = Text("Risk = 60%", font_size=35).next_to(out_arrow, direction=RIGHT)

        self.play(
            ReplacementTransform(variables, variables_new),
            ReplacementTransform(y, y_new),
            *self.set_subtitles(
                "Assume we want an explanation for a specific prediction. This is the target."
            ),
            run_time=self.default_run_time,
        )

        self.next_slide()

        age = variables_new[2]
        age.generate_target()
        age.target.color = BLUE
        self.play(
            Circumscribe(age),
            MoveToTarget(age),
            *self.set_subtitles(
                "Assume we want to compute an attribution for the variable ``age''."
            ),
            run_time=self.default_run_time,
        )

        self.next_slide()

        no_age = MathTex(*["A =", "???"], font_size=44, color=BLUE).move_to(age)
        no_output = Text("???", font_size=35).next_to(out_arrow, direction=RIGHT)
        self.play(
            Transform(age, no_age),
            Transform(y_new, no_output),
            *self.set_subtitles(
                "We would like to ask the model what it would predict if it didn't know a value for age. If this changes the output drastically, then age is important."
            ),
            run_time=self.default_run_time,
        )

        self.next_slide()

        zero_age = MathTex(*["A =", "0"], font_size=44, color=BLUE).move_to(age)
        y_zero = Text("Risk = -315%", font_size=35).next_to(out_arrow, direction=RIGHT)
        self.play(
            Transform(age, zero_age),
            Transform(y_new, y_zero),
            *self.set_subtitles(
                "We could set age to zero, but the model doesn't interpret this as ``no age'', but as ``an age of zero''."
            ),
        )

        self.next_slide()

        values = [9, 41, 24, 38, "..."]
        outputs = ["Risk = 15%", "Risk = 30%", "Risk = 10%", "Risk = 50%", "..."]
        self.play(
            *self.set_subtitles(
                "Instead, we could let the age vary across all possible values, and measure the average output."
            )
        )
        for value, output in zip(values, outputs):
            new_age = MathTex(*["A = ", f"{value}"], font_size=44, color=BLUE).move_to(
                age
            )
            new_output = Text(output, font_size=35).next_to(out_arrow, direction=RIGHT)
            self.play(
                Transform(age, new_age),
                Transform(y_new, new_output),
                run_time=self.default_run_time,
            )
        self.play(*self.set_subtitles("This is still imperfect, but better."))

        self.next_slide()
        self.play(
            *self.set_subtitles(
                r"This is the second ingredient of an explanation: the way of \textbf{removing variables.}"
            )
        )

        self.next_slide()

        orig_age = MathTex(f"A = 45", font_size=44).move_to(age)
        orig_out = Text("Risk = 60%", font_size=35).next_to(out_arrow, direction=RIGHT)
        self.play(
            Transform(age, orig_age),
            Transform(y_new, orig_out),
            *self.set_subtitles(
                "We can define our method by removing each variable individually. The resulting change in output is then the attribution value."
            ),
            run_time=self.default_run_time,
        )

        self.next_slide()

        new_f = Tex(
            r"""
        IF $B > 130$ OR $C > 100$\\
        THEN Risk = 60\%\\
        ELSE Risk = 10\%
        """,
            font_size=40,
        )

        self.play(
            Transform(f, new_f),
            *self.set_subtitles(
                "Now assume our model is actually a very simple OR function."
            ),
            run_time=self.default_run_time,
        )

        self.next_slide()

        bp_values = [120, 110, 140, 130, "..."]
        c_values = [100, 105, 115, 90, "..."]

        bp = variables_new[3]
        c = variables_new[4]
        self.play(
            *self.set_subtitles(
                "In this case, removing any individual variable never changes the output."
            )
        )
        for value in bp_values:
            new_bp = MathTex(*["B =", f"{value}"], font_size=44, color=BLUE).move_to(bp)
            self.play(
                Transform(bp, new_bp, run_time=self.default_run_time),
            )
        orig_bp = MathTex(*["B =", "135"], font_size=44).move_to(bp)
        self.play(Transform(bp, orig_bp, run_time=self.default_run_time))

        self.next_slide()
        for value in c_values:
            new_c = MathTex(*["C =", f"{value}"], font_size=44, color=BLUE).move_to(c)
            self.play(
                Transform(c, new_c, run_time=self.default_run_time),
            )
        orig_c = MathTex(*["C =", "110"], font_size=44).move_to(c)
        self.play(Transform(c, orig_c, run_time=self.default_run_time))
        self.play(
            *self.set_subtitles(
                "The method would basically say that none of the variables was important!"
            )
        )

        self.next_slide()
        self.play(
            *self.set_subtitles(
                "The problem is that the method never looks at multiple variables at the same time."
            )
        )

        self.next_slide()
        self.play(
            *self.set_subtitles(
                r"We could solve this by removing each \textit{subset} of variables, but then we have more values than variables."
            )
        )

        self.next_slide()
        self.play(
            *self.set_subtitles(
                r"We need to somehow summarize these values into a single score per variable. This is the third ingredient of the method: the way of \textbf{aggregating effects.}"
            )
        )

        self.cleanup_slide()

    def construct_chapter3_2(self):
        self.next_slide()
        title = Tex("Three ingredients:", font_size=60).shift(2 * UP)
        p = Tex(
            r"\item Target: what to explain\vspace{-0.5em}",
            r"\item Removal: how to remove variables\vspace{-0.5em}",
            r"\item Aggregation: how to summarize effects\vspace{-0.5em}",
            tex_environment="enumerate", font_size=50,
            ).to_edge(LEFT, buff=1)

        self.play(
            Write(title),
            *self.set_subtitles(
                "In my PhD, I showed that a large selection of attribution methods can be summarized using these three ingredients."
            ),
        )
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
        self.play(
            Write(eqn),
            *self.set_subtitles(
                "Formally, I showed that many methods can be expressed using this formula."
            ),
        )

        self.next_slide()

        lhs, rhs = eqn
        self.play(FadeOut(rhs))
        orig_point = lhs.get_center()
        self.play(
            lhs.animate.center(),
            *self.set_subtitles(
                r"The left-hand side is a function: it takes the model $f$, the variable $i$ and the input point $\mathbf{x}$ as arguments."
            ),
            run_time=self.default_run_time,
        )

        self.next_slide()

        rhss = [
            r"$= f(\mathbf{x})$",
            r"$= P_{S}(f)(\mathbf{x})$",
            r"$= \Phi(P_{S}(f))(\mathbf{x})$",
            r"$= \alpha_{S}\Phi(P_{S}(f))(\mathbf{x})$",
            r"$= \sum_{S \subseteq [d]} \alpha_{S}\Phi(P_{S}(f))(\mathbf{x})$",
        ]

        subs = [
            r"On the right-hand side, we first simply have the model $f$, evaluated at $\mathbf{x}$.",
            r"The operator $P_S$ produces a new function that no longer depends on the subset $S$: it removes the variables in $S$.",
            r"The operator $\Phi$ encodes the specific aspect of the model that we want to explain: the target.",
            r"This value gets computed for each subset $S$ and multiplied with a constant $\alpha_S$.",
            r"Finally, we sum all of these terms for all of the subsets of variables.",
        ]

        rhs1 = Tex(rhss[0], font_size=70).move_to(rhs.get_center())
        self.play(lhs.animate.move_to(orig_point))
        self.play(
            FadeIn(rhs1), *self.set_subtitles(subs[0]), run_time=self.default_run_time
        )

        for sub, rhsi in zip(subs[1:], rhss[1:]):
            self.next_slide()
            rhs_tex = Tex(rhsi, font_size=70).move_to(rhs.get_center())
            self.play(
                Transform(rhs1, rhs_tex),
                *self.set_subtitles(sub),
                run_time=self.default_run_time,
            )
        self.cleanup_slide()

    def construct_chapter3_4(self):
        self.next_slide()
        self.play(
            *self.set_subtitles(
                "We can describe the attribution method that we just designed using this formula as well."
            )
        )
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

        self.play(
            Write(full_eqn),
            *self.set_subtitles(
                "I formally showed that this equation describes several existing attribution methods, even though they were all originally designed in radically different ways."
            ),
            run_time=self.default_run_time,
        )
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
        self.play(
            Write(children, run_time=self.long_run_time),
            *self.set_subtitles(
                r"An important concept that I haven't touched upon is \textit{interaction.} Assume there are 2 children: Child A and Child B."
            ),
        )

        self.next_slide()

        eqns = VGroup(
            *[
                MathTex(r"b(\{\}) = 0"),
                MathTex(r"b(\{A\}) = 25"),
                MathTex(r"b(\{B\}) = 20"),
                MathTex(r"b(\{A,B\}) = 5"),
                MathTex(r"b(\{A,B\})", r"<", r"b(\{A\})", "+", r"b(\{B\})"),
                MathTex(r"i(\{A,B\})", "=", "b(\{A,B\})", "-", "(", "b(\{A\})", "+", "b(\{B\})", ")"),
            ]
        ).arrange(DOWN)

        self.play(
            children.animate.to_corner(UL).shift(LEFT).scale(0.5),
            *self.set_subtitles(
                "Assume also that we have a room full of Lego bricks that need to be cleaned up."
            ),
            run_time=self.default_run_time,
        )

        subs = [
            "If no children clean up any bricks, then 0 bricks will be cleaned up.",
            "If Child A cleans up bricks on their own, then 25 bricks are cleaned up after around 5 minutes.",
            "Similarly, child B cleans up 20 bricks in about the same time.",
            "However, when the two children are allowed to clean up bricks together, the number drops drastically: together, they only clean up 5 bricks.",
            "What we see is that the number of bricks Child A and Child B clean up together is lower than the sum of the numbers they can clean up separately.",
            r"The difference between these numbers is the \textit{interaction effect.}",
        ]

        for eqn, sub in zip(eqns[:-1], subs[:-1]):
            self.next_slide()
            self.play(
                Write(eqn, run_time=self.default_run_time), *self.set_subtitles(sub)
            )

        self.next_slide()
        self.play(
            Transform(
                eqns[-2],
                MathTex(r"5","<", "25", "+", "20").move_to(eqns[-2]),
                run_time=self.default_run_time,
            )
        )

        self.next_slide()
        self.play(
            Write(eqns[-1], run_time=self.default_run_time),
            *self.set_subtitles(subs[-1]),
        )

        self.next_slide()
        self.play(
            Transform(
                eqns[-1],
                MathTex(r"i(\{A,B\})", "=", "5", "-", "(", "25", "+", "20", ")").move_to(eqns[-1]),
                run_time=self.default_run_time,
            )
        )

        self.next_slide()
        self.play(
            Transform(eqns[-1], MathTex(r"i(\{A,B\})", "=", "-40").move_to(eqns[-1])),
            run_time=self.default_run_time,
        )

        self.cleanup_slide()

    def construct_chapter4_2(self):
        equations = MathTex(
            r"f(\mathbf{x}) &= 2x_1 + 4x_2 - 3x_1x_2 + x_3 + 5\\", # 0
            #
            r"P_{\{1,2,3\}}(f)(\mathbf{x})", # 1
            r"&= f(0,0,0)\\", # 2
            r"&= 2*0 + 4*0 - 3*0*0 + 0 + 5\\", # 3
            r"&= 5\\", # 4
            #
            r"P_{\{2,3\}}(f)(\mathbf{x})", # 5
            r"&= f(x_1,0,0) \\", # 6
            r"&= 2x_1 + 4*0 - 3*x_1*0 + 0 + 5 \\", # 7
            r"&= 2x_1 + 5\\", # 8
            #
            r"e_1(x_1)", # 9
            r"&=", # 10
            r"P_{\{2,3\}}(f)(\mathbf{x})", "-", r"P_{\{1,2,3\}}(f)(\mathbf{x}) \\", # 11 - 13
            r"&= 2x_1\\", # 14
            #
            r"e_2(x_2)", r"&= 4x_2\\", # 15, 16
            #
            r"e_3(x_3) &= x_3\\", # 17
            #
            r"e_{1,2}(x_1,x_2)", # 18
            r"&=", r"P_{\{3\}}(f)(\mathbf{x})", "-", r"P_{\{1,2,3\}}(f)(\mathbf{x})\\", # 19 - 22
            r"&= (2x_1 + 4x_2 - 3x_1x_2 + 0 + 5) - 5\\", # 23
            r"&= 2x_1 + 4x_2 - 3x_1x_2\\", # 24
            #
            r"i_{1,2}(x_1,x_2)", # 25
            r"&=",r"e_{1,2}(x_1,x_2)",r"-(",r"e_1(x_1)",r"+",r"e_2(x_2)",r")\\", # 26 - 32
            r"&= (2x_1 + 4x_2 - 3x_1x_2) - (2x_1 + 4x_2)\\", # 33
            r"&= -3x_1x_2", # 34
        ).shift(4 * DOWN)
        equations[4].color = RED
        equations[8].color = BLUE
        equations[11].color = BLUE
        equations[14].color = GREEN
        equations[13].color = RED
        equations[16].color = PURPLE
        equations[24].color = ORANGE
        equations[22].color = RED
        equations[27].color = ORANGE
        equations[29].color = GREEN
        equations[31].color = PURPLE

        subs = {
            equations[
                1
            ]: "Assume we remove variables by setting them to 0. What happens when we remove all variables?",
            equations[4]: "We get 5.",
            equations[5]: "Now what if we only know variable $x_1$?",
            equations[
                6
            ]: r"We remove all \textit{other} variables, i.e. $x_2$ and $x_3$, by setting them to zero.",
            equations[
                9
            ]: r"The difference between these two functions can be viewed as the \textit{effect} of knowing the value of $x_1$. This is the \textit{direct effect} of $x_1$.",
            equations[15]: "We can do the same for $x_2$ and $x_3$.",
            equations[
                18
            ]: r"Similarly, we can define the \textit{total effect} of $x_1$ and $x_2$ together by removing $x_3$.",
            equations[
                25
            ]: r"The \textit{interaction effect} between $x_1$ and $x_2$ is now the difference between their total effect and the sum of their individual direct effects.",
        }

        self.next_slide()
        self.play(
            Write(equations[0], run_time=self.default_run_time),
            *self.set_subtitles(
                "To see how this is related to attributions, we can look at an example function."
            ),
        )

        for eqn in equations[1:5]:
            self.next_slide()
            if eqn in subs:
                self.play(
                    Write(eqn, run_time=self.default_run_time),
                    *self.set_subtitles(subs[eqn]),
                )
            else:
                self.play(Write(eqn, run_time=self.default_run_time))

        self.next_slide()
        self.play(FadeOut(eqn, run_time=self.default_run_time) for eqn in equations[2:4])
        self.play(
            equations[4].animate.move_to(equations[2]).align_to(equations[2], LEFT),
            run_time=self.default_run_time,
        )

        for eqn in equations[5:9]:
            self.next_slide()
            if eqn in subs:
                self.play(
                    Write(eqn.shift(1.5 * UP), run_time=self.default_run_time),
                    *self.set_subtitles(subs[eqn]),
                )
            else:
                self.play(Write(eqn.shift(1.5 * UP), run_time=self.default_run_time))

        self.next_slide()
        self.play(FadeOut(eqn, run_time=self.default_run_time) for eqn in equations[6:8])
        self.play(
            equations[8].animate.move_to(equations[6]).align_to(equations[6], LEFT),
            run_time=self.default_run_time,
        )

        self.next_slide()
        self.play(
            Write(equations[9].shift(3 * UP), run_time=self.default_run_time),
            Write(equations[10].shift(3 * UP), run_time=self.default_run_time),
            *self.set_subtitles(subs[equations[9]]),
        )

        self.next_slide()
        self.play(*[Write(equations[i].shift(3 * UP)) for i in range(11,14)], run_time=self.default_run_time)

        self.next_slide()
        self.play(Write(equations[14].shift(3 * UP)), run_time = self.default_run_time)

        self.next_slide()
        self.play(*[FadeOut(equations[i], run_time=self.default_run_time) for i in range(10, 14)])
        self.play(
            equations[14].animate.move_to(equations[10]).align_to(equations[10], LEFT),
            run_time=self.default_run_time,
        )

        self.next_slide()
        self.play(
            Write(
                equations[15].next_to(equations[14], RIGHT).shift(0.5 * RIGHT),
                run_time=self.default_run_time,
            ),
            Write(equations[16].next_to(equations[15], RIGHT), run_time=self.default_run_time),
            *self.set_subtitles(subs[equations[15]]),
        )
        self.play(Write(equations[17].next_to(equations[16], RIGHT).shift(0.5 * RIGHT)))

        self.next_slide()
        self.play(
            Write(equations[18].shift(5.25 * UP), run_time=self.default_run_time),
            *self.set_subtitles(subs[equations[18]]),
        )
        self.next_slide()
        self.play(
            *[Write(equations[i].shift(5.25 * UP), run_time=self.default_run_time) for i in range(19, 23)]
        )

        for eqn in equations[23:25]:
            self.next_slide()
            if eqn in subs:
                self.play(
                    Write(eqn.shift(5.25 * UP), run_time=self.default_run_time),
                    *self.set_subtitles(subs[eqn]),
                )
            else:
                self.play(Write(eqn.shift(5.25 * UP), run_time=self.default_run_time))

        self.next_slide()
        self.play(FadeOut(eqn, run_time=self.default_run_time) for eqn in equations[19:24])
        self.play(
            equations[24].animate.move_to(equations[19]).align_to(equations[19], LEFT),
            run_time=self.default_run_time,
        )

        self.next_slide()
        self.play(
            Write(equations[25].shift(6.75 * UP), run_time=self.default_run_time),
            *self.set_subtitles(subs[equations[25]]),
        )
        self.next_slide()
        self.play(*[Write(eqn.shift(6.75 * UP), run_time=self.default_run_time) for eqn in equations[26:33]])

        self.next_slide()
        self.play(*[Write(equations[33].shift(6.75 * UP), run_time=self.default_run_time)])

        self.next_slide()
        self.play(*[Write(equations[34].shift(6.75 * UP), run_time=self.default_run_time)])

        self.next_slide()
        self.play(FadeOut(eqn, run_time=self.default_run_time) for eqn in equations[26:34])
        self.play(
            equations[34].animate.move_to(equations[26]).align_to(equations[26], LEFT),
            run_time=self.default_run_time,
        )

        self.cleanup_slide()

    def construct_chapter4_3(self):
        self.next_slide()
        equations = MathTex(
            r"f(\mathbf{x}) &=",
            "2x_1",
            " + 4x_2",
            "- 3x_1x_2",
            "+ x_3",
            r" + 5\\",
            r"&= ",
            "e_1(x_1)",
            "+ e_2(x_2)",
            " + i_{1,2}(x_1,x_2)",
            " + e_3(x_3)",
            " + c",
        )
        equations[1].color = BLUE
        equations[7].color = BLUE
        equations[2].color = RED
        equations[8].color = RED
        equations[3].color = YELLOW
        equations[9].color = YELLOW
        equations[4].color = GREEN
        equations[10].color = GREEN
        self.play(
            Write(equations),
            *self.set_subtitles(
                r"What we've just created is an \textit{additive decomposition} of the function $f$: we split the function into smaller parts (decomposition) that, when summed together (additive), give us the original function $f$."
            ),
            run_time=self.long_run_time,
        )
        self.next_slide()
        self.play(
            *self.set_subtitles(
                "The specific decomposition depends entirely on how we \textit{remove} input variables."
            )
        )
        self.cleanup_slide()

    def construct_chapter4_4(self):
        self.next_slide()
        eqn = (
            MathTex(
                r"""
        f &= e_\emptyset\\
        &+ e_1 + e_2 + e_3 \\
        &+ i_{1,2} + i_{1,3} + i_{2,3} \\
        &+ i_{1,2,3} \\
        """
            )
            .to_edge(LEFT)
            .shift(RIGHT)
        )
        self.play(
            Write(eqn),
            *self.set_subtitles(
                r"It turns out that \textit{any} function $f$ can be decomposed in this way."
            ),
            run_time=self.long_run_time,
        )

        graph = make_hasse(
            vertex_config={"radius": 0.3},
            edge_config={"stroke_width": 10},
            labels={
                0: MathTex(r"c", color=BLACK),
                1: MathTex(r"e_1", color=BLACK),
                2: MathTex(r"e_2", color=BLACK),
                3: MathTex(r"i_{1,2}", color=BLACK, font_size=35),
                4: MathTex(r"e_3", color=BLACK),
                5: MathTex(r"i_{1,3}", color=BLACK, font_size=35),
                6: MathTex(r"i_{2,3}", color=BLACK, font_size=35),
                7: MathTex(r"i_{1,2,3}", color=BLACK, font_size=25),
            },
        ).to_corner(DR, buff=0.1)
        self.next_slide()
        self.play(
            Write(graph),
            *self.set_subtitles(
                "This diagram shows the decomposition: each node corresponds to a subset of the variables, and therefore to a component of the decomposition."
            ),
            run_time=self.long_run_time * 2,
        )

        self.cleanup_slide()

    def construct_chapter4_5(self):
        self.next_slide()
        title = Text("Three ingredients:", font_size=50).shift(2 * UP)
        p = Tex(
            r"\item Target: what to explain\vspace{-0.5em}",
            r"\item Removal: how to remove variables\vspace{-0.5em}",
            r"\item Aggregation: how to summarize effects\vspace{-0.5em}",
            tex_environment="enumerate", font_size=50).to_edge(LEFT, buff=1)

        self.play(Write(title))
        self.play(
            Write(p, run_time=2 * self.long_run_time),
            *self.set_subtitles(
                "In my PhD, I showed the ways of removing variables correspond exactly to the ways of decomposing functions."
            ),
        )

        self.next_slide()
        decomp = Tex(
            r"\item Target: what to explain\vspace{-0.5em}",
            r"\item Decomposition: How to decompose the function\vspace{-0.5em}",
            r"\item Aggregation: how to summarize effects\vspace{-0.5em}",
            tex_environment="enumerate", font_size=50).to_edge(LEFT, buff=1)
        self.play(
            Transform(p, decomp, run_time=self.default_run_time),
            *self.set_subtitles(
                "Therefore, we can replace the second ingredient by a choice of decomposition."
            ),
        )

        self.cleanup_slide()

    def construct_chapter5_1(self):
        self.next_slide()
        title = Text("SHAP:", font_size=50).shift(2 * UP)

        decomp = Tex(
            r"\item Target: output of $f$ at $\mathbf{x}$\vspace{-0.5em}",
            r"\item Removal: average output over dataset\vspace{-0.5em}",
            r"\item Aggregation: $\dots$\vspace{-0.5em}",
            tex_environment="enumerate", font_size=50).to_edge(LEFT, buff=1)

        self.play(Write(title, run_time=self.default_run_time),
                  *self.set_subtitles("SHAP is a very popular but very complicated existing attribution method. However, we can also study it using the three ingredients we've seen."))

        for sentence in decomp:
            self.next_slide()
            self.play(Write(sentence, run_time=self.long_run_time))

        self.cleanup_slide()

    def construct_chapter5_2(self):
        self.next_slide()
        self.play(*self.set_subtitles("SHAP removes all possible subsets of variables and aggregates the results in a complicated way."))
        self.next_slide()
        graph = make_hasse(
            vertex_config={"radius": 0.3},
            edge_config={"stroke_width": 10},
            labels={
                0: MathTex(r"c", color=BLACK),
                1: MathTex(r"e_1", color=BLACK),
                2: MathTex(r"e_2", color=BLACK),
                3: MathTex(r"i_{1,2}", color=BLACK, font_size=35),
                4: MathTex(r"e_3", color=BLACK),
                5: MathTex(r"i_{1,3}", color=BLACK, font_size=35),
                6: MathTex(r"i_{2,3}", color=BLACK, font_size=35),
                7: MathTex(r"i_{1,2,3}", color=BLACK, font_size=25),
            },
        ).to_corner(DR, buff=0.1)
        self.play(Write(graph, run_time=self.long_run_time),
                  *self.set_subtitles("However, it becomes much simpler if we instead look at the additive decomposition that corresponds to SHAP."))

        eqn = MathTex(
            r"\mathrm{SHAP}(f,1,\mathbf{x})",
            r"&= e_1(\mathbf{x})\\",
            r"&+ (i_{1,2}(\mathbf{x}) + i_{1,3}(\mathbf{x}))/2\\",
            r"&+ i_{1,2,3}(\mathbf{x})/3",
        ).to_edge(LEFT)

        self.next_slide()
        self.play(Write(eqn[0], run_time=self.default_run_time),
                  *self.set_subtitles("If we want to compute the SHAP value for variable 1 at $\mathbf{x}$ ..."))

        self.next_slide()
        self.play(Write(eqn[1], run_time=self.default_run_time), Circumscribe(graph.vertices[1],run_time=self.default_run_time),
                  *self.set_subtitles("... we take the direct effect of variable 1 at $\mathbf{x}$ ... "))

        self.next_slide()
        self.play(
            Write(eqn[2]),
            Circumscribe(graph.vertices[3]),
            Circumscribe(graph.vertices[5]),
            *self.set_subtitles("... plus the interaction effects between variable 1 and one other variable, divided by two ..."),
            run_time=self.default_run_time
        )

        self.next_slide()
        self.play(Write(eqn[3]), Circumscribe(graph.vertices[7]),
                  *self.set_subtitles("... plus the interaction effect between all three variables, divided by three."),
                  run_time=self.default_run_time)

        self.next_slide()

        self.play(*self.set_subtitles("In general, SHAP divides the interaction effect between a set of variables equally among those variables."))

        self.cleanup_slide()

    def construct_chapter5_3(self):
        self.next_slide()
        self.play(*self.set_subtitles("SHAP technically removes variables by averaging over the entire dataset."))

        self.next_slide()
        self.play(*self.set_subtitles("Typically we instead choose about 100 samples, and assume that's close enough."))

        self.next_slide()
        self.play(*self.set_subtitles("SHAP removes all possible subsets of variables. This number grows exponentially as the number of variables increases."))

        self.next_slide()
        self.play(*self.set_subtitles("Again, we typically choose e.g. about 100 subsets and assume that's close enough."))

        self.next_slide()
        self.play(*self.set_subtitles("However, this still requires us to compute the output of the model $100 * 100 = 10.000$ times."))

        self.next_slide()
        graph = make_hasse(
            vertex_config={"radius": 0.3},
            edge_config={"stroke_width": 10},
            labels={
                0: MathTex(r"c", color=BLACK),
                1: MathTex(r"e_1", color=BLACK),
                2: MathTex(r"e_2", color=BLACK),
                3: MathTex(r"i_{1,2}", color=BLACK, font_size=35),
                4: MathTex(r"e_3", color=BLACK),
                5: MathTex(r"i_{1,3}", color=BLACK, font_size=35),
                6: MathTex(r"i_{2,3}", color=BLACK, font_size=35),
                7: MathTex(r"i_{1,2,3}", color=BLACK, font_size=25),
            },
        ).to_edge(DOWN, buff=0.1)
        self.play(Write(graph, run_time=self.long_run_time),
                  *self.set_subtitles("If we would have a decomposition of the function, we would only have to compute the output of each component once."))

        self.next_slide()
        self.play(*self.set_subtitles("Typically, most of the effects are in interactions between small subsets of variables, so we can save more computation by only computing the components up to some subset size $k$."))

        self.next_slide()
        graph_trained = make_hasse(
            vertex_config={"radius": 0.3},
            edge_config={"stroke_width": 10},
            labels={
                0: MathTex(r"f_\emptyset", color=BLACK),
                1: MathTex(r"f_1", color=BLACK),
                2: MathTex(r"f_2", color=BLACK),
                3: MathTex(r"f_{1,2}", color=BLACK, font_size=35),
                4: MathTex(r"f_3", color=BLACK),
                5: MathTex(r"f_{1,3}", color=BLACK, font_size=35),
                6: MathTex(r"f_{2,3}", color=BLACK, font_size=35),
                7: MathTex(r"f_{1,2,3}", color=BLACK, font_size=25),
            },
        ).to_edge(DOWN, buff=0.1)
        self.play(Transform(graph, graph_trained, run_time=self.default_run_time),
                  *self.set_subtitles("This is what PDD-SHAP does: train a model to approximate each component, and use those models to quickly approximate the SHAP value."))
        self.cleanup_slide()

    def construct_chapter5_4(self):
        self.next_slide()
        img = ImageMobject("r2_scores.png").scale(0.4).to_edge(DOWN)
        self.play(FadeIn(img, run_time=self.default_run_time),
                  *self.set_subtitles("This figure shows the accuracy of the approximation in relation to the maximal subset size."))

        self.next_slide()
        self.play(*self.set_subtitles("We see that for some datasets, we are able to approximate SHAP values quite well with only interactions between 2 or 3 variables. For others, we need 4 or more variables."))

        self.next_slide()
        img2 = ImageMobject("inference_time.png").scale(0.4).to_edge(DOWN)
        self.play(FadeOut(img), FadeIn(img2),
                  *self.set_subtitles("This figure shows the computation time required to compute SHAP values using PDD-SHAP vs. classical methods. Note that the Y axis is logarithmic."),
                  run_time=self.default_run_time)
        self.cleanup_slide()

    def construct_conclusion(self):
        self.next_slide()
        # TODO: Use Tex enumerate here
        benchmark1 = Text("1. Benchmark:", font_size=35, color=BLUE).shift(4 * LEFT)
        benchmark2 = (
            Text("there is no universal measure of quality", font_size=35)
            .next_to(benchmark1, RIGHT)
            .align_to(benchmark1, UP)
        )

        framework1 = (
            Text("2. Framework: ", font_size=35, color=BLUE)
            .next_to(benchmark1, DOWN)
            .align_to(benchmark1, LEFT)
        )
        framework2 = (
            Text("unifying removal-based attribution methods", font_size=35)
            .align_to(benchmark2, LEFT)
            .align_to(framework1, UP)
        )
        framework3 = (
            Text("using functional decomposition", font_size=35)
            .next_to(framework2, DOWN)
            .align_to(benchmark2, LEFT)
        )

        pddshap1 = (
            Text("3. PDD-SHAP:", font_size=35, color=BLUE)
            .next_to(framework3, DOWN)
            .align_to(framework1, LEFT)
        )
        pddshap2 = (
            Text("fast approximation algorithm", font_size=35)
            .align_to(benchmark2, LEFT)
            .align_to(pddshap1, UP)
        )
        pddshap3 = (
            Text("for existing attributions", font_size=35)
            .next_to(pddshap2, DOWN)
            .align_to(benchmark2, LEFT)
        )

        all = VGroup(benchmark1, benchmark2, framework1, framework2, framework3, pddshap1, pddshap2, pddshap3).center()

        self.play(Write(all[0]))
        self.play(Write(all[1]))
        self.next_slide()
        self.play(Write(all[2]))
        self.play(Write(all[3]))
        self.play(Write(all[4]))
        self.next_slide()
        self.play(Write(all[5]))
        self.play(Write(all[6]))
        self.play(Write(all[7]))

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

        self.construct_toc_slide(chapter=5)
        self.construct_chapter5_1()
        self.construct_chapter5_2()
        self.construct_chapter5_3()
        self.construct_chapter5_4()

        self.construct_toc_slide(chapter=6)
        self.construct_conclusion()

        # TODO: check runtime of all self.play() calls
        # TODO: replace parameter searching animation using DecimalNumber
