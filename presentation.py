from manim import *
from manim_slides import Slide, ThreeDSlide
from sklearn.metrics import root_mean_squared_error

from util import paragraph, example_function_1, linreg_univariate, linreg_multivariate, parabolic_reg, nn_reg

config.background_color = "#262626ff"



class Presentation(ThreeDSlide):
    def construct_toc_slide(self, chapter):
        contents_title = Text("Contents", color=BLACK, font_size=32).to_corner(UL)
        contents = paragraph(
            f"1. Introduction",
            f"2. Feature Attribution Benchmark",
            f"3. Removal-Based Attribution Methods",
            f"4. Functional Decomposition",
            f"5. PDD-SHAP",
            f"6. Conclusion",
            color=WHITE,
            font_size=24,
        ).align_to(contents_title, LEFT)

        self.play(FadeIn(contents))
        self.next_slide()
        cur = contents[chapter - 1]

        cur.generate_target()
        cur.target.set_fill(color=BLUE)
        cur.target.scale(1.2)

        self.play(
            FadeOut(contents[: chapter - 1]), FadeOut(contents[chapter:]), MoveToTarget(cur)
        )

        self.next_slide()
        self.play(FadeOut(contents[chapter - 1]))
        self.next_slide()

    def construct_titleslide(self):
        title1 = Text(
            "Unifying Attribution-Based Explanation Methods",
            color=WHITE,
            font_size=32,
        )
        title2 = Text(
            "in Machine Learning",
            color=WHITE,
            font_size=32,
        ).next_to(title1, DOWN)
        author_date = Text(
            "Arne Gevaert - November 7th 2024", color=WHITE, font_size=24
        ).next_to(title2, DOWN)

        self.next_slide()
        self.play(FadeIn(title1), FadeIn(title2))
        self.play(FadeIn(author_date))
        self.next_slide()
        self.play(FadeOut(title1), FadeOut(title2), FadeOut(author_date))

    def construct_chapter1_1(self):
        # GENERAL FUNCTION
        rect = Rectangle(width=5, height=3)
        f = MathTex("f", font_size=100)
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

        self.play(*[FadeOut(obj) for obj in self.mobjects_without_canvas])

        self.next_slide()

    def construct_chapter1_2(self):
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

        self.next_slide()
        self.play(*[FadeOut(obj) for obj in self.mobjects_without_canvas])
        self.next_slide()

    def construct_chapter1_3(self):
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

        self.next_slide()
        self.play(*[FadeOut(obj) for obj in self.mobjects_without_canvas])
        self.next_slide()

    def construct_chapter1_4(self):
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

        self.next_slide()
        self.play(*[FadeOut(obj) for obj in self.mobjects_without_canvas])
        self.next_slide()

    def construct_chapter1_5(self):
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

        self.next_slide()
        self.play(*[FadeOut(obj) for obj in self.mobjects_without_canvas])
        self.next_slide()

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

        self.next_slide()
        self.play(*[FadeOut(obj) for obj in self.mobjects_without_canvas])
        self.next_slide()

    def construct_chapter1_7(self):
        # ORIGINAL SETUP
        # Draw axes
        axis_config = {"include_ticks": True, "include_numbers": True, }
        plane = NumberPlane(
            x_axis_config=axis_config, 
            y_axis_config=axis_config, 
            background_line_style={
                "stroke_width": 4,
                "stroke_opacity": 0.2
            },
            x_range=(-6, 22, 2),
            x_length=config.frame_width,
            y_range=(-1, 7),
            y_length=config.frame_height)

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
        reg_line = plane.plot(lambda x: lm.predict(np.array(x).reshape(-1,1))[0])

        self.play(
            Write(plane),
            *[Create(dot) for dot in dots],
            Create(reg_line)
        )

        self.next_slide()

        # CHANGE TO PARABOLIC DATA
        def parabola(x):
            return x**2/15 - x + 4
            
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
        para_line = plane.plot(lambda x: lm_para.predict(np.array(x).reshape(-1,1))[0])
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
        nn_line = plane.plot(lambda x: nn.predict(np.array(x).reshape(-1,1))[0])
        self.play(Write(nn_line))

        self.next_slide()
        self.play(
            *[FadeOut(obj) for obj in self.mobjects_without_canvas]
        )

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
                l2_coefs[i], l1_coefs[i], '+' if l1_intercepts[i] > 0 else '', l1_intercepts[i]
            )
            if i % 2 == 0:
                substring = substring + '\\\\ &'
            if l2_coefs[i] > 0:
                nn_string += '+'
            nn_string += substring
        
        formula = MathTex(nn_string).scale(1.5)
        self.play(Write(formula))
        self.next_slide()

        formula.generate_target()
        formula.target.scale(0.5)

        self.play(MoveToTarget(formula, run_time=5))

        self.next_slide()
        self.play(
            *[FadeOut(obj) for obj in self.mobjects_without_canvas]
        )
        self.next_slide()
    
    def construct(self):
        #self.construct_titleslide()
        #self.construct_toc_slide(chapter=1)
        #self.construct_chapter1_1()
        #self.construct_chapter1_2()
        #self.construct_chapter1_3()
        #self.construct_chapter1_4()
        #self.construct_chapter1_5()
        self.construct_chapter1_6()
        self.construct_chapter1_7()
        self.construct_chapter1_8()
        self.construct_toc_slide(chapter=2)


