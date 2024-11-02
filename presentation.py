from manim import *
from manim_slides import Slide
from util import paragraph, TITLE_FONT_SIZE, CONTENT_FONT_SIZE

config.background_color = "#262626ff"

class Presentation(Slide):
    def construct_intro(self):
        title1 = Text(
            "Unifying Attribution-Based Explanation Methods", color=WHITE, font_size=TITLE_FONT_SIZE,
        )
        title2 = Text(
            "in Machine Learning", color=WHITE, font_size=TITLE_FONT_SIZE,
        ).next_to(title1, DOWN)
        author_date = (
            Text("Arne Gevaert - November 7th 2024", color=WHITE, font_size=CONTENT_FONT_SIZE)
            .next_to(title2, DOWN)
        )

        self.next_slide(notes="# Welcome!")
        self.play(FadeIn(title1), FadeIn(title2))
        self.play(FadeIn(author_date))

        self.slide_title = Text(
            "Contents", color=BLACK, font_size=TITLE_FONT_SIZE
        ).to_corner(UL)

        contents = paragraph(
            f"1. Introduction",
            f"2. Feature Attribution Benchmark",
            f"3. Removal-Based Attribution Methods",
            f"4. Functional Decomposition",
            f"5. PDD-SHAP",
            f"6. Conclusion",
            color=WHITE,
            font_size=CONTENT_FONT_SIZE,
        ).align_to(self.slide_title, LEFT)

        self.next_slide(notes="Table of contents")
        self.wipe(self.mobjects_without_canvas, [*self.canvas_mobjects, contents])
    
    def construct(self):
        self.construct_intro()