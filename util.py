from manim import *


def paragraph(*strs, alignment=LEFT, direction=DOWN, **kwargs):
    texts = VGroup(*[Text(s, **kwargs) for s in strs]).arrange(direction)

    if len(strs) > 1:
        for text in texts[1:]:
            text.align_to(texts[0], direction=alignment)

    return texts

TITLE_FONT_SIZE = 32
CONTENT_FONT_SIZE = 0.8 * TITLE_FONT_SIZE