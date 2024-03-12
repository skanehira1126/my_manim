from manim import *


class CreateCircle(Scene):
    def construct(self):
        """
        描画関数はここに書くよ
        """

        # create a circle
        circle = Circle()
        circle.set_fill(PINK, opacity=0.5)

        # 描画する
        self.play(Create(circle))
