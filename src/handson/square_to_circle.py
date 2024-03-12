from manim import *


class SquareToCircle(Scene):

    def construct(self):

        circle = Circle()
        circle.set_fill(PINK, opacity=0.5)

        square = Square()
        square.rotate(PI / 4)  # 回転

        self.play(Create(square))
        self.play(Transform(square, circle))  # 四角から円に変換する
        self.play(FadeOut(circle))  # じわりと消す
