from manim import *


class DifferentRotations(Scene):

    def construct(self):
        left_square = Square(color=BLUE, fill_opacity=0.7).shift(2 * LEFT)
        right_square = Square(color=GREEN, fill_opacity=0.7).shift(2 * RIGHT)

        # manimは最初と最後の状態を補間するように機能する
        # animate.rotateは180度回転で一致するので縮むように見える
        # 明確に回転させたいときはRotateを使う必要がある
        self.play(
            left_square.animate.rotate(PI),
            Rotate(right_square, angle=PI),
            run_time=2,
        )
        self.wait()
