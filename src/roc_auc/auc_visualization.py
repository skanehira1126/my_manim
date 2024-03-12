from manim import *


class AUCVisualization(Scene):
    def construct(self):
        axes = Axes(
            x_range=[0, 1],
            y_range=[0, 1],
            axis_config={"color": BLUE},
            x_axis_config={"include_tip": False, "numbers_to_exclude": [0, 1]},
            y_axis_config={"include_tip": False, "numbers_to_exclude": [0, 1]},
        )

        # モデル1のROC曲線（ここではダミーデータ）
        roc_curve_1 = axes.plot(lambda x: x**0.5, color=RED)
        auc_1 = axes.get_area(roc_curve_1, x_range=(0, 1), color=RED, opacity=0.2)

        # モデル2のROC曲線（ここではダミーデータ）
        roc_curve_2 = axes.plot(lambda x: x**2, color=GREEN)
        auc_2 = axes.get_area(roc_curve_2, x_range=(0, 1), color=GREEN, opacity=0.2)

        # アニメーション
        self.play(Create(axes), run_time=2)
        self.play(Create(roc_curve_1), FadeIn(auc_1), run_time=2)
        self.wait(1)
        self.play(
            ReplacementTransform(roc_curve_1, roc_curve_2),
            ReplacementTransform(auc_1, auc_2),
            run_time=2,
        )
        self.wait(2)
