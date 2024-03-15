from os import wait

from manim import *
from sklearn.metrics import roc_curve
from utils.dummy_data import make


class AUCVisualization(Scene):
    def construct(self):
        axes = get_axes(diff=0.1)

        # 適当なデータセット
        good_label, good_prob = make(40, sensitivity=0.75)
        fpr, tpr, thresholds = roc_curve(good_label, good_prob)

        # AUCカーブを直線で結ぶ
        roc_curve_points = [axes.c2p(fpr, tpr) for fpr, tpr in zip(fpr, tpr)]
        area_polygon = Polygon(
            *roc_curve_points, axes.c2p(1, 0), fill_opacity=0.5, color=WHITE, fill_color=GREEN
        )

        # AUCを見やすくするための枠線などを描く
        graphs = VGroup()
        graphs += axes.get_horizontal_line(axes.c2p(1, 1), color=BLUE)
        graphs += axes.get_vertical_line(axes.c2p(1, 1), color=BLUE)
        dashed_graph = DashedVMobject(axes.plot(lambda x: x, x_range=[0, 1], color=BLUE))

        # 描画
        self.play(Create(axes), run_time=2)
        self.play(Create(dashed_graph), run_time=2)
        self.add(graphs)
        self.play(DrawBorderThenFill(area_polygon), run_time=3)


def get_axes(
    diff: float = 0.05,
):
    from manim import BLUE, Axes

    axes = Axes(
        x_range=[0 - diff, 1 + diff, diff],
        y_range=[0 - diff, 1 + diff, diff],
        x_length=5,
        y_length=5,
        axis_config={
            "color": BLUE,
            "numbers_to_include": [0, 1],
        },
        x_axis_config={"include_tip": False},
        y_axis_config={"include_tip": False},
    )

    return axes
