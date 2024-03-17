from dataclasses import dataclass
from itertools import product

import numpy as np
from manim import *
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from utils.dummy_data import make


@dataclass
class ManimConfig:
    # PrecisionやRecall, AUCを計算するデータ数
    rows: int
    cols: int

    @property
    def total_points(self):
        return self.cols * self.rows


class DynamicMetricsText:

    def __init__(
        self,
        scene: Scene,
        axes_auc_curve: Axes,
        number_line: NumberLine,
        threshold_line: DashedLine,
        labels,
        probs,
        dot_color,
    ):

        # 軸など基準になるもの
        self.scene = scene
        self.axes_auc_curve = axes_auc_curve
        self.number_line = number_line
        self.threshold_line = threshold_line

        # 表示のためのデータ
        self.labels = labels
        self.probs = probs
        self.fpr, self.tpr, self.threshold = roc_curve(self.labels, self.probs)

        # AUCの描画
        self.dot_color = dot_color
        self.point_coordinate_list = []
        self.point_object_list = []
        self.is_add = False
        self.previous_threshold = None

    def add_dot_on(self):
        self.is_add = True

    def add_dot_off(self):
        self.is_add = False

    def update_metrics(
        self,
        texts: VGroup,
    ):

        threshold: float = self.number_line.p2n(self.threshold_line.get_center())

        tn, fp, fn, tp = confusion_matrix(self.labels, self.probs >= threshold).ravel()

        tpr = float(f"{tp / (tp + fn):.2f}")
        fpr = float(f"{fp / (tn + fp):.2f}")

        new_texts = VGroup(
            Text(f"FPR: {fpr: .2f}", font_size=20).move_to(texts[0].get_center()),
            Text(f"TPR: {tpr: .2f}", font_size=20).move_to(texts[1].get_center()),
        )

        texts.become(new_texts)

        # AUC curveのための点を打つ
        if self.is_add:
            # 閾値があるところのfpr, tpr
            self.add_dot(fpr, tpr)

            # AUC-Curveを描くのに必要なsklearnによる点
            if self.previous_threshold is None:
                pass
            else:
                th_upper = max([threshold, self.previous_threshold])
                th_lower = min([threshold, self.previous_threshold])
                target_points = filter(
                    lambda x: ((th_upper >= x[2]) & (x[2] >= th_lower)),
                    zip(self.fpr, self.tpr, self.threshold),
                )
                for fpr, tpr, _ in target_points:
                    self.add_dot(fpr, tpr)
        self.previous_threshold = threshold

    def add_dot(self, fpr, tpr):
        if [fpr, tpr] not in self.point_coordinate_list:
            dot = Dot(self.axes_auc_curve.c2p(fpr, tpr), color=self.dot_color, fill_opacity=1)
            self.scene.add(dot)

            # 管理用の変数
            self.point_coordinate_list.append([fpr, tpr])
            self.point_object_list.append(dot)


class AUCVisualization(Scene):

    def initialize(self):
        self.manim_config = ManimConfig(
            rows=5,
            cols=5,
        )

    def construct(self):

        # 設定初期化
        self.initialize()

        # 適当なデータセット
        labels, (probs_perfect, probs_good, probs_bad) = make(
            self.manim_config.total_points, sensitivity_list=[1, 0.75, 0], var=1.2
        )

        # 箱を作成
        box = Rectangle(width=3, height=3, fill_color=WHITE, fill_opacity=0.5, stroke_color=WHITE)
        box.move_to(4 * LEFT)
        box_text = Text("Data", color=WHITE, font_size=24).move_to(box.get_top() + 0.5 * UP)
        self.play(FadeIn(box), Create(box_text))

        # 円の間隔とサイズを計算
        spacing_x = box.width / self.manim_config.cols
        spacing_y = box.height / self.manim_config.rows
        circle_radius = min(spacing_x, spacing_y) / 2 * 0.7  # 少し余裕を持たせる

        # 元のデータを作成する
        circles = []
        for (idx_col, idx_row), label in zip(
            product(range(self.manim_config.cols), range(self.manim_config.rows)), labels
        ):
            circle = Circle(
                radius=circle_radius,
                color=RED if label else BLUE,
                fill_color=RED if label else BLUE,
                fill_opacity=0.5,
            )
            circle.move_to(
                box.get_center()
                + np.array(
                    [
                        (idx_col - self.manim_config.cols / 2 + 0.5) * spacing_x,
                        (-idx_row + self.manim_config.rows / 2 - 0.5) * spacing_y,
                        0,
                    ]
                )
            )
            circles.append(circle)
        self.play(*map(Create, circles))
        group_data = VGroup(box, box_text, *circles)

        """
        予測結果の制度によってそれぞれ数直線上に並べる
        """
        sorted_lines_predicted = {}
        for probs, position, model_name in zip(
            [probs_perfect, probs_good, probs_bad],
            [2 * UP, [0, 0, 0], 2 * DOWN],
            ["Perfect", "Good", "Bad"],
        ):
            _number_line = self.move_circle_to_lines(
                labels,
                probs,
                circles,
                position=RIGHT * 3.5 + position,
                model_name=model_name,
            )
            sorted_lines_predicted[model_name] = _number_line

        # 元のデータを消して、数直線上を移動する
        self.play(FadeOut(group_data))
        self.play(
            *[
                _number_line[0].animate.scale(0.8)
                for _number_line in sorted_lines_predicted.values()
            ],
            *[
                _number_line[2:].animate.scale(0.8)
                for _number_line in sorted_lines_predicted.values()
            ],
            *[
                Transform(
                    _number_line[1],
                    Text(model_name, font_size=_number_line[1].font_size).move_to(
                        _number_line[0].get_left()
                    ),
                )
                for model_name, _number_line in sorted_lines_predicted.items()
            ],
        )
        self.play(
            *[
                _number_line.animate.move_to(_number_line.get_center() + LEFT * 6.5)
                for _number_line in sorted_lines_predicted.values()
            ],
        )

        # aucのための軸
        axes_auc = get_axes()
        axes_auc.to_edge(RIGHT)
        # 見やすいように枠線などをかく
        horizontal_line = axes_auc.get_horizontal_line(axes_auc.c2p(1, 1), color=BLUE)
        vertical_line = axes_auc.get_vertical_line(axes_auc.c2p(1, 1), color=BLUE)
        diagonal_line = DashedVMobject(axes_auc.plot(lambda x: x, x_range=[0, 1], color=BLUE))

        # 軸のラベル
        x_label = axes_auc.get_x_axis_label("FPR").scale(0.5)
        y_label = axes_auc.get_y_axis_label("TPR").scale(0.5)
        # まとめたやつ
        axes_auc_group = VGroup(
            axes_auc, horizontal_line, vertical_line, diagonal_line, x_label, y_label
        )

        """
        閾値によってTPR, FPRが変わることを描画
        """
        number_lines = []
        threshold_lines = []
        texts_metrics = []
        dynamic_metrics_list = []
        for model_name, probs in zip(
            ["Perfect", "Good", "Bad"], [probs_perfect, probs_good, probs_bad]
        ):
            # 数直線の取得
            number_line = sorted_lines_predicted[model_name][0]

            # 閾値の作成
            threshold_line = DashedLine(0.7 * UP, 0.7 * DOWN, color=YELLOW).next_to(
                number_line.n2p(0.5), buff=0
            )

            # aucのdot color
            if model_name == "Perfect":
                dot_color = GREEN
            elif model_name == "Good":
                dot_color = YELLOW
            else:
                dot_color = RED

            # テキスト更新のためのクラス
            dynamic_metrics = DynamicMetricsText(
                scene=self,
                axes_auc_curve=axes_auc,
                number_line=number_line,
                threshold_line=threshold_line,
                labels=labels,
                probs=probs,
                dot_color=dot_color,
            )

            fpr_text = Text("FPR: ", font_size=20).move_to(
                number_line.get_right() + RIGHT + 0.5 * UP
            )
            tpr_text = Text("TPR: ", font_size=20).move_to(
                number_line.get_right() + RIGHT + 0.5 * DOWN
            )
            texts = VGroup(fpr_text, tpr_text)

            dynamic_metrics.update_metrics(texts)
            # まとめて処理するために格納
            number_lines.append(number_line)
            threshold_lines.append(threshold_line)
            texts_metrics.append(texts)
            dynamic_metrics_list.append(dynamic_metrics)

        self.play(
            *[Create(_th_line) for _th_line in threshold_lines],
            *[Create(_texts) for _texts in texts_metrics],
        )
        self.wait(1)

        for to, run_time in zip([1, 0, 0.5], [1.5, 3, 1.5]):
            self.play(
                *[
                    _th_line.animate.next_to(_num_line.n2p(to), buff=0)
                    for _th_line, _num_line in zip(threshold_lines, number_lines)
                ],
                *[
                    UpdateFromFunc(_texts, _dynamic_metrics.update_metrics)
                    for _texts, _dynamic_metrics in zip(texts_metrics, dynamic_metrics_list)
                ],
                run_time=run_time,
            )
        self.wait(0.5)

        """
        Perfectから順番に閾値を動かして点を描く
        """
        self.play(
            FadeIn(axes_auc),
            FadeIn(x_label),
            FadeIn(y_label),
        )
        self.play(
            FadeIn(horizontal_line),
            FadeIn(vertical_line),
            FadeIn(diagonal_line),
        )

        dynamic_metrics_list[0].add_dot_on()
        dynamic_metrics_list[1].add_dot_on()
        dynamic_metrics_list[2].add_dot_on()

        for _th_line, _num_line, _texts, _dynamic_metrics in zip(
            threshold_lines, number_lines, texts_metrics, dynamic_metrics_list
        ):
            self.play(
                Indicate(_num_line, color=_dynamic_metrics.dot_color),
                Indicate(_th_line, color=_dynamic_metrics.dot_color),
                run_time=2,
            )

            for to in [0, 1, 0.5]:
                self.play(
                    _th_line.animate.next_to(_num_line.n2p(to), buff=0),
                    UpdateFromFunc(_texts, _dynamic_metrics.update_metrics),
                    run_time=2.5,
                )

            self.play(FadeOut(_th_line))

        self.play(*[FadeOut(_texts) for _texts in texts_metrics])

        """
        FPR, TPRを使ってAUCの領域を可視化する
        """
        for _dynamic_metrics in dynamic_metrics_list:
            for dot in _dynamic_metrics.point_object_list:
                axes_auc_group += dot
        self.play(
            axes_auc_group.animate.scale(1.2).next_to(LEFT * 0.25),
        )

        auc_list = []
        auc_text = []
        for _dynamic_metrics in dynamic_metrics_list:
            _rectangle = Polygon(
                *[
                    axes_auc.c2p(_fpr, _tpr)
                    for _fpr, _tpr in zip(_dynamic_metrics.fpr, _dynamic_metrics.tpr)
                ],
                axes_auc.c2p(1, 0),
                fill_opacity=0.5,
                color=_dynamic_metrics.dot_color,
                fill_color=_dynamic_metrics.dot_color,
            )
            auc_list.append(_rectangle)

            # AUC
            auc_text.append(
                Tex(f"= {roc_auc_score(_dynamic_metrics.labels, _dynamic_metrics.probs):.2f}")
            )
            self.play(DrawBorderThenFill(_rectangle), run_time=2.5)
        # 軸は必要ないので消す
        self.play(FadeOut(axes_auc_group))

        """
        予測結果の数直線と対応するAUCを分かりやすく並び替える
        """
        for _auc, _auc_text, _num_line in zip(auc_list, auc_text, number_lines):
            self.play(_auc.animate.scale(0.3).move_to(_num_line.get_right() + 1.5 * RIGHT))
        for _auc, _auc_text in zip(auc_list, auc_text):
            _auc_text.next_to(_auc)
            self.play(Create(_auc_text))

    def move_circle_to_lines(
        self,
        labels,
        probs,
        circles: list[Circle],
        position: np.ndarray,
        model_name: str,
    ):
        number_line = NumberLine(
            x_range=[0, 1, 0.1],
            length=6,
            color=BLUE,
            include_numbers=True,
            numbers_to_include=[0, 1],
            # number_scale_value=0.5,
        )
        number_line.move_to(position)
        text_model_name = Text(f"{model_name} Prediction", color=WHITE, font_size=24).move_to(
            number_line.get_left() + 0.5 * UP + LEFT
        )
        self.play(Create(number_line), Create(text_model_name))

        # データ自体は残しておきたいので、複製する
        circles = [circle.copy() for circle in circles]
        self.play(
            *[
                circle.animate.move_to(
                    number_line.n2p(prob) + 0.5 * (UP * label + DOWN * (1 - label)),
                )
                for circle, prob, label in zip(circles, probs, labels)
            ]
        )

        # まとめる
        group = VGroup(number_line, text_model_name, *circles)
        return group

    def plot_auc_curve(self, fpr, tpr):
        """
        AUCカーブを描く
        """
        # 軸の作成
        axes = get_axes(diff=0.1)
        axes.to_edge(RIGHT)

        # AUCカーブを直線で結ぶ
        roc_curve_points = [axes.c2p(_fpr, _tpr) for _fpr, _tpr in zip(fpr, tpr)]
        area_polygon = Polygon(
            *roc_curve_points, axes.c2p(1, 0), fill_opacity=0.5, color=WHITE, fill_color=GREEN
        )

        # AUC値をテキストとして表示
        auc_text = Text(f"AUC: {auc(fpr, tpr):.2f}", font_size=24).move_to(axes.get_top())

        # 補助線
        sub_lines = VGroup(
            horizontal_line,
            vertical_line,
            diagonal_line,
        )

        # 作成
        self.play(Create(axes), run_time=1)
        self.play(FadeIn(sub_lines), run_time=1)
        self.play(DrawBorderThenFill(area_polygon), run_time=3)

        self.play(FadeIn(auc_text))

        return VGroup(axes, sub_lines, area_polygon, auc_text)


def get_axes(
    diff: float = 0.1,
):
    from manim import BLUE, Axes

    axes = Axes(
        x_range=[-0.05, 1.05, diff],
        y_range=[-0.05, 1.05, diff],
        x_length=4,
        y_length=4,
        axis_config={
            "color": BLUE,
            "numbers_to_include": [0, 1],
        },
        x_axis_config={"include_tip": False},
        y_axis_config={"include_tip": False},
    )

    return axes
