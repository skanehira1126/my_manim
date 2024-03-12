import numpy as np
from manim import *
from sklearn.metrics import precision_score, recall_score


def calculate_precision_recall(threshold, positives, negatives):
    tp = sum(pos > threshold for pos in positives)  # True Positives
    fp = sum(neg > threshold for neg in negatives)  # False Positives
    fn = sum(pos <= threshold for pos in positives)  # False Negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall


class AUC2(Scene):
    def construct(self):
        # 軸とデータポイントの作成
        axis = NumberLine(
            x_range=[0, 1, 0.1],
            length=10,
            color=WHITE,
            include_numbers=True,
            label_direction=DOWN,
        )
        self.add(axis)

        # 正例と負例のシミュレーション
        positive_x = np.random.uniform(0.2, 1, size=50)
        negative_x = np.random.uniform(0, 0.7, size=50)
        labels = np.concatenate([np.ones(50), np.zeros(50)])
        predicts = np.concatenate([positive_x, negative_x])
        positives = [Dot(axis.number_to_point(x), color=GREEN) for x in positive_x]
        negatives = [Dot(axis.number_to_point(x), color=RED) for x in negative_x]

        for dot in positives + negatives:
            self.add(dot)

        # 閾値の初期位置
        threshold = 0.5
        threshold_line = DashedLine(
            start=axis.number_to_point(threshold),
            end=axis.number_to_point(threshold) + UP * 2,
            color=BLUE,
        )
        self.add(threshold_line)
        self.wait(2)

        # PrecisionとRecallのテキストオブジェクトの初期化
        precision_text = Text("Precision: 0.00", font_size=24).to_edge(UP)
        recall_text = Text("Recall: 0.00", font_size=24).to_edge(UP).shift(DOWN * 0.5)
        self.add(precision_text, recall_text)

        # 閾値を動かすアニメーション
        for new_threshold in np.linspace(0, 1, 50):
            precision = precision_score(labels, predicts >= new_threshold)
            recall = recall_score(labels, predicts >= new_threshold)
            new_threshold_line = DashedLine(
                start=axis.number_to_point(new_threshold),
                end=axis.number_to_point(new_threshold) + UP * 2,
                color=BLUE,
            )

            self.play(
                ReplacementTransform(threshold_line, new_threshold_line),
                run_time=0.1,
                rate_func=linear,
            )
            threshold_line = new_threshold_line

            precision_text.become(
                Text(f"Precision: {precision:.2f}", font_size=24).to_edge(UP)
            )
            recall_text.become(
                Text(f"Recall: {recall:.2f}", font_size=24)
                .to_edge(UP)
                .shift(DOWN * 0.5)
            )
            self.play(Write(precision_text), Write(recall_text), run_time=0.1)

        self.wait(2)
