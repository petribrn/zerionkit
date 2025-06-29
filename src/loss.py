import math


class Loss:
    class SquareError:
        @staticmethod
        def main(y_predict: float, y_target: float) -> list[float]:
            return [(y_target - y_predict) ** 2]

        @staticmethod
        def derivative(y_predict: float, y_target: float) -> list[float]:
            return [y_predict - y_target]

    class BinaryCrossEntropy:
        @staticmethod
        def main(y_predict: float, y_target: float) -> list[float]:
            y_predict_norm = normalize_y_for_classification(y_predict)

            return [
                - (y_target * math.log(y_predict_norm) + (1 - y_target) * math.log(1 - y_predict_norm))
            ]

        @staticmethod
        def derivative(y_predict: float, y_target: float) -> list[float]:
            return [y_predict - y_target]

    class MultiClassCrossEntropy:
        @staticmethod
        def main(y_predict: list[float], y_target: list[float]) -> list[float]:
            y_predict_norm = [normalize_y_for_classification(y_p_r) for y_p_r in y_predict]

            return [
                - sum(
                    y_t * math.log(y_p) for y_t, y_p in zip(y_target, y_predict_norm)
                )
            ]

        @staticmethod
        def derivative(y_predict: list[float], y_target: list[float]) -> list[float]:
            y_predict_norm = [normalize_y_for_classification(y_p_r) for y_p_r in y_predict]

            return [y_p - y_t for y_p, y_t in zip(y_predict_norm, y_target)]


def normalize_y_for_classification(y_predict: float) -> float:
    eps = 1e-15  # avoid log(0)
    return max(min(y_predict, 1 - eps), eps)
