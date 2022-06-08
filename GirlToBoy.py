import numpy as np


def act(x):
    return 0 if x < 0.5 else 1


def go(house, rock, attr):
    x = np.array([house, rock, attr])

    # нейронные ветки(связи):
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]

    weight1 = np.array([w11, w12])  # matrix 2x3
    weight2 = np.array([-1, 1])  # matrix 1x2

    # вычисляем сумму на входах нейронов скрытого слоя
    sum_hidden = np.dot(weight1, x)
    print(f"Значения сумм на нейронах скрытого слоя: {str(sum_hidden)}")

    out_hidden = np.array([act(x) for x in sum_hidden])
    print(f"Значения на выходах нейронов скрытого слоя: {str(out_hidden)}")

    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)
    print(f"Выходное значение НС: {str(y)}")

    return y


house = 1
rock = 0
attr = 1

res = go(house, rock, attr)
if res == 1:
    print("Ты мне нравишься")
else:
    print("Созвонимся")