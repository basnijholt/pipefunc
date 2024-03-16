from pipefunc import Pipeline, Sweep, pipefunc


def test_connect():
    @pipefunc(output_name="x")
    def f1(a, b):
        return [a + b, a - b, a * b]

    @pipefunc(output_name="y")
    def f2(x_average):
        return x_average * 2

    pipeline1 = Pipeline([f1])
    pipeline2 = Pipeline([f2])

    def average(x):
        return sum(x) / len(x)

    pipeline2.connect(pipeline1, "x", "x_average", average)

    assert len(pipeline2.functions) == 2
    assert pipeline2._reducers[("pipeline1", "x")] == ("x_average", average)

    result = pipeline2("y", a=5, b=3)
    assert result == 16


def test_connect_multiple():
    @pipefunc(output_name="x")
    def f1(a, b):
        return [a + b, a - b, a * b]

    @pipefunc(output_name="y")
    def f2(x_average):
        return x_average * 2

    @pipefunc(output_name="z")
    def f3(y_average, c):
        return y_average + c

    pipeline1 = Pipeline([f1])
    pipeline2 = Pipeline([f2])
    pipeline3 = Pipeline([f3])

    def average(x):
        return sum(x) / len(x)

    pipeline3.connect(pipeline2, "y", "y_average", average)
    pipeline2.connect(pipeline1, "x", "x_average", average)

    assert len(pipeline3.functions) == 3
    assert pipeline3._reducers[("pipeline2", "y")] == ("y_average", average)
    assert pipeline2._reducers[("pipeline1", "x")] == ("x_average", average)

    result = pipeline3("z", a=5, b=3, c=2)
    assert result == 18


def test_evaluate_sweep():
    @pipefunc(output_name="x")
    def f1(a, b):
        return [a + b, a - b, a * b]

    @pipefunc(output_name="y")
    def f2(x):
        return sum(x)

    pipeline = Pipeline([f1, f2])

    sweep = Sweep(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        },
    )

    results = pipeline.evaluate_sweep("y", sweep)
    assert results == [15, 28, 45]


def test_evaluate_sweep_with_reducer():
    @pipefunc(output_name="x")
    def f1(a, b):
        return [a + b, a - b, a * b]

    @pipefunc(output_name="y")
    def f2(x_average):
        return x_average * 2

    pipeline1 = Pipeline([f1])
    pipeline2 = Pipeline([f2])

    def average(x):
        return sum(x) / len(x)

    pipeline2.connect(pipeline1, "x", "x_average", average)

    sweep = Sweep(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        },
    )

    results = pipeline2.evaluate_sweep("y", sweep)
    assert results == [10, 14, 18]


def test_connect_three_pipelines():
    @pipefunc(output_name="x")
    def f1(a, b):
        return [a + b, a - b, a * b]

    @pipefunc(output_name="y")
    def f2(x_average):
        return x_average * 2

    @pipefunc(output_name="z")
    def f3(y_average, c):
        return y_average + c

    @pipefunc(output_name="result")
    def f4(z_average, d):
        return z_average * d

    pipeline1 = Pipeline([f1])
    pipeline2 = Pipeline([f2])
    pipeline3 = Pipeline([f3])
    pipeline4 = Pipeline([f4])

    def average(x):
        return sum(x) / len(x)

    pipeline4.connect(pipeline3, "z", "z_average", average)
    pipeline3.connect(pipeline2, "y", "y_average", average)
    pipeline2.connect(pipeline1, "x", "x_average", average)

    assert len(pipeline4.functions) == 1
    assert pipeline4._reducers[(pipeline3, "z")] == ("z_average", average)
    assert pipeline3._reducers[(pipeline2, "y")] == ("y_average", average)
    assert pipeline2._reducers[(pipeline1, "x")] == ("x_average", average)

    sweep = Sweep(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
            "d": [10],
        },
        dims=[("a", "b"), ("c",), ("d",)],
    )

    results = pipeline4.evaluate_sweep("result", sweep)
    assert results == [170, 190, 210]
