from deeprecsys import reweight
from deeprecsys.reweight import LoopCounter


def test_counter_normal():
    spec = {reweight.OptimizationStep.ARG_MIN: 1, reweight.OptimizationStep.ARG_MAX: 2}

    counter = reweight.AlternatingCounter(step_specs=spec)
    output = []
    for _ in range(6):
        output.append(counter.touch())
    expect = [reweight.OptimizationStep.ARG_MIN,
              reweight.OptimizationStep.ARG_MAX,
              reweight.OptimizationStep.ARG_MAX,
              reweight.OptimizationStep.ARG_MIN,
              reweight.OptimizationStep.ARG_MAX,
              reweight.OptimizationStep.ARG_MAX
              ]
    assert output == expect


def test_counter_single():
    spec = {reweight.OptimizationStep.ARG_MIN: 1, reweight.OptimizationStep.ARG_MAX: 0}

    counter = reweight.AlternatingCounter(step_specs=spec)
    output = []
    for _ in range(3):
        output.append(counter.touch())
    expect = [reweight.OptimizationStep.ARG_MIN,
              reweight.OptimizationStep.ARG_MIN,
              reweight.OptimizationStep.ARG_MIN
              ]
    assert output == expect


def test_counter_single_2():
    spec = {reweight.OptimizationStep.ARG_MIN: 0, reweight.OptimizationStep.ARG_MAX: 1}

    counter = reweight.AlternatingCounter(step_specs=spec)
    output = []
    for _ in range(3):
        output.append(counter.touch())
    expect = [reweight.OptimizationStep.ARG_MAX,
              reweight.OptimizationStep.ARG_MAX,
              reweight.OptimizationStep.ARG_MAX
              ]
    assert output == expect


def test_loop_counter():
    spec = [(reweight.OptimizationStep.ARG_MIN, 2), (reweight.OptimizationStep.ARG_MAX, 3)]
    counter = LoopCounter(spec)
    output = []
    for x in counter:
        output.append(x)
    expect = [reweight.OptimizationStep.ARG_MIN,
              reweight.OptimizationStep.ARG_MAX,
              reweight.OptimizationStep.ARG_MAX,
              reweight.OptimizationStep.ARG_MIN,
              reweight.OptimizationStep.ARG_MAX,
              reweight.OptimizationStep.ARG_MAX
              ]


def test_loop_counter_2():
    spec = [(reweight.OptimizationStep.ARG_MIN_F, 1),
            (reweight.OptimizationStep.ARG_MIN_W, 2),
            (reweight.OptimizationStep.ARG_MAX_G, 3)]
    counter = LoopCounter(spec)
    output = []
    for x in counter:
        output.append(x)
    expect = [reweight.OptimizationStep.ARG_MIN_F,
              reweight.OptimizationStep.ARG_MIN_W,
              reweight.OptimizationStep.ARG_MAX_G,
              reweight.OptimizationStep.ARG_MAX_G,
              reweight.OptimizationStep.ARG_MAX_G,
              reweight.OptimizationStep.ARG_MIN_W,
              reweight.OptimizationStep.ARG_MAX_G,
              reweight.OptimizationStep.ARG_MAX_G,
              reweight.OptimizationStep.ARG_MAX_G
              ]


