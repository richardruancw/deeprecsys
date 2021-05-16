from deeprecsys import reweight


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


