from skopt.space import Real, Integer, Categorical
from skopt.optimizer import Optimizer

from rafiki.model import BaseKnob, FloatKnob, IntegerKnob, CategoricalKnob, FixedKnob
from ..advisor import BaseAdvisor

class SkoptAdvisor(BaseAdvisor):
    '''
    Uses `skopt`'s `Optimizer`
    '''   
    def __init__(self, knob_config):
        self._dimensions = self._get_dimensions(knob_config)
        self._optimizer = Optimizer(list(self._dimensions.values()))

    def propose(self):
        point = self._optimizer.ask()
        return { knob : value for (knob, value) in zip(self._dimensions.keys(), point) }

    def feedback(self, knobs, score):
        point = [ knobs[name] for name in self._dimensions.keys() ]
        self._optimizer.tell(point, -score)

    def _get_dimensions(self, knob_config):
        dimensions = {
            name: _knob_to_dimension(x)
                for (name, x)
                in knob_config.items()
        }
        return dimensions

def _knob_to_dimension(knob):
    if isinstance(knob, CategoricalKnob):
        return Categorical(knob.values)
    elif isinstance(knob, FixedKnob):
        return Categorical([knob.value])
    elif isinstance(knob, IntegerKnob):
        return Integer(knob.value_min, knob.value_max)
    elif isinstance(knob, FloatKnob):
        if knob.is_exp:
            return Real(knob.value_min, knob.value_max, 'log-uniform')
        else:
            return Real(knob.value_min, knob.value_max, 'uniform')
    