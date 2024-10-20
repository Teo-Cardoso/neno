from abc import abstractmethod


class Runner:
    def __init__(self, input_type, output_types):
        self._input_type = input_type
        self._output_types = output_types

    def check_inputs(self, input_type) -> bool:
        return self._input_type == input_type

    def check_outputs(self, output_types) -> bool:
        return self._output_types == output_types

    @abstractmethod
    def execute(self):
        pass
