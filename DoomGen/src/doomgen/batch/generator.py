from typing import Any, Dict, List, Type, Iterator
import itertools
from copy import deepcopy
from .scenario import Scenario

class BatchGenerator:
    """
    Helper for generating batches of scenarios from a parameter grid.
    """

    def __init__(self, scenario_cls: Type[Scenario], param_grid: Dict[str, List[Any]]):
        """
        Args:
            scenario_cls: The class of the Scenario to generate.
            param_grid: A dictionary mapping parameter names to lists of possible values.
        """
        self.scenario_cls = scenario_cls
        self.param_grid = param_grid

    def generate_configs(self) -> Iterator[Dict[str, Any]]:
        """
        Yields configuration dictionaries for every combination in the parameter grid.
        """
        # separate keys and values
        keys = self.param_grid.keys()
        values = self.param_grid.values()

        # itertools.product generates cartesian product
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def generate_scenarios(self, base_name: str = "batch") -> Iterator[Scenario]:
        """
        Yields initialized Scenario instances for every combination.
        """
        for i, config in enumerate(self.generate_configs()):
            name = f"{base_name}_{i}"
            # Create scenario with the combined config
            # We merge with default config inside the Scenario constructor
            scenario = self.scenario_cls(config, name=name)
            yield scenario

    def count_combinations(self) -> int:
        """Returns the total number of combinations."""
        count = 1
        for v in self.param_grid.values():
            count *= len(v)
        return count
