from .BaseAgent import BaseHierarchicalNavigationAgent
from .registry import registry
from typing import Literal

class HierarchicalNavigationAgent(BaseHierarchicalNavigationAgent):

    CONFIGS = registry

    def __init__(
        self,
        classification_name:Literal["COICOP_2018", "ICATUS_2016", "NAF_rev2", "SEA_2021"],
        **kwargs,
    ) -> None:
        if classification_name not in self.CONFIGS:
            raise ValueError(f"Unknown classification system: {classification_name}")

        super().__init__(
            classification_name=classification_name,
            **self.CONFIGS[classification_name],
            **kwargs,
        )