from .BaseAgent import BaseHierarchicalNavigationAgent
from .registry import registry
from typing import Literal

class HierarchicalNavigationAgent(BaseHierarchicalNavigationAgent):

    """
    `BaseHierarchicalNavigationAgent` with the bundled configurations in front of it.

    `CONFIGS` holds a ready-made signature and the four tool descriptions per shipped
    classification, so only the credentials and the model have to be supplied. For your own
    taxonomy either add an entry to `CONFIGS` or construct
    `BaseHierarchicalNavigationAgent` directly and pass the signature and tool descriptions
    yourself; both are the same code path.

    Parameters:
        classification_name (str): a key of `CONFIGS`.
        **kwargs (Any): forwarded to `BaseHierarchicalNavigationAgent`, e.g. `api_key`, `model_name`, `api_base`.
    """
    CONFIGS = registry

    def __init__(
        self,
        classification_name:Literal["COICOP_2018", "ICATUS_2016", "NAF_rev2", "SEA_2021"],
        **kwargs,
    ) -> None:
        """
        Looks the configuration up in `CONFIGS` and hands it to the base class.

        Anything you pass yourself is forwarded unchanged, so a bundled entry can be
        overridden per call.

        Raises:
            ValueError: if `classification_name` has no entry in `CONFIGS`.
        """
        if classification_name not in self.CONFIGS:
            raise ValueError(f"Unknown classification system: {classification_name}")

        super().__init__(
            classification_name=classification_name,
            **self.CONFIGS[classification_name],
            **kwargs,
        )