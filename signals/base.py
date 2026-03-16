from abc import ABC, abstractmethod
import pandas as pd


# ABC (Abstract Base Class): defines a common interface that all signal types must follow.
# Any class inheriting from Signal is forced to implement calc(), ensuring consistency.
# Signal itself cannot be instantiated directly — only its child classes can.
class Signal(ABC):
    def __init__(self, name: str):
        self.name = name

    # Implemented by child class(es)
    @abstractmethod
    def calc(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @classmethod
    @abstractmethod
    def param_schema(cls) -> list[dict]:
        # Returns parameter definitions so the UI can render widgets dynamically.
        # Each dict: {"name": "n_fast", "label": "Fast period", "type": "int", "default": 20, "min": 2, "max": 500}
        pass
