"""
EHR Preprocessing Module.

Provides standard preprocessing interfaces for patient trajectory data,
including normalization, filtering, and handling missing observations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseEHRPreprocessor(ABC):
    """
    Abstract Base Class for EHR dataset cleaning and feature scaling.
    """

    @abstractmethod
    def fit_transform(self, raw_records: Any) -> Any:
        """
        Fit preprocessor parameters on training data and transform records.

        Args:
            raw_records: Input raw data.

        Returns:
            Preprocessed data structure.
        """
        pass

    @abstractmethod
    def transform(self, raw_records: Any) -> Any:
        """
        Transform test/validation records using fitted parameters.

        Args:
            raw_records: Input raw data.

        Returns:
            Preprocessed data structure.
        """
        pass
