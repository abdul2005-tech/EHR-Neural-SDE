"""
Raw EHR Data Ingestion Module.

Provides abstract interfaces for loading raw longitudinal EHR datasets
without assuming a specific underlying schema or database structure.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseDataIngestor(ABC):
    """
    Abstract Base Class for loading raw EHR data sources.
    """

    @abstractmethod
    def load_raw_data(self, source_path: str) -> Any:
        """
        Load raw EHR records from storage.

        Args:
            source_path: Path to the raw data file or directory.

        Returns:
            Raw data object (e.g., DataFrame or dictionary of tables).
        """
        pass

    @abstractmethod
    def validate_schema(self, raw_data: Any) -> bool:
        """
        Validate that raw data contains minimum required field identifiers.

        Args:
            raw_data: Raw loaded dataset.

        Returns:
            True if schema is valid, False otherwise.
        """
        pass
