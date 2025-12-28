from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
import mne

class BaseLoader(ABC):
    def __init__(self, config: Dict[str, Any], raw_data_path: Path):
        self.config = config
        self.raw_data_path = raw_data_path

    @abstractmethod
    def get_subject_list(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def load_subject(self, subject_info: Dict[str, Any]) -> Optional[mne.io.Raw]:
        pass