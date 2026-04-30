from typing import Optional
from pydantic import BaseModel
from datasets import Dataset, DatasetDict, load_dataset
from typing import Optional, Self


class Item(BaseModel):
    id: int
    name: str 
    peakCCU: int 
    required_age: int 
    price: float 
    dlcCount: int 
    supportWindows: bool 
    supportMac: bool 
    supportLinux: bool 
    positive: int 
    negative: int 
    achievements: int 
    recommendations: Optional[int] = None 
    release_year: Optional[int] = None 
    release_month: Optional[int] = None 
    release_day: Optional[int] = None 
    small_description: Optional[str] = None 
    min_estimatedOwners: Optional[int] = None 
    max_estimatedOwners: Optional[int] = None 
    supported_languages: Optional[int] = None 
    num_developers: Optional[int] = None 
    num_publishers: Optional[int] = None 
    num_categories: Optional[int] = None 
    num_genres: Optional[int] = None

    @staticmethod
    def push_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        """Push Item lists to HuggingFace Hub"""
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """Load from HuggingFace Hub and reconstruct Items"""
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )