from typing import Optional
from pydantic import BaseModel
from datasets import Dataset, DatasetDict, load_dataset
from typing import Optional, Self


class Stream_Item(BaseModel):
    id: int
    prompt: Optional[str] = None 
    completion: Optional[str] = None

    @classmethod
    def push_to_hub(cls, dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        """Push Item lists to HuggingFace Hub"""
        DatasetDict(
            {
                "train": Dataset.from_list([cls.model_validate(item).model_dump() for item in train]),
                "validation": Dataset.from_list([cls.model_validate(item).model_dump() for item in val]),
                "test": Dataset.from_list([cls.model_validate(item).model_dump() for item in test]),
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