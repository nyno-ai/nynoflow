import json
from copy import deepcopy
from enum import Enum
from typing import Any, Optional, cast

import pytest
from pydantic import BaseModel, Field

from nynoflow.utils.output_parser import output_parser


class Gender(str, Enum):
    """Gender enum. Used to test enum parsing."""

    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class Address(BaseModel):
    """Address model. Used to test standard nested models."""

    street: str
    city: str
    state: str
    zip_code: str


class SocialMedia(BaseModel):
    """Social media model. Used to test optional nested models."""

    facebook: Optional[str]
    twitter: Optional[str]


class Employee(BaseModel):
    """Employee model. Used to test nested models."""

    id: int
    name: str
    email: str
    gender: Gender
    age: int = Field(..., gt=18, lt=65)
    address: Address
    social_media: Optional[SocialMedia] = None


class TestOutputParser:
    """Test the output parser."""

    valid_employee_data = {
        "id": 1,
        "name": "John Doe",
        "email": "john.doe@example.com",
        "gender": "male",
        "age": 30,
        "address": {
            "street": "123 Main St",
            "city": "Springfield",
            "state": "IL",
            "zip_code": "62704",
        },
        "social_media": {"facebook": "john.doe", "twitter": "johndoe"},
    }

    def test_valid_data(self) -> None:
        """Expect no raised exceptions with valid data."""
        valid_employee_json = json.dumps(self.valid_employee_data)
        output = output_parser(Employee, valid_employee_json)
        assert output

    def test_missing_key(self) -> None:
        """Expect a validation error if a required key is missing. Gender key is missing."""
        data = deepcopy(self.valid_employee_data)
        del data["gender"]
        json_data = json.dumps(data)
        with pytest.raises(ValueError):
            output_parser(Employee, json_data)

    def test_missing_nested_key(self) -> None:
        """Expect a validation error if a required nested key is missing. Missing address state."""
        data = cast(dict[str, Any], deepcopy(self.valid_employee_data))
        del data["address"]["state"]

        json_data = json.dumps(data)
        with pytest.raises(ValueError):
            output_parser(Employee, json_data)

    def test_missing_optional_key(self) -> None:
        """Expect a validation error if an optional key is missing. Missing Social Media key."""
        data = deepcopy(self.valid_employee_data)
        del data["social_media"]
        json_data = json.dumps(data)
        output_parser(Employee, json_data)

    def test_enum(self) -> None:
        """Except an exception if an enum value is not as expected. Gender is not a valid enum."""
        data = deepcopy(self.valid_employee_data)
        data["gender"] = "INVALID"
        json_data = json.dumps(data)
        with pytest.raises(ValueError):
            output_parser(Employee, json_data)

    def test_invalid_range(self) -> None:
        """Expect a validation error if a value is not in the expected range. Age is not in range."""
        data = deepcopy(self.valid_employee_data)
        data["age"] = 10
        json_data = json.dumps(data)
        with pytest.raises(ValueError):
            output_parser(Employee, json_data)
