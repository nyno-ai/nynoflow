import json
from copy import deepcopy
from enum import Enum
from typing import Any, Optional, cast

import pytest
from pydantic import BaseModel, Field
from pytest_mock import MockerFixture

from nynoflow.chats import Chat
from nynoflow.chats._chatgpt._chatgpt import ChatgptProvider
from nynoflow.exceptions import InvalidResponseError
from tests.chat.helpers import render_chatgpt_response


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

    def setup_method(self) -> None:
        """Setup the test."""
        self.employee = {
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
        self.chat = Chat(
            providers=[
                ChatgptProvider(
                    api_key="sk-123",
                    model="gpt-3.5-turbo-0613",
                )
            ]
        )

    def test_valid_data(self, mocker: MockerFixture) -> None:
        """Expect no raised exceptions with valid data."""
        json_data = json.dumps(self.employee)
        mocker.patch(
            "openai.ChatCompletion.create",
            return_value=render_chatgpt_response(json_data),
        )
        output = self.chat.completion_with_output_formatter(
            "doesnt matter we are mocking the response", output_format=Employee
        )
        assert output == Employee.model_validate(self.employee)

    def test_missing_key(self, mocker: MockerFixture) -> None:
        """Expect a validation error if a required key is missing. Gender key is missing."""
        del self.employee["gender"]
        json_data = json.dumps(self.employee)
        mocker.patch(
            "openai.ChatCompletion.create",
            return_value=render_chatgpt_response(json_data),
        )
        with pytest.raises(InvalidResponseError):
            self.chat.completion_with_output_formatter(
                "doesnt matter", output_format=Employee
            )

    def test_missing_nested_key(self, mocker: MockerFixture) -> None:
        """Expect a validation error if a required nested key is missing. Missing address state."""
        data = cast(dict[str, Any], deepcopy(self.employee))
        del data["address"]["state"]
        json_data = json.dumps(data)
        mocker.patch(
            "openai.ChatCompletion.create",
            return_value=render_chatgpt_response(json_data),
        )

        with pytest.raises(InvalidResponseError):
            self.chat.completion_with_output_formatter(
                "doesnt matter", output_format=Employee
            )

    def test_missing_optional_key(self, mocker: MockerFixture) -> None:
        """Expect a validation error if an optional key is missing. Missing Social Media key."""
        del self.employee["social_media"]
        json_data = json.dumps(self.employee)
        mocker.patch(
            "openai.ChatCompletion.create",
            return_value=render_chatgpt_response(json_data),
        )
        output = self.chat.completion_with_output_formatter(
            "doesnt matter", output_format=Employee
        )
        assert output == Employee.model_validate(self.employee)
        assert output.social_media is None

    def test_enum(self, mocker: MockerFixture) -> None:
        """Except an exception if an enum value is not as expected. Gender is not a valid enum."""
        self.employee["gender"] = "INVALID"
        json_data = json.dumps(self.employee)
        mocker.patch(
            "openai.ChatCompletion.create",
            return_value=render_chatgpt_response(json_data),
        )
        with pytest.raises(InvalidResponseError):
            self.chat.completion_with_output_formatter(
                "doesnt matter", output_format=Employee
            )

    def test_invalid_range(self, mocker: MockerFixture) -> None:
        """Expect a validation error if a value is not in the expected range. Age is not in range."""
        self.employee["age"] = 10
        json_data = json.dumps(self.employee)
        mocker.patch(
            "openai.ChatCompletion.create",
            return_value=render_chatgpt_response(json_data),
        )

        with pytest.raises(InvalidResponseError):
            self.chat.completion_with_output_formatter(
                "doesnt matter", output_format=Employee
            )

    def test_retries_success(self, mocker: MockerFixture) -> None:
        """Expect a success after multiple errors in the allocated retry count."""
        json_data = json.dumps(self.employee)

        mocker.patch(
            "openai.ChatCompletion.create",
            # Will fail the first 2 times because invalid json
            side_effect=[
                render_chatgpt_response(json_data[:-10]),
                render_chatgpt_response(json_data[:-10]),
                render_chatgpt_response(json_data),
            ],
        )

        output = self.chat.completion_with_output_formatter(
            "doesnt matter", output_format=Employee, auto_fix_retries=5
        )

        assert output == Employee.model_validate(self.employee)

    def test_retries_failure(self, mocker: MockerFixture) -> None:
        """Expect a failure after multiple errors in the allocated retry count."""
        json_data = json.dumps(self.employee)

        mocker.patch(
            "openai.ChatCompletion.create",
            # Will fail the first 2 times because invalid json
            side_effect=[
                render_chatgpt_response(json_data[:-10]),
                render_chatgpt_response(json_data[:-10]),
            ],
        )

        with pytest.raises(InvalidResponseError):
            self.chat.completion_with_output_formatter(
                "doesnt matter", output_format=Employee, auto_fix_retries=0
            )
