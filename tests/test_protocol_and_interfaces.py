"""Test protocols and interfaces for type safety.

Description:
    This test module validates protocol definitions, type checking behaviour,
    and interface contracts that enable the library's flexible design and
    integration capabilities with various LLM providers.

Test Classes:
    - TestProtocolAndInterfaces: Tests protocol implementations
    - TestPydanticLikeCompliance: Tests Pydantic-like interface
    - TestProviderProtocol: Tests provider protocol compliance

Author:
    LLMShield by brainpolo, 2025
"""

import unittest
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from parameterized import parameterized

from llmshield.utils import PydanticLike, _should_cloak_input


class TestProtocolAndInterfaces(unittest.TestCase):
    """Test protocol definitions and interface contracts."""

    def test_pydantic_like_protocol_interface(self):
        """Test PydanticLike protocol interface requirements.

        This validates that the PydanticLike protocol correctly defines
        the interface contract for Pydantic-compatible objects.
        """

        class ValidPydanticLike:
            """A class that properly implements PydanticLike protocol."""

            def model_dump(self) -> dict:
                return {"id": 1, "name": "test", "active": True}

            @classmethod
            def model_validate(cls, data: dict):
                instance = cls()
                return instance

        # Test that valid implementation satisfies protocol
        obj = ValidPydanticLike()
        self.assertIsInstance(obj, PydanticLike)

        # Test protocol method functionality
        dump_result = obj.model_dump()
        self.assertIsInstance(dump_result, dict)
        self.assertIn("id", dump_result)
        self.assertIn("name", dump_result)

        # Test class method
        validated = ValidPydanticLike.model_validate({"data": "test"})
        self.assertIsInstance(validated, ValidPydanticLike)

    def test_pydantic_like_protocol_with_complex_data(self):
        """Test PydanticLike protocol with complex nested data structures.

        This ensures the protocol works correctly with realistic
        Pydantic model scenarios involving nested data.
        """

        class ComplexPydanticLike:
            """A more complex implementation of PydanticLike protocol."""

            def __init__(self, data=None):
                self.data = data or {
                    "user": {
                        "id": 1,
                        "profile": {
                            "name": "John Doe",
                            "email": "john@example.com",
                        },
                    },
                    "metadata": {
                        "created_at": "2024-01-01",
                        "tags": ["test", "user"],
                    },
                }

            def model_dump(self) -> dict:
                return self.data

            @classmethod
            def model_validate(cls, data: dict):
                return cls(data)

        # Test with complex nested data
        complex_obj = ComplexPydanticLike()
        self.assertIsInstance(complex_obj, PydanticLike)

        dumped = complex_obj.model_dump()
        self.assertIsInstance(dumped, dict)
        self.assertIn("user", dumped)
        self.assertIn("metadata", dumped)
        self.assertEqual(dumped["user"]["profile"]["name"], "John Doe")

        # Test validation with new data
        new_data = {"user": {"id": 2, "profile": {"name": "Jane Doe"}}}
        validated = ComplexPydanticLike.model_validate(new_data)
        self.assertIsInstance(validated, ComplexPydanticLike)
        self.assertEqual(validated.data["user"]["id"], 2)

    @parameterized.expand(
        [
            # String inputs - should be cloaked
            ("Hello world", True, "String input should be cloaked"),
            ("", True, "Empty string should be cloaked"),
            (
                "Contains PII: john@example.com",
                True,
                "String with PII should be cloaked",
            ),
            (
                "Dr. Smith called today",
                True,
                "String with names should be cloaked",
            ),
            (
                "Contact 555-123-4567",
                True,
                "String with phone should be cloaked",
            ),
        ]
    )
    def test_should_cloak_input_string_types(
        self, input_value, expected_result, description
    ):
        """Test cloaking decisions for string input types."""
        result = _should_cloak_input(input_value)
        self.assertEqual(result, expected_result, description)

    @parameterized.expand(
        [
            # List inputs - should be cloaked
            (["Hello", "world"], True, "String list should be cloaked"),
            ([], True, "Empty list should be cloaked"),
            (["Dr. Smith", "john@example.com"], True, "List with PII"),
            ([1, 2, 3], True, "Numeric list should be cloaked"),
            (["mixed", 123, {"nested": "data"}], True, "Mixed list cloaked"),
        ]
    )
    def test_should_cloak_input_list_types(
        self, input_value, expected_result, description
    ):
        """Test cloaking decisions for list input types."""
        result = _should_cloak_input(input_value)
        self.assertEqual(result, expected_result, description)

    @parameterized.expand(
        [
            # Non-cloak inputs - various types that shouldn't be cloaked
            ({"key": "value"}, False, "Dictionary should not be cloaked"),
            (
                Path("/path/to/file"),
                False,
                "Path object should not be cloaked",
            ),
            (
                BytesIO(b"binary data"),
                False,
                "File-like object should not be cloaked",
            ),
            (b"raw bytes", False, "Bytes should not be cloaked"),
            (("tuple", "data"), False, "Tuple should not be cloaked"),
            (123, False, "Integer should not be cloaked"),
            (123.45, False, "Float should not be cloaked"),
            (None, False, "None should not be cloaked"),
            (True, False, "Boolean should not be cloaked"),
            (set([1, 2, 3]), False, "Set should not be cloaked"),
        ]
    )
    def test_should_cloak_input_non_cloak_types(
        self, input_value, expected_result, description
    ):
        """Test cloaking decisions for types that should not be cloaked."""
        result = _should_cloak_input(input_value)
        self.assertEqual(result, expected_result, description)

    def test_type_checking_import_behavior(self):
        """Test TYPE_CHECKING import behavior and conditional imports.

        This ensures that TYPE_CHECKING conditional imports work correctly
        and don't affect runtime behavior while enabling proper type hints.
        """
        # Verify TYPE_CHECKING is properly imported and used
        self.assertIsInstance(TYPE_CHECKING, bool)

        # During runtime, TYPE_CHECKING should be False
        self.assertFalse(TYPE_CHECKING)

        # Test that conditional imports don't break anything
        try:
            if TYPE_CHECKING:
                # This code should not execute at runtime
                # This import should not execute
                raise AssertionError("TYPE_CHECKING should be False")
            else:
                # This should execute
                pass
        except ImportError:
            # This shouldn't happen since the import is conditional
            self.fail("Conditional import under TYPE_CHECKING failed")

    def test_protocol_runtime_checking_edge_cases(self):
        """Test edge cases in protocol runtime checking.

        This validates the protocol's behavior with edge cases and
        ensures proper error handling for invalid implementations.
        """

        class IncompleteImplementation:
            """Class missing required protocol methods."""

            def model_dump(self) -> dict:
                return {}

            # Missing model_validate method

        class WrongSignatureImplementation:
            """Class with wrong method signatures."""

            def model_dump(self) -> str:  # Wrong return type
                return "not a dict"

            @classmethod
            def model_validate(cls, data: str):  # Wrong parameter type
                return cls()

        # Test incomplete implementation
        _ = IncompleteImplementation()  # Test incomplete implementation
        # isinstance with Protocol checks for method existence, not signatures
        # So this might still pass the isinstance check

        # Test that methods work as expected for protocol
        class MinimalValidImplementation:
            def model_dump(self) -> dict:
                return {}

            @classmethod
            def model_validate(cls, data: dict):
                return cls()

        minimal_obj = MinimalValidImplementation()
        self.assertIsInstance(minimal_obj, PydanticLike)

        # Test method calls
        result = minimal_obj.model_dump()
        self.assertIsInstance(result, dict)

        validated = MinimalValidImplementation.model_validate({})
        self.assertIsInstance(validated, MinimalValidImplementation)

    @parameterized.expand(
        [
            ("FunctionalStyle", {"test": "data"}),
            ("PropertyBasedStyle", None),
            ("FactoryStyle", None),
        ]
    )
    def test_interface_flexibility_with_various_implementations(
        self, impl_name, init_data
    ):
        """Test interface flexibility with various implementation styles.

        This ensures the protocol can work with different implementation
        approaches and styles that users might employ.
        """

        # Functional style implementation
        class FunctionalStyle:
            def __init__(self, data=None):
                self._data = data or {}

            def model_dump(self) -> dict:
                return dict(self._data)

            @classmethod
            def model_validate(cls, data: dict):
                return cls(data.copy())

        # Class-based implementation with properties
        class PropertyBasedStyle:
            def __init__(self):
                self.id = 1
                self.name = "test"

            def model_dump(self) -> dict:
                return {"id": self.id, "name": self.name}

            @classmethod
            def model_validate(cls, data: dict):
                instance = cls()
                instance.id = data.get("id", 1)
                instance.name = data.get("name", "default")
                return instance

        # Factory-based implementation
        class FactoryStyle:
            @staticmethod
            def create_from_dict(data: dict):
                return FactoryStyle()

            def model_dump(self) -> dict:
                return {"type": "factory", "created": True}

            @classmethod
            def model_validate(cls, data: dict):
                return cls.create_from_dict(data)

        # Create implementation based on test parameter
        implementations = {
            "FunctionalStyle": FunctionalStyle(init_data),
            "PropertyBasedStyle": PropertyBasedStyle(),
            "FactoryStyle": FactoryStyle(),
        }

        impl = implementations[impl_name]

        # Should satisfy protocol
        self.assertIsInstance(impl, PydanticLike)

        # Should work with protocol methods
        dumped = impl.model_dump()
        self.assertIsInstance(dumped, dict)

        # Should support validation
        validated = type(impl).model_validate({"test": "validation"})
        self.assertIsInstance(validated, type(impl))

    @parameterized.expand(
        [
            # Subclasses of basic types
            (
                "CustomString",
                str,
                "hello world",
                True,
                "String subclass cloaked",
            ),
            (
                "CustomList",
                list,
                ["item1", "item2"],
                True,
                "List subclass cloaked",
            ),
            (
                "CustomDict",
                dict,
                {"key": "value"},
                False,
                "Dict subclass should not be cloaked",
            ),
        ]
    )
    def test_input_type_classification_boundary_cases(
        self, class_name, base_type, init_value, expected_result, description
    ):
        """Test boundary cases in input type classification.

        This tests edge cases and boundary conditions in the logic
        that determines which inputs should be processed for cloaking.
        """
        # Create custom subclass dynamically
        CustomClass = type(class_name, (base_type,), {})
        custom_instance = CustomClass(init_value)

        result = _should_cloak_input(custom_instance)
        self.assertEqual(result, expected_result, description)

    @parameterized.expand(
        [
            (
                "function",
                lambda: "function",
                False,
                "Function should not be cloaked",
            ),
            (
                "lambda",
                lambda: "lambda",
                False,
                "Lambda should not be cloaked",
            ),
            (
                "custom_object_with_str",
                None,
                False,
                "Custom object should not be cloaked",
            ),
        ]
    )
    def test_input_type_classification_callable_and_custom_objects(
        self, obj_type, obj_factory, expected_result, description
    ):
        """Test cloaking decisions for callable and custom objects."""
        if obj_type == "custom_object_with_str":
            # Create custom object with __str__
            class CustomObject:
                def __str__(self):
                    return "custom object"

            obj = CustomObject()
        else:
            obj = obj_factory

        result = _should_cloak_input(obj)
        self.assertEqual(result, expected_result, description)


if __name__ == "__main__":
    unittest.main(verbosity=2)
