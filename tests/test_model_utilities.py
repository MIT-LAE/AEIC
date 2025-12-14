from AEIC.utils.models import CIBaseModel, CIStrEnum


def test_ci_str_enum():
    class Color(CIStrEnum):
        RED = "red"
        GREEN = "green"
        BLUE = "blue"

    assert str(Color.RED) == "red"
    assert Color("RED") == Color.RED
    assert Color("green") == Color.GREEN
    assert Color("BlUe") == Color.BLUE
    assert Color._missing_("yellow") is None


def test_ci_base_model():
    class NestedModel(CIBaseModel):
        value: int

    class TestModel(CIBaseModel):
        name: str
        nested: NestedModel
        items: list[NestedModel]

    input_data = {
        "NAME": "Test",
        "NESTED": {"VALUE": 42},
        "ITEMS": [{"VALUE": 1}, {"value": 2}, {"VaLuE": 3}],
    }

    model = TestModel.model_validate(input_data)
    assert model.name == "Test"
    assert model.nested.value == 42
    assert [item.value for item in model.items] == [1, 2, 3]


def test_ci_base_model_with_invalid_key():
    class TestModel(CIBaseModel):
        name: str

    input_data = {
        "NAME": "Test",
        "INVALID_KEY": "Should be ignored",
    }

    model = TestModel.model_validate(input_data)
    assert model.name == "Test"
    assert not hasattr(model, "INVALID_KEY")
