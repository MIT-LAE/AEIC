from pydantic import RootModel

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

    # _normalize_dict does NOT drop unknown keys — when `field_map.get(k.lower(), k)`
    # finds no field, it passes the original key through into the normalized
    # dict. The actual silent-drop happens at Pydantic's default
    # `extra='ignore'` policy, which surfaces as `model_extra is None`
    # (no extras retained) and `INVALID_KEY` not appearing in
    # `model_fields_set`. Pin both so a future tightening of either layer
    # (e.g. switching to `extra='forbid'` or having `_normalize_dict` drop
    # unknown keys explicitly) trips this test.
    normalized = TestModel._normalize_dict(input_data)
    # Known keys are case-folded to the field name; unknown keys are passed
    # through verbatim (they're not in `field_map`).
    assert normalized["name"] == "Test"
    assert normalized["INVALID_KEY"] == "Should be ignored"

    model = TestModel.model_validate(input_data)
    assert model.name == "Test"
    assert not hasattr(model, "INVALID_KEY")
    assert model.model_extra is None  # Pydantic's `extra='ignore'` drops it
    assert "INVALID_KEY" not in model.model_fields_set


def test_ci_base_model_top_level_list():
    """The `normalize_keys` `elif isinstance(values, list)` branch only
    fires for top-level lists, e.g. when a `RootModel[list[X]]` is
    validated. Each list item is normalized if it's a dict, passed through
    unchanged otherwise — pin both legs.
    """

    class Item(CIBaseModel):
        value: int

    class Items(RootModel[list[Item]]):
        pass

    items = Items.model_validate(
        [
            {"VALUE": 1},  # case-folded by _normalize_dict
            {"value": 2},  # already lowercase
            {"VaLuE": 3},  # mixed case
        ]
    )
    assert [it.value for it in items.root] == [1, 2, 3]

    # And the inner `else v` arm: non-dict items in a top-level list are
    # passed through unchanged. Call `normalize_keys` directly because a
    # mixed-shape list wouldn't survive downstream model validation.
    raw = [{"VALUE": 1}, "passthrough", 7]
    out = Item.normalize_keys(raw)
    assert out == [{"value": 1}, "passthrough", 7]
