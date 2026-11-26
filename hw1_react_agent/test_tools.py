import json

import pytest
from pydantic import ValidationError, BaseModel
from typing import cast

from tools import (
    TOOLS, calculate_order, check_availability,
    get_item_details, search_items, SearchItemsInput
)


def validate(tool, args: dict):
    """Перевіряє лише Pydantic-схему tool, без виконання функції й без handle_validation_error."""
    return tool.args_schema(**args)

def run(tool, args: dict) -> dict:
    """Викликає tool і розпарсює JSON-відповідь."""
    return json.loads(tool.invoke(args))


def ids(result: dict) -> list[str]:
    return [item["id"] for item in result["items"]]

class TestSearchItems:
    def test_by_category(self):
        res = run(search_items, {"category": "desserts"})
        assert ids(res) == ["DISH-010", "DISH-009"]  # відсортовано за ціною: 145, 155

    def test_max_price(self):
        res = run(search_items, {"max_price": 100})
        assert ids(res) == ["DISH-012", "DISH-011"]  # 85, 95

    def test_dietary_tags_require_all(self):
        res = run(search_items, {"dietary_tags": ["vegan"], "limit": 10})
        assert ids(res) == ["DISH-011", "DISH-010", "DISH-004"]

    def test_limit(self):
        res = run(search_items, {"limit": 1})
        assert res["found"] == 1

    def test_unavailable_hidden_by_default(self):
        assert run(search_items, {"query": "лосось"})["found"] == 0
        res = run(search_items, {"query": "лосось", "only_available": False})
        assert ids(res) == ["DISH-006"]

    def test_query_inflected_word(self):
        res = run(search_items, {"query": "лосося", "only_available": False})
        assert ids(res) == ["DISH-006"]

    def test_exclude_fish_keeps_mushrooms(self):
        # регресія: «гриби» не повинні збігатися з алергеном fish
        res = run(search_items, {"query": "гриби", "exclude_allergens": ["fish"]})
        assert ids(res) == ["DISH-007"]

    def test_exclude_dairy_keeps_coconut_milk(self):
        # регресія: «кокосове молоко» не є молочним алергеном
        res = run(search_items, {"category": "desserts", "exclude_allergens": ["dairy"]})
        assert ids(res) == ["DISH-010"]

    def test_exclude_dairy_keeps_lactose_free_latte(self):
        res = run(search_items, {"category": "drinks", "exclude_allergens": ["dairy"]})
        assert sorted(ids(res)) == ["DISH-011", "DISH-012"]

    def test_blank_query_means_no_filter(self):
        res = run(search_items, {"query": "   "})
        assert res["found"] == 3  # limit за замовчуванням

    def test_nothing_found(self):
        res = run(search_items, {"query": "піца"})
        assert res["found"] == 0
        assert "message" in res

    @pytest.mark.parametrize("bad_args", [
        {"max_price": -5},
        {"max_prep_time_min": 0},
        {"limit": 0},
        {"limit": 21},
        {"category": "pizza"},
        {"dietary_tags": ["keto"]},
        {"exclude_allergens": ["nuts"]},
        {"unknown_field": 1},
    ])
    def test_validation_errors(self, bad_args):
        with pytest.raises(ValidationError):
            validate(search_items, bad_args)

    def test_unavailable_matches_reported(self):
        res = run(search_items, {"query": "лосось"})
        assert res["found"] == 0
        assert [m["id"] for m in res["unavailable_matches"]] == ["DISH-006"]

    def test_unavailable_matches_respect_filters(self):
        # лосось коштує 420 і має алерген fish: під ці фільтри він не підходить
        assert "unavailable_matches" not in run(search_items, {"query": "лосось", "max_price": 100})
        assert "unavailable_matches" not in run(search_items, {"query": "лосось", "exclude_allergens": ["fish"]})

class TestGetItemDetails:
    def test_success_and_id_normalization(self):
        res = run(get_item_details, {"item_id": " dish-001 "})
        assert res["id"] == "DISH-001"
        assert res["price"] == 165
        assert "ingredients" in res and "allergens" in res

    def test_not_found(self):
        res = run(get_item_details, {"item_id": "DISH-999"})
        assert "error" in res
        assert "hint" in res

    @pytest.mark.parametrize("bad_id", ["123", "DISH-1", "DISH-0001", "", "pizza"])
    def test_invalid_id_format(self, bad_id):
        with pytest.raises(ValidationError):
            validate(get_item_details, {"item_id": bad_id})

    def test_missing_and_extra_fields(self):
        with pytest.raises(ValidationError):
            validate(get_item_details, {})
        with pytest.raises(ValidationError):
            validate(get_item_details, {"item_id": "DISH-001", "foo": 1})

class TestCheckAvailability:
    def test_mixed_statuses(self):
        res = run(check_availability, {"item_ids": ["DISH-001", "DISH-006", "DISH-999"]})
        statuses = {i["id"]: i["status"] for i in res["items"]}
        assert statuses == {
            "DISH-001": "available",
            "DISH-006": "unavailable",
            "DISH-999": "not_found",
        }
        assert res["all_available"] is False

    def test_all_available(self):
        res = run(check_availability, {"item_ids": ["DISH-001", "DISH-002"]})
        assert res["all_available"] is True

    def test_duplicates_removed(self):
        res = run(check_availability, {"item_ids": ["DISH-001", "dish-001"]})
        assert len(res["items"]) == 1

    def test_empty_list(self):
        with pytest.raises(ValidationError):
            validate(check_availability, {"item_ids": []})

    def test_too_many_ids(self):
        many = [f"DISH-{i:03d}" for i in range(1, 12)]  # 11 штук
        with pytest.raises(ValidationError):
            validate(check_availability, {"item_ids": many})

    def test_bad_id_in_list(self):
        with pytest.raises(ValidationError):
            validate(check_availability, {"item_ids": ["DISH-001", "abc"]})

class TestCalculateOrder:
    def test_basic_total(self):
        res = run(calculate_order, {"items": [
            {"item_id": "DISH-010", "quantity": 2},
            {"item_id": "DISH-011", "quantity": 1},
        ]})
        assert res["total"] == 385  # 2*145 + 95
        assert res["currency"] == "UAH"
        assert res["estimated_prep_time_min"] == 5
        assert len(res["lines"]) == 2

    def test_default_quantity_is_one(self):
        res = run(calculate_order, {"items": [{"item_id": "DISH-010"}]})
        assert res["total"] == 145

    def test_duplicates_merged(self):
        res = run(calculate_order, {"items": [
            {"item_id": "DISH-010", "quantity": 1},
            {"item_id": "dish-010", "quantity": 1},
        ]})
        assert len(res["lines"]) == 1
        assert res["lines"][0]["quantity"] == 2
        assert res["total"] == 290

    def test_prep_time_is_max(self):
        res = run(calculate_order, {"items": [
            {"item_id": "DISH-005", "quantity": 1},  # 18 хв, 250
            {"item_id": "DISH-010", "quantity": 1},  # 5 хв, 145
        ]})
        assert res["estimated_prep_time_min"] == 18
        assert res["total"] == 395

    def test_unavailable_item(self):
        res = run(calculate_order, {"items": [{"item_id": "DISH-006", "quantity": 1}]})
        assert "error" in res
        assert res["unavailable"] == ["DISH-006"]

    def test_unknown_item(self):
        res = run(calculate_order, {"items": [{"item_id": "DISH-999", "quantity": 1}]})
        assert "error" in res
        assert res["not_found"] == ["DISH-999"]

    @pytest.mark.parametrize("bad_args", [
        {"items": []},
        {"items": [{"item_id": "DISH-010", "quantity": 0}]},
        {"items": [{"item_id": "DISH-010", "quantity": 21}]},
        {"items": [{"item_id": "DISH-010", "quantity": 12},
                   {"item_id": "DISH-010", "quantity": 12}]},      # сума 24 > 20
        {"items": [{"item_id": "bad", "quantity": 1}]},
        {"items": [{"item_id": "DISH-010", "quantity": 1, "note": "x"}]},  # зайве поле у вкладеній моделі
        {"items": [{"item_id": "DISH-010"}], "promo_code": "X"},           # поле, якого немає в схемі
    ])
    def test_validation_errors(self, bad_args):
        with pytest.raises(ValidationError):
            validate(calculate_order, bad_args)

class TestSchemas:
    def test_search_dedupes_lists(self):
        s = SearchItemsInput(dietary_tags=["vegan", "vegan", "gluten_free"])
        assert s.dietary_tags == ["vegan", "gluten_free"]

    @pytest.mark.parametrize("t", TOOLS, ids=lambda t: t.name)
    def test_tool_schema_is_strict_and_described(self, t):
        schema = t.args_schema.model_json_schema()
        assert schema["additionalProperties"] is False
        for name, prop in schema["properties"].items():
            assert prop.get("description"), f"{t.name}.{name} без description"

    @pytest.mark.parametrize("t", TOOLS, ids=lambda t: t.name)
    def test_tool_has_docstring(self, t):
        assert len(t.description) > 50

    def test_nested_order_line_is_strict(self):
        args_schema = calculate_order.args_schema
        assert args_schema is not None and isinstance(args_schema, type)
        schema = cast(type[BaseModel], args_schema).model_json_schema()
        assert schema["$defs"]["OrderLine"]["additionalProperties"] is False

class TestValidationHandling:
    def test_tool_returns_error_text_instead_of_raising(self):
        out = search_items.invoke({"max_price": -5})
        assert isinstance(out, str) and out          # агент отримає текст, граф не впаде
        assert "max_price" in out                    # і побачить, яке саме поле хибне