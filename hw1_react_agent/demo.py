# demo.py
import json

from pydantic import ValidationError

from tools import calculate_order, check_availability, get_item_details, search_items


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def ok(tool, args: dict) -> None:
    """Успішний виклик: друкуємо аргументи та красиво відформатований результат."""
    print(f"\n✅ {tool.name}({json.dumps(args, ensure_ascii=False)})")
    result = json.loads(tool.invoke(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def fail(tool, args: dict) -> None:
    print(f"\n❌ {tool.name}({json.dumps(args, ensure_ascii=False)})")
    try:
        tool.args_schema(**args)
        print("   !!! Помилки не було, а мала бути")
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            print(f"   Validation error (expected): {loc}: {err['msg']}")
    print(f"   Що побачить агент: {tool.invoke(args)}")

# --------------------------------------------------------------------------------------
def main() -> None:
    section("1. search_items")
    ok(search_items, {"category": "desserts"})
    ok(search_items, {"query": "гриби", "exclude_allergens": ["fish"]})
    ok(search_items, {"dietary_tags": ["vegan"], "max_price": 150, "limit": 5})
    ok(search_items, {"query": "піца"})                        # нічого не знайдено, але це не помилка
    fail(search_items, {"max_price": -5})
    fail(search_items, {"category": "pizza"})
    fail(search_items, {"limit": 100, "foo": "bar"})

    section("2. get_item_details")
    ok(get_item_details, {"item_id": " dish-007 "})            # нормалізація id
    ok(get_item_details, {"item_id": "DISH-999"})              # error у відповіді, не виняток
    fail(get_item_details, {"item_id": "123"})
    fail(get_item_details, {})

    section("3. check_availability")
    ok(check_availability, {"item_ids": ["DISH-001", "DISH-006", "DISH-999"]})
    ok(check_availability, {"item_ids": ["DISH-001", "dish-001"]})   # дублікат прибрано
    fail(check_availability, {"item_ids": []})
    fail(check_availability, {"item_ids": ["DISH-001", "abc"]})

    section("4. calculate_order")
    ok(calculate_order, {"items": [
        {"item_id": "DISH-010", "quantity": 2},
        {"item_id": "DISH-011", "quantity": 1},
    ]})
    ok(calculate_order, {"items": [{"item_id": "DISH-006", "quantity": 1}]})   # недоступна
    fail(calculate_order, {"items": []})
    fail(calculate_order, {"items": [{"item_id": "DISH-010", "quantity": 0}]})
    fail(calculate_order, {"items": [{"item_id": "DISH-010", "quantity": 1, "note": "x"}]})


if __name__ == "__main__":
    main()