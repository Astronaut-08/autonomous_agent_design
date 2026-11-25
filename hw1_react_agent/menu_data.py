import json
import re

from pathlib import Path
from typing import Literal

MENU = json.loads((Path(__file__).parent / "mock_data.json").read_text(encoding="utf-8"))['menu']
MENU_BY_ID: dict[str, dict] = {dish['id']: dish for dish in MENU}

CategoryType = Literal["soups", "salads", "mains", "desserts", "drinks"]
DietaryTagType = Literal["vegan", "vegetarian", "gluten_free", "lactose_free", "high_protein"]
AllergenType = Literal["gluten", "dairy", "eggs", "fish", "sesame"]

ALLERGEN_PATTERNS: dict[str, re.Pattern] = {
    "gluten": re.compile(r"\b(глютен|пшениц|борошн|тісто|пампушк|спагеті|макарон)"),
    "dairy":  re.compile(r"\b(молок|вершк|сметан|пармезан|морозив|пломбір|сир(а|у|ом)?\b|вершков\w* масл)"),
    "eggs":   re.compile(r"\b(яйц|яєчн|жовтк)"),
    "fish":   re.compile(r"\b(риб|лосос|туне|скумбр)"),
    "sesame": re.compile(r"\b(кунжут|тахіні)"),
}
ITEM_ID_RE = re.compile(r"^DISH-\d{3}$")


# --- допоміжні функції ---------------------------------------
def find_dish(item_id: str) -> dict | None:
    return MENU_BY_ID.get(item_id)

def has_excluded_allergen(dish: dict, excluded: list[AllergenType] | None) -> bool:
    if not excluded:
        return False

    dish_allergens = {a.lower() for a in dish.get("allergens", [])}
    ingredients_text = " ".join(dish.get("ingredients", [])).lower()
    tags = set(dish.get("dietary_tags", []))

    for allergen in excluded:
        if allergen in dish_allergens:
            return True
        if allergen == "dairy" and "lactose_free" in tags:
            continue  # склад не скануємо: «кокосове молоко» не є молочним алергеном
        if ALLERGEN_PATTERNS[allergen].search(ingredients_text):
            return True
    return False

def normalize_item_id(value: str) -> str:
    """' dish-001 ' -> 'DISH-001'; кидає ValueError, якщо формат неправильний."""
    value = value.strip().upper()
    if not ITEM_ID_RE.match(value):
        raise ValueError(
            f"Невірний формат id '{value}'. Очікується DISH-001 (DISH- і три цифри). "
            "Отримати id можна через search_items."
        )
    return value

MAX_QTY_PER_ITEM = 20