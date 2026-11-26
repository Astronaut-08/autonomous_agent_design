import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, ConfigDict
from menu_data import (
    CategoryType, AllergenType, DietaryTagType, 
    MAX_QTY_PER_ITEM, normalize_item_id, has_excluded_allergen, 
    MENU, find_dish
)

def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)

# ── Pydantic-схеми для параметрів tool ───────────────────────────
class StrictModel(BaseModel):
    """Базова модель яка забороняє додаткові поля для LMM"""
    model_config = ConfigDict(extra='forbid')

class SearchItemsInput(StrictModel):
    """Параметри пошуку страв у меню."""
    
    query: str | None = Field(
        default=None,
        description=(
        "Ключові слова з НАЗВИ страви або ІНГРЕДІЄНТІВ (наприклад 'лосось', 'гриби'). "
        "НЕ вказуйте тут тип страви (десерт, напій, салат), дієту, ціну чи алерген: "
        "для цього є окремі параметри."
        )
    )
    category: CategoryType | None = Field(
        default=None,
        description=(
        "Тип страви: десерти → desserts, напої → drinks, супи → soups, "
        "салати → salads, основні/гарячі страви → mains."
        )
    )
    dietary_tags: list[DietaryTagType] | None = Field(
        default=None,
        description=(
        "Обов'язкові дієтичні теги: веганське → vegan, вегетаріанське → vegetarian, "
        "безглютенове → gluten_free, безлактозне → lactose_free, багато білка → high_protein. "
        "Страва має містити УСІ вказані теги."
        )
    )
    exclude_allergens: list[AllergenType] | None = Field(
        default=None,
        description='Алергени, яких НЕ повинно бути ні в полі allergens, ні в інгредієнтах.'
    )
    max_price: float | None = Field(
        default=None,
        description='Максимальна ціна за порцію в гривнях (UAH).',
        gt=0
    )
    max_prep_time_min: int | None = Field(
        default=None,
        description='Максимальний час приготування у хвилинах.',
        gt=0
    )
    only_available: bool = Field(
        default=True,
        description='Якщо True — повертати лише страви з available=True.'
    )
    limit: int = Field(
        default=3,
        ge=1,
        le=20,
        description='Максимальна кількість позицій у результаті.'
    )
 
    @field_validator("query")
    @classmethod
    def clean_query(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None  # порожній рядок = «без запиту»

    @field_validator("dietary_tags", "exclude_allergens")
    @classmethod
    def dedupe(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        return list(dict.fromkeys(v))  # прибирає дублі, зберігає порядок

class ItemDetailsInput(StrictModel):
    """Параметри для отримання повної інформації про одну страву."""

    item_id: str = Field(
        description="Ідентифікатор страви у форматі DISH-001 (id беруть із результатів search_items).",
        examples=["DISH-001"],
    )

    @field_validator("item_id")
    @classmethod
    def check_item_id(cls, v: str) -> str:
        return normalize_item_id(v)

class CheckAvailabilityInput(StrictModel):
    """Параметри для перевірки наявності кількох страв."""

    item_ids: list[str] = Field(
        min_length=1,
        max_length=10,
        description="Список id страв для перевірки наявності (від 1 до 10), наприклад ['DISH-001', 'DISH-006'].",
    )

    @field_validator("item_ids")
    @classmethod
    def check_item_ids(cls, v: list[str]) -> list[str]:
        normalized = [normalize_item_id(i) for i in v]
        return list(dict.fromkeys(normalized))

class OrderLine(StrictModel):
    """Одна позиція замовлення."""

    item_id: str = Field(description="Ідентифікатор страви, наприклад DISH-010.")
    quantity: int = Field(
        default=1, ge=1, le=MAX_QTY_PER_ITEM,
        description=f"Кількість порцій (від 1 до {MAX_QTY_PER_ITEM}).",
    )

    @field_validator("item_id")
    @classmethod
    def check_item_id(cls, v: str) -> str:
        return normalize_item_id(v)

class CalculateOrderInput(StrictModel):
    """Параметри для розрахунку вартості замовлення."""

    items: list[OrderLine] = Field(
        min_length=1,
        max_length=15,
        description="Позиції замовлення: список об'єктів {item_id, quantity}.",
    )

    @field_validator("items")
    @classmethod
    def merge_duplicates(cls, v: list[OrderLine]) -> list[OrderLine]:
        totals: dict[str, int] = {}
        for line in v:
            totals[line.item_id] = totals.get(line.item_id, 0) + line.quantity

        too_many = [i for i, q in totals.items() if q > MAX_QTY_PER_ITEM]
        if too_many:
            raise ValueError(
                f"Для {', '.join(too_many)} сумарна кількість перевищує {MAX_QTY_PER_ITEM}."
            )
        return [OrderLine(item_id=i, quantity=q) for i, q in totals.items()]

# ── Tool з Pydantic-схемою ───────────────────────────────────────
CATEGORY_UA = {
    "soups": "супи суп", "salads": "салати салат", "mains": "основні гарячі страви",
    "desserts": "десерти десерт", "drinks": "напої напій",
}

def _matches_query(dish: dict, query: str) -> bool:
    """Кожне слово запиту має знайтися в назві, категорії або складі."""
    searchable = " ".join([
        dish.get("name", ""),
        dish.get("category", ""),
        CATEGORY_UA.get(dish["category"], ""),
        " ".join(dish.get("ingredients", [])),
    ]).lower()

    for word in query.lower().split():
        stem = word[:-2] if len(word) > 5 else word  # «лосося» -> «лосо» знайде «лосось»
        if stem not in searchable:
            return False
    return True


@tool(args_schema=SearchItemsInput)
def search_items(
    query: str | None = None,
    category: CategoryType | None = None,
    dietary_tags: list[DietaryTagType] | None = None,
    exclude_allergens: list[AllergenType] | None = None,
    max_price: float | None = None,
    max_prep_time_min: int | None = None,
    only_available: bool = True,
    limit: int = 3,
) -> str:
    """Знайти страви в меню за ключовими словами, категорією, дієтою, алергенами, ціною та часом приготування.

    Використовуйте цей інструмент, коли клієнт шукає, що замовити, або просить
    порекомендувати страву. Це перший крок: він повертає id страв, які потім
    можна передати в get_item_details, check_availability та calculate_order.

    Повертає JSON: {"found": N, "items": [...]}, де кожна позиція має id, name,
    category, price (грн), available, dietary_tags, allergens, prep_time_min.
    Результати відсортовані за зростанням ціни.
    """
    required_tags = set(dietary_tags or [])
    results, hidden = [], []
    for dish in MENU:
        if category and dish["category"] != category:
            continue
        if required_tags and not required_tags.issubset(dish.get("dietary_tags", [])):
            continue
        if has_excluded_allergen(dish, exclude_allergens):
            continue
        if max_price is not None and dish["price"] > max_price:
            continue
        if max_prep_time_min is not None and dish["prep_time_min"] > max_prep_time_min:
            continue
        if query and not _matches_query(dish, query):
            continue
        if only_available and not dish["available"]:   # остання перевірка
            hidden.append({"id": dish["id"], "name": dish["name"]})
            continue
        results.append(dish)

    results.sort(key=lambda d: d["price"])
    items = [
        {k: d[k] for k in ("id", "name", "category", "price", "available",
                           "dietary_tags", "allergens", "prep_time_min")}
        for d in results[:limit]
    ]

    response = {"found": len(items), "items": items}
    if hidden:
        response["unavailable_matches"] = hidden
        response["note"] = "Ці страви підходять під запит, але зараз недоступні: повідом про це клієнту."
    if not items:
        response["message"] = (
            "За заданими критеріями страв не знайдено. Перевірте параметри: "
            "тип страви задається в category, а не в query."
        )
    return _json(response)

@tool(args_schema=ItemDetailsInput)
def get_item_details(item_id: str) -> str:
    """Отримати повну інформацію про одну страву за її id.

    Використовуйте, коли id страви вже відомий (зазвичай зі search_items) і клієнт
    питає про склад, вагу, алергени, час приготування або ціну конкретної страви.

    Повертає JSON з усіма полями страви: id, name, category, price, weight_g,
    prep_time_min, available, dietary_tags, allergens, ingredients.
    Якщо страви з таким id немає, повертає {"error": ..., "hint": ...}.
    """
    dish = find_dish(item_id)
    if dish is None:
        return _json({
            "error": f"Страву {item_id} не знайдено в меню.",
            "hint": "Скористайтеся search_items, щоб знайти правильний id.",
        })
    return _json(dish)

@tool(args_schema=CheckAvailabilityInput)
def check_availability(item_ids: list[str]) -> str:
    """Перевірити, чи доступні зараз для замовлення вказані страви.

    Використовуйте перед формуванням замовлення або коли клієнт питає, чи є
    певна страва. Приймає від 1 до 10 id.

    Повертає JSON: {"all_available": bool, "items": [{"id", "name", "status"}]},
    де status це "available", "unavailable" (страва є в меню, але її зараз
    немає) або "not_found" (такого id немає в меню).
    """
    items = []
    for item_id in item_ids:
        dish = find_dish(item_id)
        if dish is None:
            items.append({"id": item_id, "name": None, "status": "not_found"})
        else:
            status = "available" if dish["available"] else "unavailable"
            items.append({"id": item_id, "name": dish["name"], "status": status})

    all_available = all(i["status"] == "available" for i in items)
    return _json({"all_available": all_available, "items": items})

@tool(args_schema=CalculateOrderInput)
def calculate_order(items: list[OrderLine]) -> str:
    """Порахувати вартість замовлення за списком страв і кількостей.

    Використовуйте, коли клієнт вирішив, що замовляє, і хоче дізнатися суму та
    орієнтовний час приготування. Передавайте позиції як {"item_id", "quantity"}.
    Дублікати одного id об'єднуються автоматично.

    Повертає JSON: {"lines": [{"id", "name", "quantity", "unit_price",
    "line_total"}], "total", "currency": "UAH", "estimated_prep_time_min"}.
    Якщо якась страва не знайдена або недоступна, замовлення не рахується, а
    повертається {"error": ..., "not_found": [...], "unavailable": [...]}.
    """
    lines = [l if isinstance(l, OrderLine) else OrderLine(**l) for l in items]

    not_found, unavailable, priced = [], [], []
    for line in lines:
        dish = find_dish(line.item_id)
        if dish is None:
            not_found.append(line.item_id)
        elif not dish["available"]:
            unavailable.append(line.item_id)
        else:
            priced.append((line, dish))

    if not_found or unavailable:
        return _json({
            "error": "Замовлення не можна порахувати.",
            "not_found": not_found,
            "unavailable": unavailable,
            "hint": "Приберіть ці позиції або підберіть заміну через search_items.",
        })

    result_lines = [
        {
            "id": dish["id"],
            "name": dish["name"],
            "quantity": line.quantity,
            "unit_price": dish["price"],
            "line_total": dish["price"] * line.quantity,
        }
        for line, dish in priced
    ]
    return _json({
        "lines": result_lines,
        "total": sum(l["line_total"] for l in result_lines),
        "currency": "UAH",
        "estimated_prep_time_min": max(d["prep_time_min"] for _, d in priced),
    })


TOOLS = [search_items, get_item_details, check_availability, calculate_order]

def _format_validation_error(e) -> str:
    parts = [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
    return "Невалідні параметри виклику, виправте та повторіть: " + "; ".join(parts)

for t in TOOLS:
    t.handle_validation_error = _format_validation_error