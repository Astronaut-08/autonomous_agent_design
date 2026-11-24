from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator


# search_items, get_item_details, check_availability, calculate_order

# ── Pydantic-схема для параметрів tool ───────────────────────────
class WeatherInput(BaseModel):
    """Параметри для запиту пошуку страви."""
    city: str = Field(description='Назва міста (наприклад, "Kyiv")')
    days: int = Field(default=1, description='Кількість днів прогнозу (1-7)')
 
    @field_validator('city')
    @classmethod
    def city_not_empty(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError('Назва міста повинна містити мінімум 2 символи')
        return v.strip()
 
    @field_validator('days')
    @classmethod
    def days_in_range(cls, v: int) -> int:
        if not 1 <= v <= 7:
            raise ValueError('Кількість днів має бути від 1 до 7')
        return v
 
# ── Tool з Pydantic-схемою ───────────────────────────────────────
@tool(args_schema=WeatherInput)
def get_weather(city: str, days: int = 1) -> str:
    """Отримати прогноз погоди для міста.
 
    Використовуйте цей інструмент, коли потрібна інформація
    про погоду в конкретному місті.
 
    Args:
        city: Назва міста.
        days: Кількість днів прогнозу (1-7).
 
    Returns:
        Рядок з прогнозом погоди.
    """
    # Mock-дані (у реальному проєкті — API-виклик)
    weather_data = {
        'Kyiv': {'temp': 15, 'condition': 'Хмарно'},
        'Lviv': {'temp': 12, 'condition': 'Дощ'},
        'Odesa': {'temp': 18, 'condition': 'Сонячно'},
    }
    data = weather_data.get(city, {'temp': 20, 'condition': 'Ясно'})
    return f'{city}: {data["temp"]}°C, {data["condition"]} (прогноз на {days} дн.)'
 
# ── Unit test ────────────────────────────────────────────────────
# Валідація Pydantic працює:
try:
    WeatherInput(city='', days=1)  # Повинно впасти
except Exception as e:
    print(f'Validation error (expected): {e}')
 
# Tool працює:
result = get_weather.invoke({'city': 'Kyiv', 'days': 3})
print(f'Weather: {result}')
