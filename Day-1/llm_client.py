"""Минимальный клиент к LLM DeepSeek (API совместим с OpenAI Chat Completions).

Публичное API:
  ask(prompt, ...)       -> str            — только текст ответа (используют cli.py, web.py)
  ask_full(prompt, ...)  -> LLMResult      — текст + finish_reason + расход токенов
  PROFILES               -> dict           — готовые профили контроля ответа

Профиль — это набор параметров, задающих «уровень контроля» над ответом:
  * format_instruction — явное описание формата (добавляется к system-промпту);
  * max_tokens         — жёсткое ограничение длины на стороне API;
  * stop               — стоп-последовательности (условие завершения ответа).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv

load_dotenv()  # подхватываем переменные из .env, если файл есть

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
TIMEOUT = 60  # секунд, чтобы запрос не висел вечно

BASE_SYSTEM = "Ты полезный ассистент. Отвечай кратко и по делу."

# Маркер завершения ответа для «ограниченного» профиля: модель обязана его напечатать,
# а API обрывает генерацию на нём (stop), поэтому в готовом тексте он не виден.
END_MARKER = "END"

PROFILES: dict[str, dict] = {
    # Без ограничений — модель отвечает как хочет.
    "free": {
        "format_instruction": "",
        "max_tokens": None,
        "stop": None,
    },
    # С ограничениями: явный формат + предел длины + условие завершения.
    "structured": {
        "format_instruction": (
            "Формат ответа строго такой: маркированный список ровно из 3 пунктов, "
            "каждый пункт — с новой строки и начинается с «- », одно предложение. "
            "Весь ответ — не более 45 слов. Без вступления и заключения. "
            f"После третьего пункта с новой строки напиши {END_MARKER} и больше ничего."
        ),
        # max_tokens — жёсткий предохранитель (≈ в 3 раза короче типичного free-ответа);
        # основную форму задаёт инструкция, а обрывает генерацию stop-маркер END.
        "max_tokens": 220,
        "stop": [END_MARKER],
    },
}


class LLMError(RuntimeError):
    """Понятная ошибка для верхнего уровня (CLI / веб)."""


@dataclass
class LLMResult:
    """Результат запроса с метаданными для сравнения профилей."""

    text: str
    finish_reason: str = ""            # "stop", "length", ...
    prompt_tokens: int = 0
    completion_tokens: int = 0
    request_params: dict = field(default_factory=dict)  # что реально ушло в API

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise LLMError(
            "Не задан DEEPSEEK_API_KEY. Скопируйте .env.example в .env "
            "и впишите свой ключ (или экспортируйте переменную окружения)."
        )
    return key


def ask_full(
    prompt: str,
    system: str = BASE_SYSTEM,
    *,
    format_instruction: str = "",
    max_tokens: int | None = None,
    stop: list[str] | None = None,
) -> LLMResult:
    """Отправляет prompt в LLM с заданным уровнем контроля и возвращает LLMResult.

    Бросает LLMError с человекочитаемым сообщением при любой проблеме.
    """
    if not prompt or not prompt.strip():
        raise LLMError("Пустой запрос.")

    system_content = system
    if format_instruction:
        system_content = f"{system}\n\n{format_instruction}"

    payload: dict = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if stop:
        payload["stop"] = stop

    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {_api_key()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:  # сеть недоступна, таймаут и т.п.
        raise LLMError(f"Сетевая ошибка при обращении к API: {exc}") from exc

    if response.status_code == 401:
        raise LLMError("API отклонил ключ (401). Проверьте DEEPSEEK_API_KEY.")
    if not response.ok:
        # Тело ответа может содержать подсказку, но не ключ — выводить безопасно.
        raise LLMError(f"API вернул {response.status_code}: {response.text[:500]}")

    try:
        data = response.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return LLMResult(
            text=choice["message"]["content"].strip(),
            finish_reason=choice.get("finish_reason", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            request_params={
                "max_tokens": max_tokens,
                "stop": stop,
                "format_instruction": bool(format_instruction),
            },
        )
    except (ValueError, KeyError, IndexError) as exc:
        raise LLMError(f"Неожиданный формат ответа API: {response.text[:500]}") from exc


def ask(prompt: str, system: str = BASE_SYSTEM, *, profile: str = "free") -> str:
    """Упрощённая обёртка: возвращает только текст ответа.

    profile — ключ из PROFILES ("free" без ограничений, "structured" с ограничениями).
    """
    if profile not in PROFILES:
        raise LLMError(f"Неизвестный профиль '{profile}'. Доступны: {', '.join(PROFILES)}.")
    return ask_full(prompt, system, **PROFILES[profile]).text
