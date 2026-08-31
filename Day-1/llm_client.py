"""Минимальный клиент к LLM DeepSeek (API совместим с OpenAI Chat Completions).

Единственная публичная функция — ask(): отправляет запрос и возвращает текст ответа.
Её переиспользуют и CLI (cli.py), и веб-интерфейс (web.py).
"""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()  # подхватываем переменные из .env, если файл есть

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
TIMEOUT = 60  # секунд, чтобы запрос не висел вечно


class LLMError(RuntimeError):
    """Понятная ошибка для верхнего уровня (CLI / веб)."""


def _api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise LLMError(
            "Не задан DEEPSEEK_API_KEY. Скопируйте .env.example в .env "
            "и впишите свой ключ (или экспортируйте переменную окружения)."
        )
    return key


def ask(prompt: str, system: str = "Ты полезный ассистент. Отвечай кратко и по делу.") -> str:
    """Отправляет prompt в LLM и возвращает текст ответа.

    Бросает LLMError с человекочитаемым сообщением при любой проблеме.
    """
    if not prompt or not prompt.strip():
        raise LLMError("Пустой запрос.")

    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {_api_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
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
        return response.json()["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, IndexError) as exc:
        raise LLMError(f"Неожиданный формат ответа API: {response.text[:500]}") from exc
