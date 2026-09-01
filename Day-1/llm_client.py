"""Минимальный клиент к LLM DeepSeek (API совместим с OpenAI Chat Completions).

Публичное API:
  ask(prompt, ...)                    -> str        — только текст (используют cli.py, web.py)
  ask_full(prompt, ...)              -> LLMResult   — текст + finish_reason + расход токенов
  PROFILES                          -> dict        — профиль «свободный» (без ограничений)
  FORMATS                           -> dict        — варианты формата для профиля «строгий»
  LENGTH_PRESETS                    -> dict        — пресеты длины ответа в словах
  build_structured_profile(fmt, n)  -> dict        — собрать «строгий» профиль

Профиль — это набор параметров, задающих «уровень контроля» над ответом:
  * format_instruction — явное описание формата (добавляется к system-промпту);
  * max_tokens         — жёсткое ограничение длины на стороне API;
  * stop               — стоп-последовательности (условие завершения ответа).

Два профиля:
  «свободный» — без ограничений, модель отвечает как хочет;
  «строгий»   — собирается под выбор пользователя: формат (FORMATS) и предельная
                длина в словах (пресеты LENGTH_PRESETS или произвольное число).
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

# Названия профилей (они же — значения флага --profile и поля формы).
PROFILE_FREE = "свободный"
PROFILE_STRICT = "строгий"

# Профиль без ограничений — модель отвечает как хочет.
PROFILES: dict[str, dict] = {
    PROFILE_FREE: {
        "format_instruction": "",
        "max_tokens": None,
        "stop": None,
    },
}

# Варианты формата для профиля «строгий»: ключ -> {название, фрагмент инструкции}.
FORMATS: dict[str, dict[str, str]] = {
    "список-3": {
        "label": "Список из 3 пунктов",
        "instruction": (
            "в виде маркированного списка ровно из 3 пунктов; каждый пункт — с новой "
            "строки, начинается с «- », одно предложение"
        ),
    },
    "список-5": {
        "label": "Список из 5 пунктов",
        "instruction": (
            "в виде маркированного списка ровно из 5 пунктов; каждый пункт — с новой "
            "строки, начинается с «- »"
        ),
    },
    "шаги": {
        "label": "Нумерованные шаги",
        "instruction": "в виде нумерованного списка шагов (1., 2., 3. …); каждый шаг — с новой строки",
    },
    "абзац": {
        "label": "Один абзац",
        "instruction": "в виде одного связного абзаца, без списков, заголовков и переносов строк",
    },
    "резюме": {
        "label": "Одно предложение (TL;DR)",
        "instruction": "в виде одного ёмкого предложения-резюме",
    },
    "вопрос-ответ": {
        "label": "Пары «вопрос — ответ»",
        "instruction": "в формате «Вопрос: … / Ответ: …», 2–3 пары",
    },
}

# Пресеты длины ответа: число слов -> подпись для интерфейса.
# Пользователь может указать и своё число (1..MAX_WORDS).
LENGTH_PRESETS: dict[int, str] = {
    10: "10 слов — очень кратко",
    50: "50 слов — кратко",
    200: "200 слов — развёрнуто",
}

DEFAULT_PROFILE = PROFILE_FREE
DEFAULT_FORMAT = "список-3"
DEFAULT_WORDS = 50
MAX_WORDS = 2000        # верхняя граница вменяемого запроса
REASONING_BUDGET = 900  # запас токенов на скрытое рассуждение модели (см. build_structured_profile)


def build_structured_profile(fmt: str = DEFAULT_FORMAT, words: int = DEFAULT_WORDS) -> dict:
    """Собирает «строгий» профиль под выбор пользователя.

    fmt   — ключ из FORMATS (описание формата ответа);
    words — предельная длина ответа в словах (>= 1).

    Все три рычага контроля включаются вместе:
      * формат        — явной инструкцией в system-промпте;
      * длина         — и в инструкции («не более N слов»), и жёстко в max_tokens;
      * завершение    — стоп-последовательностью END плюс требование её напечатать.
    """
    if fmt not in FORMATS:
        raise LLMError(f"Неизвестный формат '{fmt}'. Доступны: {', '.join(FORMATS)}.")
    try:
        words = int(words)
    except (TypeError, ValueError):
        raise LLMError("Число слов должно быть целым.") from None
    if not 1 <= words <= MAX_WORDS:
        raise LLMError(f"Число слов должно быть от 1 до {MAX_WORDS}.")

    instruction = (
        f"Ответь строго {FORMATS[fmt]['instruction']}. "
        f"Весь ответ — не более {words} слов. Без вступления и заключения. "
        f"Закончив ответ, с новой строки напиши {END_MARKER} и больше ничего."
    )
    return {
        "format_instruction": instruction,
        # deepseek-v4-flash — рассуждающая модель: скрытые reasoning-токены тоже
        # тратят бюджет max_tokens. Поэтому кэп = запас на рассуждение (REASONING_BUDGET)
        # + грубая оценка ответа. Реальную длину и форму держит инструкция в промпте,
        # а генерацию завершает стоп-маркер END; max_tokens — только предохранитель.
        "max_tokens": REASONING_BUDGET + words * 6,
        "stop": [END_MARKER],
    }


class LLMError(RuntimeError):
    """Понятная ошибка для верхнего уровня (CLI / веб)."""


@dataclass
class LLMResult:
    """Результат запроса с метаданными для сравнения профилей."""

    text: str
    finish_reason: str = ""            # "stop", "length", ...
    prompt_tokens: int = 0
    completion_tokens: int = 0         # включает reasoning_tokens
    reasoning_tokens: int = 0          # скрытое рассуждение (deepseek-v4-flash)
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
        text = choice["message"]["content"].strip()
        finish_reason = choice.get("finish_reason", "")
        reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    except (ValueError, KeyError, IndexError) as exc:
        raise LLMError(f"Неожиданный формат ответа API: {response.text[:500]}") from exc

    if not text:
        raise LLMError(
            "Модель вернула пустой ответ"
            + (f" (finish_reason={finish_reason})" if finish_reason else "")
            + (
                f": весь бюджет max_tokens={max_tokens} ушёл на рассуждение "
                f"({reasoning_tokens} токенов). Увеличьте лимит слов."
                if finish_reason == "length"
                else "."
            )
        )

    return LLMResult(
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        reasoning_tokens=reasoning_tokens,
        request_params={
            "max_tokens": max_tokens,
            "stop": stop,
            "format_instruction": bool(format_instruction),
        },
    )


def resolve_profile(profile: str, fmt: str = DEFAULT_FORMAT, words: int = DEFAULT_WORDS) -> dict:
    """Возвращает параметры запроса по имени профиля.

    «свободный» — без ограничений;
    «строгий»   — формат fmt + предел words слов + стоп-маркер END.
    """
    if profile == PROFILE_FREE:
        return PROFILES[PROFILE_FREE]
    if profile == PROFILE_STRICT:
        return build_structured_profile(fmt, words)
    raise LLMError(
        f"Неизвестный профиль '{profile}'. Доступны: {PROFILE_FREE}, {PROFILE_STRICT}."
    )


def ask(
    prompt: str,
    system: str = BASE_SYSTEM,
    *,
    profile: str = DEFAULT_PROFILE,
    fmt: str = DEFAULT_FORMAT,
    words: int = DEFAULT_WORDS,
) -> str:
    """Упрощённая обёртка: возвращает только текст ответа."""
    return ask_full(prompt, system, **resolve_profile(profile, fmt, words)).text
