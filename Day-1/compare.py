#!/usr/bin/env python3
"""Сравнение одного и того же запроса с разным уровнем контроля ответа.

Отправляет DEMO_PROMPT дважды:
  1. без ограничений         (профиль "free")
  2. с ограничениями         (профиль "structured": явный формат + max_tokens + stop)
и печатает оба ответа с метаданными (finish_reason, токены, число слов).

Использование:
    python compare.py                 # фиксированный демо-запрос
    python compare.py "Свой вопрос"   # тот же прогон на своём запросе
"""

import sys

from llm_client import BASE_SYSTEM, PROFILES, LLMError, ask_full

DEMO_PROMPT = "Расскажи о пользе утренней зарядки."


def run(title: str, prompt: str, profile_name: str) -> None:
    params = PROFILES[profile_name]
    result = ask_full(prompt, BASE_SYSTEM, **params)

    print("=" * 70)
    print(title)
    print("-" * 70)
    print("Параметры запроса:")
    print(f"  format_instruction : {'да' if params['format_instruction'] else 'нет'}")
    print(f"  max_tokens         : {params['max_tokens']}")
    print(f"  stop               : {params['stop']}")
    print("-" * 70)
    print(result.text)
    print("-" * 70)
    print(
        f"finish_reason={result.finish_reason!r}  "
        f"слов={result.word_count}  "
        f"символов={len(result.text)}  "
        f"completion_tokens={result.completion_tokens}"
    )
    print()


def main() -> int:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return 0

    prompt = " ".join(sys.argv[1:]).strip() or DEMO_PROMPT
    print(f"Запрос: {prompt}\n")

    try:
        run("БЕЗ ОГРАНИЧЕНИЙ (profile=free)", prompt, "free")
        run("С ОГРАНИЧЕНИЯМИ (profile=structured)", prompt, "structured")
    except LLMError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("Вывод: один и тот же запрос, разный уровень контроля через API.")
    print("  free       — длина и структура на усмотрение модели.")
    print("  structured — формат задан в промпте, длина ограничена max_tokens,")
    print("               генерация завершается по stop-последовательности.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
