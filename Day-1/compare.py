#!/usr/bin/env python3
"""Сравнение одного и того же запроса с разным уровнем контроля ответа.

Отправляет запрос дважды:
  1. профиль «свободный» — без ограничений;
  2. профиль «строгий»   — выбранный формат + предел слов + стоп-маркер END
и печатает оба ответа с метаданными (finish_reason, токены, число слов).

Использование:
    python compare.py                                    # демо-запрос, строгий = список-3 / 50 слов
    python compare.py "Свой вопрос"
    python compare.py --format шаги --words 200 "Вопрос"
"""

import argparse

from llm_client import (
    BASE_SYSTEM,
    DEFAULT_FORMAT,
    DEFAULT_WORDS,
    FORMATS,
    PROFILE_FREE,
    PROFILE_STRICT,
    LLMError,
    ask_full,
    resolve_profile,
)

DEMO_PROMPT = "Расскажи о пользе утренней зарядки."


def run(title: str, prompt: str, params: dict) -> None:
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
        f"completion_tokens={result.completion_tokens} "
        f"(из них reasoning={result.reasoning_tokens})"
    )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("prompt", nargs="*", help="запрос (по умолчанию — демо про утреннюю зарядку)")
    parser.add_argument(
        "--format", dest="fmt", choices=list(FORMATS), default=DEFAULT_FORMAT, metavar="ФОРМАТ",
        help=f"формат ответа для профиля «{PROFILE_STRICT}» (по умолчанию {DEFAULT_FORMAT})",
    )
    parser.add_argument(
        "--words", type=int, default=DEFAULT_WORDS, metavar="N",
        help=f"предел длины в словах для «{PROFILE_STRICT}» (по умолчанию {DEFAULT_WORDS})",
    )
    args = parser.parse_args()

    prompt = " ".join(args.prompt).strip() or DEMO_PROMPT
    print(f"Запрос: {prompt}\n")

    try:
        run(f"БЕЗ ОГРАНИЧЕНИЙ (профиль «{PROFILE_FREE}»)", prompt, resolve_profile(PROFILE_FREE))
        run(
            f"С ОГРАНИЧЕНИЯМИ (профиль «{PROFILE_STRICT}», формат={args.fmt}, слов≤{args.words})",
            prompt,
            resolve_profile(PROFILE_STRICT, args.fmt, args.words),
        )
    except LLMError as exc:
        print(f"Ошибка: {exc}")
        return 1

    print("=" * 70)
    print("Вывод: один и тот же запрос, разный уровень контроля через API.")
    print(f"  «{PROFILE_FREE}»  — длина и структура на усмотрение модели.")
    print(f"  «{PROFILE_STRICT}»   — формат и длина заданы пользователем, генерация")
    print("               завершается по stop-последовательности END.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
