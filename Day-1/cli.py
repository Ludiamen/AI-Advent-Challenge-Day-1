#!/usr/bin/env python3
"""CLI: отправляет запрос в LLM и печатает ответ в консоль.

Использование:
    python cli.py "Ваш вопрос"                     # разовый запрос, профиль free
    python cli.py --profile structured "Вопрос"    # с ограничениями формата/длины
    python cli.py                                  # интерактивный режим (Ctrl+C — выход)

Профили (уровень контроля ответа) описаны в llm_client.PROFILES:
    free       — без ограничений;
    structured — явный формат + max_tokens + stop-последовательность.
"""

import argparse
import sys

from llm_client import PROFILES, LLMError, ask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Запрос к LLM DeepSeek из консоли.")
    parser.add_argument("prompt", nargs="*", help="текст запроса; без него — интерактивный режим")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="free",
        help="уровень контроля ответа (по умолчанию: free)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prompt = " ".join(args.prompt).strip()

    if prompt:  # разовый запрос
        try:
            print(ask(prompt, profile=args.profile))
        except LLMError as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            return 1
        return 0

    # Интерактивный режим
    print(f"LLM CLI (профиль: {args.profile}). Введите вопрос и нажмите Enter. Ctrl+C — выход.\n")
    while True:
        try:
            prompt = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            return 0

        if not prompt:
            continue

        try:
            print(f"LLM: {ask(prompt, profile=args.profile)}\n")
        except LLMError as exc:
            print(f"Ошибка: {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
