#!/usr/bin/env python3
"""CLI: отправляет запрос в LLM и печатает ответ в консоль.

Использование:
    python cli.py "Ваш вопрос"     # разовый запрос
    python cli.py                   # интерактивный режим (Ctrl+C для выхода)
"""

import sys

from llm_client import LLMError, ask


def main() -> int:
    args = sys.argv[1:]

    if args:  # разовый запрос из аргументов командной строки
        try:
            print(ask(" ".join(args)))
        except LLMError as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            return 1
        return 0

    # Интерактивный режим
    print("LLM CLI. Введите вопрос и нажмите Enter. Ctrl+C — выход.\n")
    while True:
        try:
            prompt = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            return 0

        if not prompt:
            continue

        try:
            print(f"LLM: {ask(prompt)}\n")
        except LLMError as exc:
            print(f"Ошибка: {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
