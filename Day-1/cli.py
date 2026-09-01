#!/usr/bin/env python3
"""CLI: отправляет запрос в LLM и печатает ответ в консоль.

Быстрый старт:
    python cli.py "Ваш вопрос"                       # свободный ответ
    python cli.py --profile строгий --format список-5 --words 10 "Вопрос"
    python cli.py                                    # интерактивный режим

Полная справка с примерами: python cli.py --help
"""

import argparse
import sys

from llm_client import (
    DEFAULT_FORMAT,
    DEFAULT_WORDS,
    FORMATS,
    MAX_WORDS,
    PROFILE_FREE,
    PROFILE_STRICT,
    LLMError,
    ask,
)

# --- справка -----------------------------------------------------------------

_FORMATS_HELP = "\n".join(f"  {key:<13} — {meta['label']}" for key, meta in FORMATS.items())

EPILOG = f"""\
профили (--profile):
  {PROFILE_FREE:<13} — без ограничений, модель отвечает как хочет
  {PROFILE_STRICT:<13} — заданный формат (--format) + предел длины (--words)
                  + завершение по стоп-маркеру END

форматы (--format), действуют только с «--profile {PROFILE_STRICT}»:
{_FORMATS_HELP}

длина (--words): любое число 1..{MAX_WORDS}; типичные значения 10 / 50 / 200

примеры:
  # свободный ответ
  python cli.py "Расскажи о пользе утренней зарядки"

  # строгий: 5 пунктов, максимум 10 слов
  python cli.py --profile {PROFILE_STRICT} --format список-5 --words 10 "Чем полезна ходьба?"

  # строгий: один абзац до 200 слов
  python cli.py --profile {PROFILE_STRICT} --format абзац --words 200 "Чем полезна ходьба?"

  # строгий: одно предложение-резюме, своё число слов
  python cli.py --profile {PROFILE_STRICT} --format резюме --words 25 "Чем полезна ходьба?"

  # интерактивный режим со строгим профилем (флаги задаются один раз при запуске)
  python cli.py --profile {PROFILE_STRICT} --format шаги --words 120
  # затем в диалоге: «?» или «справка» — показать эту памятку, Ctrl+C — выход
"""

INTERACTIVE_HELP = f"""\
Команды диалога:
  ?  /  справка   — показать эту памятку
  Ctrl+C          — выход
Профиль и формат заданы при запуске флагами --profile / --format / --words.
"""

# --- логика ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Запрос к LLM DeepSeek из консоли.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", nargs="*", help="текст запроса; без него — интерактивный режим")
    parser.add_argument(
        "--profile",
        choices=(PROFILE_FREE, PROFILE_STRICT),
        default=PROFILE_FREE,
        help=f"уровень контроля ответа (по умолчанию: {PROFILE_FREE})",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=list(FORMATS),
        default=DEFAULT_FORMAT,
        metavar="ФОРМАТ",
        help=f"формат ответа для профиля «{PROFILE_STRICT}» (по умолчанию: {DEFAULT_FORMAT})",
    )
    parser.add_argument(
        "--words",
        type=int,
        default=DEFAULT_WORDS,
        metavar="N",
        help=f"предел длины ответа в словах для «{PROFILE_STRICT}» (по умолчанию: {DEFAULT_WORDS})",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prompt = " ".join(args.prompt).strip()
    kw = {"profile": args.profile, "fmt": args.fmt, "words": args.words}

    if prompt:  # разовый запрос
        try:
            print(ask(prompt, **kw))
        except LLMError as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            return 1
        return 0

    # Интерактивный режим
    tail = "" if args.profile == PROFILE_FREE else f", формат: {args.fmt}, слов: ≤{args.words}"
    print(f"LLM CLI (профиль: {args.profile}{tail}).")
    print("Введите вопрос. «?» — справка, Ctrl+C — выход.\n")
    while True:
        try:
            prompt = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            return 0

        if not prompt:
            continue
        if prompt.lower() in ("?", "справка", "help"):
            print(INTERACTIVE_HELP)
            continue

        try:
            print(f"LLM: {ask(prompt, **kw)}\n")
        except LLMError as exc:
            print(f"Ошибка: {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
