#!/usr/bin/env python3
"""Простой веб-интерфейс: запрос, выбор модели / профиля / формата / длины ответа.

Запуск:
    python web.py
Затем открыть http://127.0.0.1:5000

Работа с формой описана прямо на странице в блоке «❓ Справка по форме».
"""

from flask import Flask, render_template, request

from llm_client import (
    DEFAULT_FORMAT,
    DEFAULT_MODEL,
    DEFAULT_WORDS,
    FORMATS,
    LENGTH_PRESETS,
    MODELS,
    PROFILE_FREE,
    PROFILE_STRICT,
    LLMError,
    ask_full,
    resolve_profile,
)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    form = {
        "prompt": "",
        "model": DEFAULT_MODEL,
        "profile": PROFILE_FREE,
        "fmt": DEFAULT_FORMAT,
        "words_preset": str(DEFAULT_WORDS),
        "words_custom": "",
    }
    result = None
    error = None

    if request.method == "POST":
        for key in form:
            form[key] = request.form.get(key, form[key]).strip()

        # Своё число слов, если задано, переопределяет пресет.
        words = form["words_custom"] or form["words_preset"]
        model = form["model"] if form["model"] in MODELS else DEFAULT_MODEL

        try:
            params = resolve_profile(form["profile"], form["fmt"], words)
            result = ask_full(form["prompt"], model=model, **params)
        except LLMError as exc:
            error = str(exc)

    return render_template(
        "index.html",
        form=form,
        models=MODELS,
        profiles=(PROFILE_FREE, PROFILE_STRICT),
        strict_profile=PROFILE_STRICT,
        formats=FORMATS,
        length_presets=LENGTH_PRESETS,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    # debug=False: не отдаём отладчик и трассировки наружу.
    app.run(host="127.0.0.1", port=5000, debug=False)
