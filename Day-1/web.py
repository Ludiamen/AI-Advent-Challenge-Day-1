#!/usr/bin/env python3
"""Простой веб-интерфейс: поле ввода, выбор профиля контроля ответа, ответ LLM.

Запуск:
    python web.py
Затем открыть http://127.0.0.1:5000
"""

from flask import Flask, render_template, request

from llm_client import PROFILES, LLMError, ask_full

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    profile = "free"
    result = None
    error = None

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        profile = request.form.get("profile", "free")
        if profile not in PROFILES:
            profile = "free"
        try:
            result = ask_full(prompt, **PROFILES[profile])
        except LLMError as exc:
            error = str(exc)

    return render_template(
        "index.html",
        prompt=prompt,
        profile=profile,
        profiles=sorted(PROFILES),
        result=result,
        error=error,
    )


if __name__ == "__main__":
    # debug=False: не отдаём отладчик и трассировки наружу.
    app.run(host="127.0.0.1", port=5000, debug=False)
