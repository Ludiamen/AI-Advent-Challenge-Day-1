#!/usr/bin/env python3
"""Простой веб-интерфейс: одно поле ввода, кнопка, ответ LLM на странице.

Запуск:
    python web.py
Затем открыть http://127.0.0.1:5000
"""

from flask import Flask, render_template, request

from llm_client import LLMError, ask

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    answer = None
    error = None

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        try:
            answer = ask(prompt)
        except LLMError as exc:
            error = str(exc)

    return render_template("index.html", prompt=prompt, answer=answer, error=error)


if __name__ == "__main__":
    # debug=False: не отдаём отладчик и трассировки наружу.
    app.run(host="127.0.0.1", port=5000, debug=False)
