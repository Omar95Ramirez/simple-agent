from simple_agent.core import extract_url, guess_url, summarize_html


def test_extract_url():
    task = "Summarize https://example.com/test"
    assert extract_url(task) == "https://example.com/test"


def test_guess_url_python():
    task = "Tell me about Python"
    assert guess_url(task) == "https://www.python.org/"


def test_guess_url_openai():
    task = "What is OpenAI?"
    assert guess_url(task) == "https://openai.com/"


def test_summarize_html():
    html = "<html><head><title>Test</title></head><body>Hello world</body></html>"
    out = summarize_html(html, max_lines=2)

    assert "TITLE: Test" in out
    assert "Hello world" in out
