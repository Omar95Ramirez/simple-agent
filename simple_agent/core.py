import re


def extract_url(task: str) -> str | None:
    m = re.search(r"(https?://\S+)", task)
    if m:
        return m.group(1).rstrip(").,]}>\"'")
    return None


def guess_url(task: str) -> str | None:
    t = task.lower()
    if "python" in t:
        return "https://www.python.org/"
    if "openai" in t or "chatgpt" in t:
        return "https://openai.com/"
    if "github" in t:
        return "https://github.com/"
    if "reddit" in t:
        return "https://www.reddit.com/"
    return None


def summarize_html(html: str, max_lines: int = 12) -> str:
    title = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title_text = title.group(1).strip() if title else "(no title found)"

    text = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    bullets = [text[i:i+160] for i in range(0, min(len(text), max_lines*160), 160)]
    return "TITLE: " + title_text + "\n" + "\n".join(f"- {b}…" for b in bullets)
