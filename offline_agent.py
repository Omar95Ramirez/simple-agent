import re
import requests

SYSTEM = """You are a tiny offline agent.
You can use one tool:
- web_get(url): fetches a webpage (no AI reasoning)

Given a task, do:
1) Decide a URL if needed
2) Call web_get
3) Print a short summary using simple heuristics
"""

def web_get(url: str) -> str:
    r = requests.get(url, timeout=15, headers={"User-Agent": "offline-agent/1.0"})
    r.raise_for_status()
    return r.text[:120000]

def summarize_html(html: str, max_lines: int = 12) -> str:
    # Extremely simple heuristic "summary": extract title + some visible text-ish chunks
    title = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title_text = title.group(1).strip() if title else "(no title found)"
    text = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    bullets = [text[i:i+160] for i in range(0, min(len(text), max_lines*160), 160)]
    return "TITLE: " + title_text + "\n" + "\n".join(f"- {b}…" for b in bullets)

def run():
    task = input("Task: ").strip()
    if "python" in task.lower():
        url = "https://www.python.org/"
        print(f"\n[Agent] Using web_get({url})")
        html = web_get(url)
        print("\n[Agent] Summary:\n")
        print(summarize_html(html))
    else:
        print("Try a task mentioning 'Python' to see the web tool flow.")

if __name__ == "__main__":
    run()

