import os
import sys
import json
import re
import argparse
import requests
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# AGENT UPGRADES (A,B,C,D,E)
# =========================
import time
import random
from urllib.parse import urljoin, urlparse

def required_bullets_from_task(task: str):
    """
    Detect "in N bullets" / "exactly N bullets" / "N bullets" requirements.
    Returns int or None.
    """
    t = (task or "").lower()
    patterns = [
        r'exactly\s+(\d+)\s+bullets?',
        r'in\s+(\d+)\s+bullets?',
        r'(\d+)\s+bullets?'
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            try:
                return int(m.group(1))
            except:
                pass
    return None

def count_bullets(text: str):
    return sum(1 for line in (text or "").splitlines() if line.lstrip().startswith("- "))

def normalize_url(u: str):
    u = (u or "").strip().strip('"').strip("'")
    if not u:
        return u
    if u.startswith("//"):
        return "https:" + u
    if not re.match(r"^https?://", u, re.I):
        u = "https://" + u
    return u

def extract_tool_call(reply: str):
    """
    (B) Robust tool-call parsing.
    Accepts:
      - TOOL.web_get(https://x.com)
      - TOOL.web_get(url=https://x.com)
      - TOOL.web_get("https://x.com")
      - tool: web_get https://x.com
      - web_get https://x.com
    Returns: ("web_get", url) or (None, None)
    """
    txt = (reply or "").strip()

    # TOOL.web_get(...)
    m = re.search(r"TOOL\.web_get\s*\(\s*(?:url\s*=\s*)?([^\)\n]+)\)", txt, re.I)
    if m:
        u = m.group(1).strip().rstrip(",")
        return ("web_get", normalize_url(u))

    # tool: web_get https://...
    m = re.search(r"\btool\s*:\s*web_get\b\s+(\S+)", txt, re.I)
    if m:
        return ("web_get", normalize_url(m.group(1)))

    # plain: web_get https://...
    m = re.search(r"\bweb_get\b\s+(\S+)", txt, re.I)
    if m:
        return ("web_get", normalize_url(m.group(1)))

    return (None, None)

def truncate_for_model(text: str, limit: int):
    if not text:
        return text
    return text if len(text) <= limit else (text[:limit] + "\n...[truncated]...")

def truncate_for_print(text: str, limit: int):
    if not text:
        return text
    return text if len(text) <= limit else (text[:limit] + "\n...[truncated for print]...")

def looks_like_html(s: str):
    if not s:
        return False
    head = s.lstrip()[:200].lower()
    return "<html" in head or "<!doctype" in head or "<head" in head

def html_to_structured_json(html: str, base_url: str, max_chars: int = 18000):
    """
    (C,D) Convert HTML -> JSON-ish dict string: title, headings, text, links.
    Requires beautifulsoup4 installed.
    """
    try:
        from bs4 import BeautifulSoup
    except Exception as e:
        return {"error": f"BeautifulSoup missing: {e}", "url": base_url}

    soup = BeautifulSoup(html, "html.parser")

    # drop noisy tags
    for tag in soup(["script", "style", "noscript"]):
        try:
            tag.decompose()
        except:
            pass

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    headings = []
    for h in soup.find_all(["h1", "h2", "h3"]):
        t = (h.get_text(" ", strip=True) or "").strip()
        if t:
            headings.append(t)
        if len(headings) >= 30:
            break

    # collect links (absolute)
    links = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        href = href.strip()
        # skip anchors & mailto & javascript
        if href.startswith("#") or href.lower().startswith("mailto:") or href.lower().startswith("javascript:"):
            continue
        abs_url = urljoin(base_url, href)
        # keep only http(s)
        if urlparse(abs_url).scheme in ("http", "https"):
            links.append(abs_url)
        if len(links) >= 50:
            break

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    clean = "\n".join(lines)
    clean = clean[:max_chars]

    return {
        "url": base_url,
        "title": title,
        "headings": headings,
        "links": links,
        "text": clean,
    }

def web_get_with_retries(url: str, timeout: int = 20, retries: int = 3, backoff: float = 0.7, headers=None):
    """
    (E) retries + backoff + clearer errors
    """
    import requests
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers=headers or {"User-Agent": "simple-agent"})
            r.raise_for_status()
            return r.text, None
        except Exception as e:
            last_err = e
            if attempt < retries:
                sleep_s = backoff * (2 ** (attempt - 1)) + random.random() * 0.2
                time.sleep(sleep_s)
    return None, str(last_err)



# =========================
# Config / Defaults
# =========================
CACHE_FILE_DEFAULT = "cache.json"
MEMORY_FILE_DEFAULT = "memory.json"

MODEL_DEFAULT = "gpt-4.1-mini"
MAX_STEPS_DEFAULT = 12
MAX_TOOL_CALLS_DEFAULT = 2

# Keep tool output small when feeding back to model (stability + cost)
TOOL_RESULT_LIMIT_FOR_MODEL_DEFAULT = 3000
# Keep tool output snippet short if you ever print it (we mostly don't)
TOOL_RESULT_LIMIT_FOR_PRINT_DEFAULT = 400

USER_AGENT = "simple-agent"
HTTP_TIMEOUT = 15


SYSTEM = """You are a small autonomous agent.

Process:
1. Think briefly.
2. If needed, respond exactly with TOOL:web_get(<url>) or TOOL:web_get(url=<url>).
3. After tool result, continue reasoning.
4. When finished, reply starting with FINAL: (start of line)

Keep reasoning short.
"""


# =========================
# Utilities: JSON persistence
# =========================
def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# =========================
# Cache + Memory
# =========================
def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    if not url.lower().startswith(("http://", "https://")):
        url = "https://" + url
    return url


from bs4 import BeautifulSoup
import requests


from bs4 import BeautifulSoup
import requests

def web_get(url: str, cache=None, cache_file=None, use_cache: bool=True) -> str:
    if url in WEB_CACHE:
        return WEB_CACHE[url]

    r = requests.get(url, timeout=20, headers={"User-Agent": "simple-agent"})
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Remove junk
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Normalize whitespace
    lines = [l.strip() for l in text.splitlines()]
    clean = "\n".join(l for l in lines if l)

    # Truncate so model doesn’t drown
    clean = clean[:12000]

    WEB_CACHE[url] = clean
    save_cache(WEB_CACHE)
    return clean

    r = requests.get(url, timeout=20, headers={"User-Agent": "simple-agent"})
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Remove junk
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Normalize whitespace
    lines = [l.strip() for l in text.splitlines()]
    clean = "\n".join(l for l in lines if l)

    # Truncate so model doesn’t drown
    clean = clean[:12000]

    WEB_CACHE[url] = clean
    save_cache(WEB_CACHE)
    return clean


    if use_cache and url in cache:
        return cache[url]

    r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    text = r.text[:6000]  # hard cap to keep things lightweight

    if use_cache:
        cache[url] = text
        save_json(cache_file, cache)

    return text


def save_memory(task: str, answer: str, memory_file: str):
    mem = load_json(memory_file, [])
    mem.append({"task": task, "answer": answer})
    save_json(memory_file, mem)


# =========================
# Tool parsing (whitespace tolerant)
# Accepts:
#   TOOL:web_get(https://example.com)
#   TOOL:web_get(url=https://example.com)
#   TOOL:web_get( url = https://example.com )
# =========================
TOOL_WEB_GET_RE = re.compile(
    r"^TOOL\s*:\s*web_get\s*\(\s*(?P<arg>.*?)\s*\)\s*$",
    re.IGNORECASE,
)


def parse_web_get_call(text: str):
    m = TOOL_WEB_GET_RE.match((text or "").strip())
    if not m:
        return None
    arg = m.group("arg").strip().strip('"').strip("'")
    if arg.lower().startswith("url="):
        arg = arg[4:].strip().strip('"').strip("'")
    return normalize_url(arg)


def truncate(s: str, limit: int) -> str:
    s = s or ""
    if len(s) <= limit:
        return s
    return s[:limit] + "\n...[truncated]..."


# =========================
# Agent loop
# =========================

# ======================
# CONFIDENCE STOP HELPERS
# ======================

def has_enough_bullets(text: str, n: int = 2) -> bool:
    bullets = [l for l in text.splitlines() if l.strip().startswith("- ")]
    return len(bullets) >= n


def run(task: str, *, client: OpenAI, model: str, debug: bool,
        max_steps: int, max_tool_calls: int,
        cache_file: str, memory_file: str, use_cache: bool,
        tool_result_limit_for_model: int,
        tool_result_limit_for_print: int):

    cache = load_json(cache_file, {})
    past_memory = load_json(memory_file, [])

    messages = [{"role": "system", "content": SYSTEM}]

    # Keep prompt small: include last ~10 memories
    if past_memory:
        tail = past_memory[-10:]
        messages.append({
            "role": "system",
            "content": "PAST MEMORY (for context):\n" + json.dumps(tail, indent=2)
        })

    # Option A: enforce tool usage if user requests it
    must_use_tool = "must use the tool" in (task or "").lower()
    tool_calls_used = 0
    used_urls = set()

    messages.append({"role": "user", "content": task})

    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        reply = (resp.choices[0].message.content or "").strip()

        # ---- FINAL detection (must be start-of-line) ----
        # Accept either "FINAL:" or "FINAL ANSWER:" at start of reply.
        if reply.startswith("FINAL:") or reply.startswith("FINAL ANSWER:") or has_enough_bullets(reply, 2):
            # If tool was required but never used, block FINAL and force tool usage
            if must_use_tool and tool_calls_used == 0:
                messages.append({"role": "assistant", "content": reply})
                messages.append({
                    "role": "system",
                    "content": "You MUST call TOOL:web_get(...) at least once before producing FINAL."
                })
                continue

            # Extract answer body
            if reply.startswith("FINAL ANSWER:"):
                answer = reply[len("FINAL ANSWER:"):].strip()
            else:
                answer = reply[len("FINAL:"):].strip()

            print("\n====================")
            print("FINAL ANSWER:")
            print(answer)
            print("====================\n")

            save_memory(task, answer, memory_file)

            # HARD EXIT to prevent any extra looping
            sys.exit(0)

        # ---- Tool parsing ----
        url = parse_web_get_call(reply)

        # If NOT a tool call, print normally (debug shows steps; non-debug still prints)
        if url is None:
            if debug:
                print(f"\nSTEP {step}:\n{reply}")
            else:
                # In non-debug, still show assistant progress (optional: comment out if you want quieter)
                print(f"\nSTEP {step}:\n{reply}")

            messages.append({"role": "assistant", "content": reply})
            continue

        # ---- Tool call handling ----
        if tool_calls_used >= max_tool_calls:
            # Prevent infinite tool spam
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "system",
                "content": f"You already used {max_tool_calls} tool calls. Do not call tools again; produce FINAL now."
            })
            continue

        if url in used_urls:
            # Prevent repeat URL calls in multi-hop loops
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "system",
                "content": "You already fetched that exact URL in this run. Fetch a different URL or produce FINAL."
            })
            continue

        used_urls.add(url)
        tool_calls_used += 1

        # Execute tool
        try:
            was_cached = use_cache and (url in cache)
            result_full = web_get(url, cache, cache_file, use_cache)
            status = " (cache hit)" if was_cached else ""
            print(f"\n[tool] web_get -> {url}{status}")
        except KeyboardInterrupt:
            print("\nInterrupted.")
            sys.exit(130)
        except Exception as e:
            result_full = f"Tool error fetching {url}: {e}"
            print(f"\n[tool] web_get -> {url} (error)")

        # Option B: truncate what we feed back to the model
        result_for_model = truncate(result_full, tool_result_limit_for_model)

        # (Optional) debug print a tiny snippet of tool output
        if debug:
            snippet = truncate(result_full, tool_result_limit_for_print)
            print(f"[tool] snippet:\n{snippet}\n")

        # Add messages so model can continue
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Tool result:\n{result_for_model}"})

    print("\nStopped: exceeded max steps without FINAL.")
    sys.exit(2)


# =========================
# CLI entrypoint (Upgrade #3)
# =========================
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Simple tool-using agent")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--debug", action="store_true", help="Print extra debugging output")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache.json usage")
    parser.add_argument("--cache-file", default=CACHE_FILE_DEFAULT)
    parser.add_argument("--memory-file", default=MEMORY_FILE_DEFAULT)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    parser.add_argument("--max-tool-calls", type=int, default=MAX_TOOL_CALLS_DEFAULT)
    parser.add_argument("--tool-result-limit-model", type=int, default=TOOL_RESULT_LIMIT_FOR_MODEL_DEFAULT)
    parser.add_argument("--tool-result-limit-print", type=int, default=TOOL_RESULT_LIMIT_FOR_PRINT_DEFAULT)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    try:
        task = input("Task: ").strip()
        run(
            task,
            client=client,
            model=args.model,
            debug=args.debug,
            max_steps=args.max_steps,
            max_tool_calls=args.max_tool_calls,
            cache_file=args.cache_file,
            memory_file=args.memory_file,
            use_cache=(not args.no_cache),
            tool_result_limit_for_model=args.tool_result_limit_model,
            tool_result_limit_for_print=args.tool_result_limit_print,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()

