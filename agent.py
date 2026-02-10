import os
import sys
import json
import re
import time
import random
import argparse
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv
import logging
from simple_agent.logging_utils import setup_logging
logger = logging.getLogger("simple_agent")

from openai import OpenAI
from simple_agent.core import extract_url, guess_url, summarize_html


# =========================
# Config / Defaults
# =========================
CACHE_FILE_DEFAULT = "cache.json"
MEMORY_FILE_DEFAULT = "memory.json"

MODEL_DEFAULT = "gpt-4.1-mini"
MAX_STEPS_DEFAULT = 12
MAX_TOOL_CALLS_DEFAULT = 2

TOOL_RESULT_LIMIT_FOR_MODEL_DEFAULT = 3000
TOOL_RESULT_LIMIT_FOR_PRINT_DEFAULT = 400

USER_AGENT = "simple-agent"
HTTP_TIMEOUT = 15


SYSTEM = """You are a small autonomous agent.

Process:
1. Think briefly.
2. If needed, respond exactly with TOOL:web_get(<url>) or TOOL:web_get(url=<url>).
3. After tool result, continue reasoning.
4. When finished, reply with:

FINAL ANSWER:
- bullet 1
- bullet 2

IMPORTANT:
- The FINAL marker must be at the start of a line: "FINAL:" or "FINAL ANSWER:"
- Use hyphen bullets ("- ").
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
# Bullet helpers
# =========================
def required_bullets_from_task(task: str):
    """
    Detect "in N bullets" / "exactly N bullets" / "N bullets" requirements.
    Returns int or None.
    """
    t = (task or "").lower()
    patterns = [
        r"exactly\s+(\d+)\s+bullets?",
        r"in\s+(\d+)\s+bullets?",
        r"(\d+)\s+bullets?",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def count_bullets(text: str) -> int:
    return sum(1 for line in (text or "").splitlines() if line.lstrip().startswith("- "))


# =========================
# String helpers
# =========================
def normalize_url(url: str) -> str:
    url = (url or "").strip().strip('"').strip("'")
    if not url:
        return url
    if url.startswith("//"):
        return "https:" + url
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return url


def truncate(s: str, limit: int) -> str:
    s = s or ""
    return s if len(s) <= limit else (s[:limit] + "\n...[truncated]...")


# =========================
# Tool parsing
# Accepts:
#   TOOL:web_get(https://example.com)
#   TOOL:web_get(url=https://example.com)
#   TOOL:web_get("https://example.com")
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


# =========================
# HTML cleanup (optional)
# =========================
def looks_like_html(s: str) -> bool:
    if not s:
        return False
    head = s.lstrip()[:200].lower()
    return "<html" in head or "<!doctype" in head or "<head" in head


def html_to_structured_text(html: str, base_url: str, max_chars: int = 12000) -> str:
    """
    If bs4 is installed, strip scripts/styles and return readable text.
    If not installed, return a truncated raw string.
    """
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return truncate(html, max_chars)

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        try:
            tag.decompose()
        except Exception:
            pass

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    clean = "\n".join(lines)
    return truncate(clean, max_chars)


# =========================
# Web tool (with cache + retries)
# =========================
def web_get_with_retries(url: str, timeout: int = 20, retries: int = 3, backoff: float = 0.7, headers=None):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers=headers or {"User-Agent": USER_AGENT})
            r.raise_for_status()
            return r.text, None
        except Exception as e:
            last_err = e
            if attempt < retries:
                sleep_s = backoff * (2 ** (attempt - 1)) + random.random() * 0.2
                time.sleep(sleep_s)
    return None, str(last_err)


def web_get(url: str, cache: dict, cache_file: str, use_cache: bool = True) -> str:
    url = normalize_url(url)

    if use_cache and url in cache:
        return cache[url]

    html, err = web_get_with_retries(url, timeout=HTTP_TIMEOUT, retries=3, backoff=0.7)
    if err:
        raise RuntimeError(err)

    # Convert HTML to readable text if it looks like HTML
    if looks_like_html(html):
        text = html_to_structured_text(html, url, max_chars=12000)
    else:
        text = truncate(html, 6000)

    if use_cache:
        cache[url] = text
        save_json(cache_file, cache)

    return text


# =========================
# Memory
# =========================
def save_memory(task: str, answer: str, memory_file: str):
    mem = load_json(memory_file, [])
    mem.append({"task": task, "answer": answer})
    save_json(memory_file, mem)


# =========================
# FINAL detection
# =========================
FINAL_LINE_RE = re.compile(
    r"(?m)^(?:Task:\s*)?(FINAL(?: ANSWER)?):\s*(.*)$"
)


def extract_final_body(reply: str):
    """
    Finds a line starting with FINAL: or FINAL ANSWER: (optionally prefixed by 'Task: ').
    Returns (marker, body) or (None, None).
    Body includes:
      - any text on the same line after the colon
      - plus all following lines
    """
    m = FINAL_LINE_RE.search(reply or "")
    if not m:
        return None, None

    marker = m.group(1)  # FINAL or FINAL ANSWER
    rest_same_line = m.group(2) or ""

    tail = (reply or "")[m.end():]
    # If there is content after marker on the same line, include it first
    pieces = []
    if rest_same_line.strip():
        pieces.append(rest_same_line.strip())
    if tail:
        # drop one leading newline if present
        tail = tail[1:] if tail.startswith("\n") else tail
        pieces.append(tail)

    body = "\n".join(pieces).strip()
    return marker, body


# =========================
# Agent loop
# =========================
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

    must_use_tool = "must use the tool" in (task or "").lower()
    tool_calls_used = 0
    used_urls = set()

    # bullet requirement: from task if present, else default 2
    required_bullets = required_bullets_from_task(task) or 2

    messages.append({"role": "user", "content": task})

    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        reply = (resp.choices[0].message.content or "").strip()

        # ---- FINAL detection ----
        marker, body = extract_final_body(reply)
        if marker is not None:
            # If tool was required but never used, block FINAL and force tool usage
            if must_use_tool and tool_calls_used == 0:
                messages.append({"role": "assistant", "content": reply})
                messages.append({
                    "role": "system",
                    "content": "You MUST call TOOL:web_get(...) at least once before producing FINAL."
                })
                continue

            if count_bullets(body) >= required_bullets:
                print("\n====================")
                print("FINAL ANSWER:")
                print(body)
                print("====================\n")
                save_memory(task, body, memory_file)
                sys.exit(0)

            # Not enough bullets -> force model to continue
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "system",
                "content": f"Your FINAL answer must include at least {required_bullets} bullet points (lines starting with '- '). Try again. Output only the final answer."
            })
            continue

        # ---- Fallback FINAL: accept bullet-only reply when task requires bullets ----
        # (Prevents infinite loops when the model complies with bullets but forgets FINAL.)
        if count_bullets(reply) >= required_bullets:
            # Reject acknowledgement-only bullets
            lower = reply.lower()
            if any(x in lower for x in ["understood", "awaiting", "ready to", "please provide"]):
                messages.append({"role": "assistant", "content": reply})
                messages.append({
                    "role": "system",
                    "content": "You produced acknowledgement bullets. Provide ACTUAL answers in bullet points. Output only the final answer."
                })
                continue

            # If tool was required but never used, block FINAL and force tool usage
            if must_use_tool and tool_calls_used == 0:
                messages.append({"role": "assistant", "content": reply})
                messages.append({
                    "role": "system",
                    "content": "You MUST call TOOL:web_get(...) at least once before producing FINAL."
                })
                continue

            print("\n====================")
            print("FINAL ANSWER:")
            print(reply)
            print("====================\n")
            save_memory(task, reply, memory_file)
            sys.exit(0)

        # ---- Tool parsing ----
        url = parse_web_get_call(reply)

        # If NOT a tool call, print normally
        if url is None:
            # Bootstrap URL from task if model didn't call tool (once)
            if tool_calls_used == 0:
                bootstrap_url = extract_url(task) or guess_url(task)
                if bootstrap_url:
                    url = bootstrap_url
                    print(f"\n[bootstrap] web_get -> {url}")
                    tool_calls_used += 1
                    used_urls.add(url)
                    try:
                        result_full = web_get(url, cache, cache_file, use_cache)
                    except Exception as e:
                        result_full = f"Tool error fetching {url}: {e}"
        
                    result_for_model = truncate(result_full, tool_result_limit_for_model)
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content": f"Tool result:\n{result_for_model}"})
                    continue
            if debug:
                logger.debug("STEP %s:\n%s", step, reply)
            else:
                logger.debug("STEP %s:\n%s", step, reply)
            messages.append({"role": "assistant", "content": reply})
            continue

        # ---- Tool call handling ----
        if tool_calls_used >= max_tool_calls:
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "system",
                "content": f"You already used {max_tool_calls} tool calls. Do not call tools again; produce FINAL now."
            })
            continue

        if url in used_urls:
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
            logger.info("tool web_get %s%s", url, status)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            sys.exit(130)
        except Exception as e:
            result_full = f"Tool error fetching {url}: {e}"
            logger.error("tool web_get %s (error)", url)

        result_for_model = truncate(result_full, tool_result_limit_for_model)

        if debug:
            snippet = truncate(result_full, tool_result_limit_for_print)
            print(f"[tool] snippet:\n{snippet}\n")

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Tool result:\n{result_for_model}"})

    print("\nStopped: exceeded max steps without FINAL.")
    sys.exit(2)


# =========================
# CLI
# =========================
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Simple tool-using agent")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--debug", action="store_true", help="Print extra debugging output")
    parser.add_argument("--log-level", help="DEBUG|INFO|WARNING|ERROR")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache.json usage")
    parser.add_argument("--cache-file", default=CACHE_FILE_DEFAULT)
    parser.add_argument("--memory-file", default=MEMORY_FILE_DEFAULT)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    parser.add_argument("--max-tool-calls", type=int, default=MAX_TOOL_CALLS_DEFAULT)
    parser.add_argument("--tool-result-limit-model", type=int, default=TOOL_RESULT_LIMIT_FOR_MODEL_DEFAULT)
    parser.add_argument("--tool-result-limit-print", type=int, default=TOOL_RESULT_LIMIT_FOR_PRINT_DEFAULT)
    args = parser.parse_args()
    setup_logging(args.log_level)

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
