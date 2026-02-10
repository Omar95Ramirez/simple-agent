import argparse
import logging
import requests

from simple_agent.core import extract_url, guess_url, summarize_html
from simple_agent.logging_utils import setup_logging

logger = logging.getLogger("offline_agent")

SYSTEM = """You are a tiny offline agent.
You can use one tool:
- web_get(url): fetches a webpage (no AI reasoning)

Given a task, do:
1) Decide a URL if needed
2) Call web_get
3) Print a short summary using simple heuristics
"""

def web_get(url: str) -> str:
    logger.info("web_get %s", url)
    r = requests.get(url, timeout=15, headers={"User-Agent": "offline-agent/1.0"})
    r.raise_for_status()
    return r.text[:120000]


def run(task: str | None):
    if not task:
        task = input("Task: ").strip()

    if not task:
        logger.error("No task provided.")
        return

    url = extract_url(task) or guess_url(task)
    if not url:
        logger.error("Could not infer a URL. Please include an https:// link.")
        return

    logger.info("Using URL: %s", url)

    try:
        html = web_get(url)
    except Exception as e:
        logger.exception("web_get failed: %s", e)
        return

    print("\n[Agent] Summary:\n")
    print(summarize_html(html))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Task or prompt")
    parser.add_argument("--log-level", help="DEBUG|INFO|WARNING|ERROR")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run(args.task)
