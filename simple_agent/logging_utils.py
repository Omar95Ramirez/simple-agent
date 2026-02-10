import logging
import os


def setup_logging(level: str | None = None) -> None:
    """
    Configure root logging once.
    Priority:
      1) explicit level argument
      2) env LOG_LEVEL
      3) default INFO
    """
    lvl = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    if lvl not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        lvl = "INFO"

    logging.basicConfig(
        level=getattr(logging, lvl),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Quiet noisy third-party loggers (your app still respects the chosen level)
    for noisy in ["urllib3", "httpcore", "httpx", "openai"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
