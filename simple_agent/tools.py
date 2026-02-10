from dataclasses import dataclass
from typing import Callable, Dict, Optional
import re


@dataclass
class Tool:
    name: str
    func: Callable
    description: str


TOOLS: Dict[str, Tool] = {}


def register_tool(name: str, func: Callable, description: str):
    TOOLS[name] = Tool(name=name, func=func, description=description)


TOOL_CALL_RE = re.compile(r"^TOOL:(\w+)\((.*?)\)$", re.S)


def parse_tool_call(text: str) -> Optional[tuple[str, str]]:
    """
    Parses TOOL:name(args)
    Returns (name, args) or None.

    Supports:
      - TOOL:web_get(https://example.com)
      - TOOL:web_get(url=https://example.com)
      - TOOL:web_get("https://example.com")
      - TOOL:web_get(url="https://example.com")
    """
    m = TOOL_CALL_RE.match((text or "").strip())
    if not m:
        return None
    name = m.group(1)
    args = (m.group(2) or "").strip()

    # Handle url=... wrapper if present
    if args.lower().startswith("url="):
        args = args.split("=", 1)[1].strip()

    # Strip surrounding quotes
    if (len(args) >= 2) and ((args[0] == args[-1]) and args[0] in ("'", '"')):
        args = args[1:-1].strip()

    return name, args
