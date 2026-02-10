from bs4 import BeautifulSoup
import requests

def web_get(url: str) -> str:
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
