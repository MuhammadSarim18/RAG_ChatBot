import argparse
import json
import logging
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
import urllib.robotparser
import requests
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scraper")

BASE_DOMAIN = "timesconsultant.com"
DEFAULT_START_URL = "https://timesconsultant.com"
DEFAULT_USER_AGENT = "TimesConsultantBot/1.0 (+https://timesconsultant.com)"
DEFAULT_OUTPUT_FILE = "scraped_data.json"
IGNORED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".mp4", ".mp3", ".wav",
    ".css", ".js", ".json", ".xml", ".ico", ".woff", ".woff2"
)


def init_robot_parser(base_url: str, user_agent: str, session: requests.Session) -> urllib.robotparser.RobotFileParser:
    """Fetches and parses robots.txt for the given base domain."""
    robots_url = urljoin(base_url, "/robots.txt")
    rp = urllib.robotparser.RobotFileParser()
    try:
        logger.info(f"Checking robots.txt at {robots_url}...")
        resp = session.get(robots_url, timeout=10)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
            logger.info("Successfully parsed robots.txt.")
        else:
            logger.warning(f"robots.txt returned status code {resp.status_code}, assuming all paths allowed.")
            rp.parse(["User-agent: *", "Allow: /"])
    except Exception as e:
        logger.warning(f"Could not fetch robots.txt ({e}), proceeding with caution.")
        rp.parse(["User-agent: *", "Allow: /"])
    return rp


def is_valid_internal_url(url: str, base_domain: str) -> bool:
    """Verifies that the URL belongs to the target domain and is a webpage."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False

    netloc = parsed.netloc.lower()
    if netloc != base_domain and not netloc.endswith("." + base_domain):
        return False

    # Skip static non-html asset files
    path_lower = parsed.path.lower()
    if any(path_lower.endswith(ext) for ext in IGNORED_EXTENSIONS):
        return False

    return True


def clean_url(url: str) -> str:
    """Removes fragments and normalizes URL to prevent duplicates."""
    url_without_frag, _ = urldefrag(url)
    parsed = urlparse(url_without_frag)
    path = parsed.path
    if not path or path == "/":
        path = "/"
    elif path.endswith("/"):
        path = path.rstrip("/")
    return parsed._replace(path=path).geturl()


def extract_page_data(html_content: str, url: str) -> dict | None:
    """
    Strips out nav/footer/header/script/style/aside elements,
    extracts title and visible text from p/li/h1/h2/h3 tags,
    and returns a dict if visible text length >= 200 characters.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Strip out unwanted elements
    for element in soup(["nav", "footer", "header", "script", "style", "aside"]):
        element.decompose()

    # 2. Extract page title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.get_text(strip=True)
    elif soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)
    else:
        title = url

    # 3. Pull visible text from p, li, h1, h2, h3 tags
    text_nodes = soup.find_all(["p", "li", "h1", "h2", "h3"])
    extracted_lines = []
    for node in text_nodes:
        cleaned_text = node.get_text(separator=" ", strip=True)
        if cleaned_text:
            extracted_lines.append(cleaned_text)

    # Join lines with newlines
    visible_text = "\n".join(extracted_lines)

    # 4. Skip pages under 200 characters
    if len(visible_text) < 200:
        logger.info(f"Skipping {url} (Extracted text length {len(visible_text)} < 200 characters)")
        return None

    return {
        "url": url,
        "title": title,
        "text": visible_text
    }


def scrape_site(
    start_url: str = DEFAULT_START_URL,
    max_pages: int = 50,
    delay: float = 1.0,
    output_file: str = DEFAULT_OUTPUT_FILE,
    user_agent: str = DEFAULT_USER_AGENT
):
    """Crawls and scrapes pages from the target domain."""
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    # Initialize robots.txt parser
    robot_parser = init_robot_parser(start_url, user_agent, session)

    queue = deque([clean_url(start_url)])
    visited = set()
    results = []

    logger.info(f"Starting crawl from {start_url} (Max pages: {max_pages}, Delay: {delay}s)...")

    while queue and len(results) < max_pages:
        current_url = queue.popleft()

        if current_url in visited:
            continue
        visited.add(current_url)

        # Check robots.txt permissions
        if not robot_parser.can_fetch(user_agent, current_url) and not robot_parser.can_fetch("*", current_url):
            logger.warning(f"Skipping disallowed by robots.txt: {current_url}")
            continue

        logger.info(f"Fetching [{len(results) + 1}/{max_pages} target pages]: {current_url}")

        try:
            resp = session.get(current_url, timeout=15)
            content_type = resp.headers.get("Content-Type", "")

            # Ensure we only process HTML pages
            if "text/html" not in content_type:
                logger.info(f"Skipping non-HTML content-type ({content_type}): {current_url}")
                time.sleep(delay)
                continue

            if resp.status_code != 200:
                logger.warning(f"HTTP {resp.status_code} for {current_url}")
                time.sleep(delay)
                continue

            # Extract data
            page_data = extract_page_data(resp.text, current_url)
            if page_data:
                results.append(page_data)
                logger.info(f"Added '{page_data['title']}' ({len(page_data['text'])} chars). Total kept: {len(results)}")

            # Discover internal links to crawl next
            soup = BeautifulSoup(resp.text, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()
                full_url = clean_url(urljoin(current_url, href))
                if is_valid_internal_url(full_url, BASE_DOMAIN) and full_url not in visited:
                    queue.append(full_url)

        except requests.RequestException as e:
            logger.error(f"Error requesting {current_url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing {current_url}: {e}")

        # Enforce rate limit of 1 second between requests
        time.sleep(delay)

    # Save output to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Crawl finished. Saved {len(results)} pages to '{output_file}'.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scraper for timesconsultant.com with robots.txt adherence and text extraction.")
    parser.add_argument("--start-url", default=DEFAULT_START_URL, help="Starting URL for crawl")
    parser.add_argument("--max-pages", type=int, default=50, help="Maximum number of pages to extract and save")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between requests (default: 1.0)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output JSON file name")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="Custom User-Agent header")

    args = parser.parse_args()
    scrape_site(
        start_url=args.start_url,
        max_pages=args.max_pages,
        delay=args.delay,
        output_file=args.output,
        user_agent=args.user_agent
    )
