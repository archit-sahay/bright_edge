import argparse
import json
import sys

from .pipeline import extract_topics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="be-topics",
        description="Extracts relevant topics from a URL using NLTK-based scoring.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract topics for a single URL")
    p_extract.add_argument("--url", required=True, help="URL to analyze")
    p_extract.add_argument("--top-k", type=int, default=8, help="Number of topics to return")
    p_extract.add_argument("--timeout", type=float, default=8.0, help="HTTP timeout seconds")
    p_extract.add_argument("--no-robots", action="store_true", help="Ignore robots.txt (not recommended)")
    p_extract.add_argument("--render", action="store_true", help="Render with Playwright (JS-heavy sites)")
    p_extract.add_argument("--css-topics", action="store_true", help="Allow CSS-derived topics (classes/ids)")
    p_extract.add_argument("--verbose", action="store_true", help="Verbose errors")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "extract":
        result = extract_topics(
            url=args.url,
            top_k=args.top_k,
            timeout=args.timeout,
            respect_robots=not args.no_robots,
            render=args.render,
            include_css_topics=args.css_topics,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

