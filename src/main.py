"""CLI entry point for Horizon Brief."""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from .console import make_console
from .storage.manager import StorageManager
from .orchestrator import HorizonOrchestrator


console = make_console()


def print_banner():
    """Print the application banner."""
    banner = r"""
[bold blue]
  _    _            _
 | |  | |          (_)
 | |__| | ___  _ __ _ ___  ___  _ __
 |  __  |/ _ \| '__| |_  / / _ \| '_ \
 | |  | | (_) | |  | |/ / | (_) | | | |
 |_|  |_|\___/|_|  |_/___| \___/|_| |_|
[/bold blue]
[cyan]  Horizon Brief - AI briefing layer[/cyan]
    """
    console.print(banner)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Horizon Brief - AI briefing layer on top of Horizon")
    parser.add_argument("--hours", type=int, help="Force fetch from last N hours")
    parser.add_argument("--config", help="Path to config.json (defaults to data/config.json)")
    parser.add_argument("--data-dir", help="Directory for runtime data and summaries")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed stage progress and counters without secrets or raw prompts",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full tracebacks for local debugging",
    )
    return parser


def main():
    """Main CLI entry point."""
    print_banner()

    parser = build_parser()
    args = parser.parse_args()

    try:
        config_path = Path(args.config).expanduser().resolve() if args.config else None
        data_dir = (
            Path(args.data_dir).expanduser().resolve()
            if args.data_dir
            else (config_path.parent if config_path else Path("data"))
        )
        # Load environment variables from nearby .env files without depending on cwd.
        if config_path:
            load_dotenv(config_path.parent.parent / ".env", override=False)
            load_dotenv(config_path.parent / ".env", override=False)
        load_dotenv()

        # Initialize storage manager
        storage = StorageManager(
            data_dir=str(data_dir),
            config_path=str(config_path) if config_path else None,
        )

        # Load configuration
        try:
            config = storage.load_config()
        except FileNotFoundError:
            console.print("[bold red]❌ Configuration file not found![/bold red]\n")
            console.print(
                "Run [bold cyan]uv run horizon-wizard[/bold cyan] to launch the interactive setup wizard,\n"
                "or create [cyan]data/config.json[/cyan] manually based on the template:\n"
            )
            print_config_template()
            sys.exit(1)
        except Exception as e:
            console.print(f"[bold red]❌ Error loading configuration: {e}[/bold red]")
            sys.exit(1)

        # Create and run orchestrator
        orchestrator = HorizonOrchestrator(config, storage, verbose=args.verbose)
        asyncio.run(orchestrator.run(force_hours=args.hours))

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]❌ Fatal error: {type(e).__name__}: {e}[/bold red]")
        if "args" in locals() and args.debug:
            import traceback
            traceback.print_exc()
        else:
            console.print("[dim]Run with --debug to print a full traceback.[/dim]")
        sys.exit(1)


def print_config_template():
    """Print configuration template."""
    template = """
{
  "version": "1.0",
  "ai": {
    "provider": "anthropic",
    "model": "claude-sonnet-4.5-20250929",
    "api_key_env": "ANTHROPIC_API_KEY",
    "temperature": 0.3,
    "max_tokens": 4096
  },
  "sources": {
    "github": [
      {
        "type": "user_events",
        "username": "torvalds",
        "enabled": true
      }
    ],
    "hackernews": {
      "enabled": true,
      "fetch_top_stories": 30,
      "min_score": 100
    },
    "rss": [
      {
        "name": "Example Blog",
        "url": "https://example.com/feed.xml",
        "enabled": true,
        "category": "software-engineering"
      }
    ]
  },
  "filtering": {
    "ai_score_threshold": 7.0,
    "time_window_hours": 24
  }
}

Also create a .env file with:
ANTHROPIC_API_KEY=your_api_key_here
GITHUB_TOKEN=your_github_token_here (optional but recommended)
"""
    console.print(template)


if __name__ == "__main__":
    main()
