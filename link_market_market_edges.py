"""CLI: link Polymarket markets to each other via implication edges."""

from pipeline import progress, run_pipeline


def main() -> None:
    progress("Deprecated: use `python main.py link` instead.")
    run_pipeline(link=True, export=True)


if __name__ == "__main__":
    main()
