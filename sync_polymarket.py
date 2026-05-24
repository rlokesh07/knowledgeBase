"""CLI: sync relevant Polymarket markets into database.pkl."""

from pipeline import progress, run_pipeline


def main() -> None:
    progress("Deprecated: use `python main.py sync` instead.")
    run_pipeline(sync_polymarket_only=True, export=True)


if __name__ == "__main__":
    main()
