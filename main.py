"""CLI entrypoint — unified knowledge-graph pipeline."""

import argparse

from dotenv import load_dotenv

import pipeline

# Backward-compatible re-exports for scripts that import from main.
progress = pipeline.progress
require_env = pipeline.require_env
normalize_azure_openai_endpoint = pipeline.normalize_azure_openai_endpoint
build_engine = pipeline.build_engine
assign_chunks_to_objects = pipeline.assign_chunks_to_objects
extract_and_merge_objects = pipeline.extract_and_merge_objects
AZURE_OPENAI_HTTP_API_VERSION = pipeline.AZURE_OPENAI_HTTP_API_VERSION

_progress = progress
_require_env = require_env
_normalize_azure_openai_endpoint = normalize_azure_openai_endpoint
_assign_chunks_to_objects = assign_chunks_to_objects
_extract_and_merge_objects = extract_and_merge_objects


def _run_command(command: str) -> None:
    root = pipeline.project_root()
    load_dotenv(root / ".env")

    if command in ("all", ""):
        pipeline.run_pipeline(
            index=True,
            sync=True,
            link=True,
            export=True,
            root=root,
        )
    elif command == "index":
        pipeline.run_pipeline(index=True, export=True, root=root)
    elif command == "sync":
        pipeline.run_pipeline(sync=True, export=True, root=root)
    elif command == "link":
        pipeline.run_pipeline(link=True, export=True, root=root)
    elif command == "export":
        pipeline.run_pipeline(export=True, root=root)
    else:
        raise SystemExit(f"Unknown command: {command}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the knowledge graph: documents, markets, instruments, edges, viz."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("all", "index", "sync", "link", "export"),
        help="Pipeline phase (default: all)",
    )
    args = parser.parse_args()
    _run_command(args.command)


if __name__ == "__main__":
    main()
