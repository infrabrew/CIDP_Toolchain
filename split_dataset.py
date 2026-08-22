#!/usr/bin/env python3
# ==============================================================================
# AUTHOR:          Peter A. Aldrich Jr. (PJ)
# PROJECT:         AI Platform Engineering
# MODULE:          splitting_codebase_content.py
# DESCRIPTION:     Multi-Format Dataset Splitter for .icb (Ingested Codebase)
#                  and standard RAG dataset files (.jsonl, .json, .csv, .md).
#                  Partition oversized files into target byte-size chunks without
#                  corrupting structural line or record boundaries.
# VERSION:         2.2.0
# PYTHON_VERSION:  3.8+
# DEPENDENCIES:    Standard Library Only (gzip, json, csv, argparse, pathlib)
# ==============================================================================

"""
Dataset Splitter Tool (.icb & Multi-Format Edition)
====================================================

This module provides a production-grade utility for splitting oversized codebase 
ingestion datasets into clean, byte-bounded chunk files suitable for direct LLM 
uploads, RAG vector database ingestion pipelines, and API batch payloads.

Key Technical Capabilities:
---------------------------
1. Structural Boundary Integrity:
   Splits text line-by-line or record-by-record, guaranteeing that individual JSON 
   lines, CSV rows, or code snippets are never split mid-line or corrupted.

2. .icb (Ingested Codebase) Native Support:
   Full first-class support for ARCHON `.icb` and compressed `.icb.gz` files.

3. CSV Header Duplication:
   When splitting `.csv` files, the original header row is extracted and automatically 
   prepended to the top of every generated split chunk file to ensure downstream 
   DataFrame/Pandas tools parse each chunk cleanly.

4. Valid JSON Array Wrapping:
   When targeting `-f json`, records are buffered and wrapped in standard `[...]` 
   JSON arrays per split file, guaranteeing syntax validity for every output chunk.

5. Dynamic Extension Sanitization:
   Automatically detects compound extensions (e.g., `.icb.gz`, `.jsonl.gz`) to avoid 
   redundant filename artifacts (e.g., generates `codebase_part001.icb` instead of 
   `codebase.icb_part001.icb`).

Usage Examples:
---------------
1. Split an .icb dataset into 15 MiB .icb chunks:
   $ python split_dataset.py codebase_ingested.icb -f icb -m 15

2. Split an .icb dataset into 10 MiB Gzip-compressed .icb.gz chunks:
   $ python split_dataset.py codebase_ingested.icb -f icb -m 10 -c

3. Split a CSV file into 12 MiB chunks while preserving headers:
   $ python split_dataset.py dataset.csv -f csv -m 12

4. Convert and split a JSONL file into valid JSON array chunks (.json):
   $ python split_dataset.py codebase_ingested.jsonl -f json -m 15
"""

import os
import sys
import json
import gzip
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Generator, IO


# ==============================================================================
# CORE DATASET SPLITTING ENGINE
# ==============================================================================

def split_dataset(
    input_file: Path,
    max_mb: float,
    export_format: str,
    compress: bool
) -> List[Path]:
    """
    Reads an input dataset file and partitions it into numbered chunk files, 
    ensuring no output chunk exceeds `max_mb` in size.

    Args:
        input_file (Path): Path to the source dataset file to split.
        max_mb (float): Maximum allowed file size per output chunk in MiB.
        export_format (str): Target extension format ('icb', 'jsonl', 'json', 'csv', 'txt', 'md').
        compress (bool): If True, compresses output split chunks using Gzip (.gz).

    Returns:
        List[Path]: A list of resolved Path objects pointing to all generated split chunks.

    Raises:
        FileNotFoundError: If the input_file path does not exist on disk.
    """
    # Resolve absolute path for unambiguous filesystem operations
    input_file = input_file.resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found at: {input_file}")

    # Convert megabytes (MiB) to exact byte count threshold
    max_bytes = int(max_mb * 1024 * 1024)
    export_format = export_format.lower()

    # Detect if the source file is Gzip compressed based on extension
    is_input_gz = input_file.suffix.lower() == ".gz"
    open_input_fn = gzip.open if is_input_gz else open

    # Strip existing extensions to create a clean base name for output files and folder
    base_name = input_file.name
    known_extensions = [
        ".icb.gz", ".jsonl.gz", ".json.gz", ".csv.gz", ".txt.gz", ".md.gz",
        ".icb", ".jsonl", ".json", ".csv", ".txt", ".md", ".gz"
    ]
    for ext in known_extensions:
        if base_name.endswith(ext):
            base_name = base_name[:-len(ext)]
            break

    # Construct destination directory for chunk artifacts (e.g., "codebase_chunks/")
    output_dir = input_file.parent / f"{base_name}_chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine target extension string
    out_ext = f".{export_format}"
    if export_format in ["markdown", "md"]:
        out_ext = ".md"
    if compress:
        out_ext += ".gz"

    generated_chunks: List[Path] = []
    chunk_index = 1
    current_bytes = 0

    # Select output writer: gzip.open for compressed streams, standard open for plain text
    open_output_fn = gzip.open if compress else open

    def get_chunk_path(idx: int) -> Path:
        """Helper to generate formatted chunk output file paths (e.g., repo_part001.icb)."""
        return output_dir / f"{base_name}_part{idx:03d}{out_ext}"

    print(f"\n--> Starting Dataset Splitter Pipeline...")
    print(f"  • Source File:   {input_file}")
    print(f"  • Max Chunk Size: {max_mb} MiB ({max_bytes:,} bytes)")
    print(f"  • Export Format:  {export_format.upper()} {'[Gzip Compressed]' if compress else '[Plain Text]'}")
    print(f"  • Destination:   {output_dir}\n")

    # ==========================================================================
    # STRATEGY 1: LINE-DELIMITED STREAMING (.icb, .jsonl, .txt, .md)
    # ==========================================================================
    if export_format in ["icb", "jsonl", "txt", "md"]:
        current_path = get_chunk_path(chunk_index)
        current_file = open_output_fn(current_path, "wt", encoding="utf-8")
        generated_chunks.append(current_path)

        try:
            with open_input_fn(input_file, "rt", encoding="utf-8") as f_in:
                for line in f_in:
                    # Calculate byte overhead of the current line in UTF-8
                    line_bytes = len(line.encode("utf-8"))

                    # If writing this line breaches max_bytes, close current chunk and rotate
                    if current_bytes + line_bytes > max_bytes and current_bytes > 0:
                        current_file.close()
                        chunk_size_mb = current_bytes / (1024 * 1024)
                        print(f"  • Created Chunk {chunk_index:03d}: {current_path.name} ({chunk_size_mb:.2f} MiB)")

                        chunk_index += 1
                        current_bytes = 0
                        current_path = get_chunk_path(chunk_index)
                        current_file = open_output_fn(current_path, "wt", encoding="utf-8")
                        generated_chunks.append(current_path)

                    current_file.write(line)
                    current_bytes += line_bytes
        finally:
            current_file.close()
            chunk_size_mb = current_bytes / (1024 * 1024)
            print(f"  • Created Chunk {chunk_index:03d}: {current_path.name} ({chunk_size_mb:.2f} MiB)")

    # ==========================================================================
    # STRATEGY 2: TABULAR CSV WITH HEADER REPLICATION (.csv)
    # ==========================================================================
    elif export_format == "csv":
        current_path = get_chunk_path(chunk_index)
        current_file = open_output_fn(current_path, "wt", encoding="utf-8")
        generated_chunks.append(current_path)

        try:
            with open_input_fn(input_file, "rt", encoding="utf-8") as f_in:
                # Read and retain the header line
                header_line = f_in.readline()
                header_bytes = len(header_line.encode("utf-8"))

                # Write header to the first chunk
                current_file.write(header_line)
                current_bytes += header_bytes

                for line in f_in:
                    line_bytes = len(line.encode("utf-8"))

                    # If adding this row exceeds max_bytes, rotate chunk file
                    if current_bytes + line_bytes > max_bytes and current_bytes > header_bytes:
                        current_file.close()
                        chunk_size_mb = current_bytes / (1024 * 1024)
                        print(f"  • Created Chunk {chunk_index:03d}: {current_path.name} ({chunk_size_mb:.2f} MiB)")

                        chunk_index += 1
                        current_bytes = 0
                        current_path = get_chunk_path(chunk_index)
                        current_file = open_output_fn(current_path, "wt", encoding="utf-8")
                        generated_chunks.append(current_path)

                        # Write duplicate header to top of new split chunk
                        current_file.write(header_line)
                        current_bytes += header_bytes

                    current_file.write(line)
                    current_bytes += line_bytes
        finally:
            current_file.close()
            chunk_size_mb = current_bytes / (1024 * 1024)
            print(f"  • Created Chunk {chunk_index:03d}: {current_path.name} ({chunk_size_mb:.2f} MiB)")

    # ==========================================================================
    # STRATEGY 3: SYNTAX-VALID JSON ARRAYS (.json)
    # ==========================================================================
    elif export_format == "json":
        current_records: List[Dict[str, Any]] = []

        with open_input_fn(input_file, "rt", encoding="utf-8") as f_in:
            # Peek at the first non-whitespace character to determine input JSON structure
            first_char = f_in.read(1)
            f_in.seek(0)

            records_generator: Generator[Dict[str, Any], None, None]

            if first_char == "[":
                # Input source is a standard JSON array `[...]`
                records_generator = (item for item in json.load(f_in))
            else:
                # Input source is line-delimited JSON
                def stream_jsonl():
                    for line in f_in:
                        if line.strip():
                            yield json.loads(line)
                records_generator = stream_jsonl()

            for record in records_generator:
                # Estimate serialized JSON size with formatting overhead
                record_bytes = len(json.dumps(record, ensure_ascii=False).encode("utf-8")) + 2

                if current_bytes + record_bytes > max_bytes and current_records:
                    current_path = get_chunk_path(chunk_index)
                    with open_output_fn(current_path, "wt", encoding="utf-8") as out:
                        json.dump(current_records, out, indent=2, ensure_ascii=False)

                    size_mb = len(json.dumps(current_records).encode("utf-8")) / (1024 * 1024)
                    print(f"  • Created Chunk {chunk_index:03d}: {current_path.name} ({size_mb:.2f} MiB)")
                    generated_chunks.append(current_path)

                    chunk_index += 1
                    current_bytes = 0
                    current_records = []

                current_records.append(record)
                current_bytes += record_bytes

            # Flush remaining records to final chunk
            if current_records:
                current_path = get_chunk_path(chunk_index)
                with open_output_fn(current_path, "wt", encoding="utf-8") as out:
                    json.dump(current_records, out, indent=2, ensure_ascii=False)

                size_mb = len(json.dumps(current_records).encode("utf-8")) / (1024 * 1024)
                print(f"  • Created Chunk {chunk_index:03d}: {current_path.name} ({size_mb:.2f} MiB)")
                generated_chunks.append(current_path)

    print(f"\n--> Splitting Complete! Created {len(generated_chunks)} chunk files in: {output_dir}\n")
    return generated_chunks


# ==============================================================================
# COMMAND LINE INTERFACE PARSER
# ==============================================================================

def main() -> None:
    """
    Parses command-line arguments and coordinates dataset splitting operations.
    """
    parser = argparse.ArgumentParser(
        description="ARCHON Dataset Splitter Tool (.icb & Multi-Format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python split_dataset.py codebase_ingested.icb -f icb -m 15
  python split_dataset.py codebase_ingested.icb -f icb -m 10 -c
  python split_dataset.py dataset.csv -f csv -m 12
        """
    )

    parser.add_argument("input_file", type=str, help="Path to input dataset file (.icb, .jsonl, .csv, .gz, etc.).")
    parser.add_argument("-m", "--max-mb", type=float, default=15.0, help="Maximum split chunk size in MiB (Default: 15.0).")
    parser.add_argument(
        "-f", "--format",
        choices=["icb", "jsonl", "json", "csv", "txt", "md", "markdown"],
        default="icb",
        help="Target export format for split chunks (Default: icb)."
    )
    parser.add_argument("-c", "--compress", action="store_true", help="Enable Gzip compression (.gz output chunks).")

    args = parser.parse_args()

    try:
        split_dataset(
            input_file=Path(args.input_file),
            max_mb=args.max_mb,
            export_format=args.format,
            compress=args.compress
        )
    except Exception as err:
        print(f"[Error] {err}", file=sys.stderr)
        sys.exit(1)


# ==============================================================================
# SCRIPT FOOTER & EXECUTION GUARD
# ==============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[System] Splitting operation cancelled by user. Exiting cleanly.", file=sys.stderr)
        sys.exit(130)
