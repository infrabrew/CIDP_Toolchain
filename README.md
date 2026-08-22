#Dataset Ingestion & Processing Pipeline

A suite of zero-dependency, high-performance Python utilities designed to convert raw documents, web content, and local/remote codebases into clean, compressed, minified line-delimited datasets (`.jsonl.gz`), and safely partition oversized files into target byte-size chunks without breaking line or structural record boundaries[cite: 1, 2, 4].

---

## What These Scripts Are & Why We Use Them

When building Retrieval-Augmented Generation (RAG) applications, indexing repositories into vector databases, or preparing fine-tuning datasets, raw code and unformatted binary/web content introduce significant friction:

* **Unstructured Binary Documents:** PDFs, Word documents (`.docx`), and web pages contain binary streams, HTML tags, headers, footers, and layout noise that clutter LLM context windows[cite: 4].
* **Token & Bandwidth Overhead:** Trailing whitespace, duplicate blank lines, and downloading full Git commit histories bloat token usage and network transfers[cite: 2].
* **Strict Payload & Upload Limits:** LLM batch processing APIs, embedding endpoints, and vector stores enforce strict per-file size limits[cite: 1].
* **Record Boundary Corruption:** Standard Unix commands like `split` cut mid-line, breaking JSON syntax, CSV rows, or code logic[cite: 1].

This suite provides an end-to-end processing pipeline to solve these challenges across three modules[cite: 1, 2, 4]:

1. **`codebase_pipeline.py` (Unified Ingestion & Auto-Splitting Engine):** Crawls local folders and shallow-clones remote Git repos (`--depth 1`), automatically detects and parses `.pdf` and `.docx` files into clean Markdown on the fly, applies lossless text minification (stripping 15–30% token overhead), and automatically splits generated output files into target byte-size chunks (`--split-mb`)[cite: 1, 2, 4].
2. **`doc_converter.py` (Zero-Dependency Document & Web Ingestion Engine):** Standalone converter that converts PDF files, Word documents (`.docx`), and remote HTTP/HTTPS web URLs into clean Markdown (`.md`) or plain text (`.txt`) using **Python Standard Library only**[cite: 4].
3. **`split_dataset.py` (Multi-Format Dataset Splitter):** Standalone partitioner for oversized `.icb`, `.jsonl`, `.json`, or `.csv` files into exact byte-bounded chunk files (e.g., max 15 MiB per file) while guaranteeing line, record, and header syntax integrity[cite: 1].

---

## Suite Architecture & Tool Summary

| Script Name | Primary Input Sources | Key Capabilities | Output Formats |
| :--- | :--- | :--- | :--- |
| **`codebase_pipeline.py`** | Local directories, remote Git URLs[cite: 2] | Native on-the-fly PDF/DOCX conversion[cite: 2, 4], shallow Git fetching (`--depth 1`)[cite: 2], lossless minification[cite: 2], native LangChain `Document` loader[cite: 2], and optional auto-chunk splitting (`--split-mb`)[cite: 1, 2]. | `.jsonl`, `.jsonl.gz`, plus auto-created `_chunks/` directories[cite: 1, 2] |
| **`doc_converter.py`** | Local `.pdf`, `.docx`, `.doc`, and Web URLs (`http://`, `https://`)[cite: 4] | Standard Library native parsing (`zipfile`/`xml` for Word[cite: 4], `html.parser` for Web[cite: 4], system/regex for PDF[cite: 4]). Cleans page artifacts and HTML DOM boilerplate[cite: 4]. | `.md`, `.txt`, `.md.gz`, `.txt.gz`[cite: 4] |
| **`split_dataset.py`** | `.icb`, `.jsonl`, `.json`, `.csv`, `.txt`, `.md`[cite: 1] | Byte-bounded chunking, CSV header replication across split files[cite: 1], valid JSON array (`[...]`) wrapping[cite: 1]. | `.icb`, `.jsonl`, `.json`, `.csv`, `.txt`, `.md` (optional `.gz`)[cite: 1] |

---

## Requirements & Prerequisites

* **Python:** 3.8 or higher[cite: 1, 2, 4]
* **Dependencies:** **Zero external PyPI packages required** for standard execution[cite: 1, 4]. Standard Library only (`gzip`, `json`, `csv`, `urllib`, `html.parser`, `zipfile`, `xml.etree.ElementTree`, `re`, `subprocess`, `argparse`, `pathlib`)[cite: 1, 4].
* **Optional System Dependencies:**
  * `git` CLI (required only if pulling remote Git repositories via `codebase_pipeline.py`)[cite: 2].
  * `pdftotext` CLI (optional for PDF parsing in `codebase_pipeline.py` and `doc_converter.py`; falls back to an internal pure-Python stream regex engine if absent)[cite: 4].
  * `langchain-core` (optional for `codebase_pipeline.py`; uses an internal fallback stub if missing)[cite: 2].

---

## Step-by-Step Usage Guide

### Step 1: All-In-One Repository & Document Ingestion with Auto-Splitting

Use `codebase_pipeline.py` to ingest codebases, local directories, or remote Git repositories[cite: 2]. Any `.pdf`, `.docx`, or `.doc` files contained inside the target folders or Git repos are automatically converted to clean Markdown during traversal[cite: 2, 4].

#### Example A: Ingest Local Folders & Remote Repos with Automatic Document Conversion & 10 MiB Auto-Splitting
```bash
python codebase_pipeline.py /path/to/project [https://github.com/vllm-project/vllm.git](https://github.com/vllm-project/vllm.git) -o codebase_ingested.jsonl.gz -c --split-mb 10
```[cite: 1, 2, 4]

#### Example B: Ingest Without Lossless Minification
```bash
python codebase_pipeline.py /path/to/project -o codebase_ingested.jsonl --no-minify
```[cite: 2]

#### CLI Flags (`codebase_pipeline.py`)
* `sources`: Local folder paths or remote Git HTTP/HTTPS/SSH URLs[cite: 2].
* `-o, --output`: Output file destination (Default: `codebase_ingested.jsonl`)[cite: 2].
* `-c, --compress`: Enables Gzip compression (automatically appends `.gz`)[cite: 2].
* `--no-minify`: Disables trailing space stripping and empty line collapsing[cite: 2].
* `--split-mb`: Automatically splits output dataset into max `N` MiB chunks upon completion[cite: 1, 2].

---

### Step 2: Standalone Document & Web Scraping Conversion

Use `doc_converter.py` if you need to pre-convert standalone binary documents or scrape web pages into Markdown or text prior to pipeline processing[cite: 4].

#### Example A: Convert Local PDFs, Word Files, and Web URLs Simultaneously
```bash
python doc_converter.py /path/to/docs [https://docs.python.org/3/library/unittest.html](https://docs.python.org/3/library/unittest.html) -f md
```[cite: 4]

#### Example B: Convert a Single File to Plain Text in a Custom Folder
```bash
python doc_converter.py specification.pdf -o ./clean_specs -f txt
```[cite: 4]

#### CLI Flags (`doc_converter.py`)
* `sources`: Local `.pdf`/`.docx` file paths, folder paths, or HTTP/HTTPS URLs[cite: 4].
* `-o, --output`: Destination directory (Default: `./converted_docs`)[cite: 4].
* `-f, --format`: Target extension format: `md` or `txt` (Default: `md`)[cite: 4].
* `-c, --compress`: Enables Gzip compression (`.gz` extension)[cite: 4].

---

### Step 3: Standalone Dataset Partitioning

Use `split_dataset.py` to break down existing standalone datasets or custom export files into byte-bounded chunks[cite: 1].

#### Example A: Split an `.icb` or `.jsonl` File into 15 MiB Chunks
```bash
python split_dataset.py codebase_ingested.jsonl -f jsonl -m 15
```[cite: 1]

#### Example B: Split CSV Files While Duplicating Headers Across Output Chunks
```bash
python split_dataset.py dataset.csv -f csv -m 12
