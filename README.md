#  Codebase Ingestion & Dataset Preparation Toolchain

This toolkit consists of two Python scripts designed to ingest source code repositories, clean text payloads for optimized LLM token consumption, and split oversized dataset files into byte-bounded chunks for RAG vector databases and API batch uploads.

---

## Overview of Scripts

* **`ingest_codebase_exporter.py`**: Ingests local directories and remote Git repositories into structured `.jsonl` or Gzip-compressed `.jsonl.gz` datasets. It handles lossless text minification and streams output directly to standard LangChain `Document` objects.
* **`splitting_codebase_content.py`**: Partitions large dataset files (`.icb`, `.jsonl`, `.json`, `.csv`, `.md`, `.txt`) into target byte-size limits without breaking structural lines, CSV headers, or JSON syntax boundaries.

---

## Prerequisites & Installation

Ensure Python 3.8 or higher is installed along with Git CLI for remote repository processing.

* **Standard Python**: Python 3.8+ (both scripts run natively using only Python's Standard Library).
* **System Dependency**: Git CLI installed and available on your system `PATH` (required only for shallow cloning remote repos).
* **Optional Package**: `langchain-core` (if omitted, `ingest_codebase_exporter.py` automatically uses an internal compatibility fallback stub).

---

## How to Use `ingest_codebase_exporter.py`

Use this script to ingest raw source code, strip unnecessary token bloat, and prepare structured vector store inputs.

### Why Use It:
* **Bandwidth Efficient**: Clones remote Git repos using `--depth 1` to bypass full git histories.
* **Token Saving**: Strips trailing whitespace and duplicate blank lines, cutting context length by 15–30% without affecting functional logic.
* **Memory Efficient**: Streams data with low memory usage directly into compressed Gzip files.

### Step-by-Step Instructions:

1. **Ingest local folders and remote repos into a compressed dataset**:
   ```bash
   python ingest_codebase_exporter.py /path/to/project https://github.com/psf/requests.git -o codebase.jsonl.gz -c
   ```
2. **Ingest without applying text minification**:
   ```bash
   python ingest_codebase_exporter.py /path/to/project -o codebase.jsonl --no-minify
   ```
3. **Load an existing dataset straight into memory for LangChain vector testing**:
   ```bash
   python ingest_codebase_exporter.py -o codebase.jsonl.gz --load-only
   ```

---

## How to Use `splitting_codebase_content.py`

Use this script to break down large dataset files into smaller, byte-bounded chunks for API upload limits or vector database batching.

### Why Use It:
* **Structural Safety**: Never splits records or JSON objects mid-line.
* **Header Preservation**: Automatically duplicates top CSV headers across all generated split files.
* **JSON Syntax Validity**: Formats JSON chunks as standard `[...]` arrays for instant parsing.

### Step-by-Step Instructions:

1. **Split an `.icb` file into 15 MiB chunk files**:
   ```bash
   python splitting_codebase_content.py codebase_ingested.icb -f icb -m 15
   ```
2. **Split and Gzip-compress chunks to 10 MiB limits**:
   ```bash
   python splitting_codebase_content.py codebase_ingested.icb -f icb -m 10 -c
   ```
3. **Split a large CSV file while preserving column headers on every chunk**:
   ```bash
   python splitting_codebase_content.py dataset.csv -f csv -m 12
   ```
4. **Convert and split a `.jsonl` dataset into valid JSON array files (`.json`)**:
   ```bash
   python splitting_codebase_content.py codebase_ingested.jsonl -f json -m 15
   ```
