# COD Aging Theories Pipeline

This repository implements an end-to-end workflow for collecting, parsing, filtering, and analyzing literature about theories of aging. The pipeline ingests open-access corpora, induces and refines an ontology of aging theories, links documents to ontology nodes, iteratively expands the search space, and produces analyst-facing reports and visualizations.

## Artifact Specifications
All required artifacts share deterministic naming conventions and storage locations so contributors can validate results and reproduce the pipeline. Paths are relative to the repository root unless stated otherwise.

### 1. Corpus Metadata and Full Text Store
- **Location:** `storage/docs_store/`
- **Naming:**
  - Parsed documents: `storage/docs_store/corpus_v{MAJOR}.{MINOR}.{PATCH}/docs.parquet`
  - License checklist: `storage/docs_store/corpus_v{MAJOR}.{MINOR}.{PATCH}/licenses.csv`
  - Raw JSON lines from ingest: `storage/docs_store/raw/corpus_seed_YYYYMMDD.jsonl`
- **Format:**
  - `docs.parquet` conforms to the *Document* schema from `AGENT.md` with strongly typed columns (e.g., arrays for `source_ids`, structs for `authors`).
  - `corpus_seed_*.jsonl` stores one JSON object per record with the same schema (see example below).
- **Schema Fields:**
  ```json
  {
    "id": "doi:10.1234/abcd",
    "source_ids": ["openalex:W...", "pubmed:12345"],
    "title": "...",
    "abstract": "...",
    "year": 2021,
    "venue": "Journal ...",
    "authors": [{"name":"...", "orcid":"..."}],
    "license": "CC-BY-4.0",
    "oa_status": "gold|green|hybrid|bronze|closed",
    "fulltext_url": "https://...",
    "fulltext_path": "s3://.../doi_10_1234_abcd.pdf",
    "text_checksum": "sha256:...",
    "entities": {
      "organisms": ["Homo sapiens","Mus musculus"],
      "interventions": ["rapamycin","CR"],
      "outcomes": ["lifespan","healthspan"],
      "theory_mentions": ["antagonistic pleiotropy","disposable soma"]
    },
    "relevance_score": 0.0,
    "is_review": true,
    "notes": "provenance/logs..."
  }
  ```
- **Generation Commands:**
  - `make seed_ingest` → creates `corpus_seed_*.jsonl` (requires API credentials and network access).
  - `make resolve_parse` → produces the versioned Parquet store and license checklist from the ingested seed (depends on `make seed_ingest`).

### 2. Ontology Releases
- **Location:** `releases/ontology_v{MAJOR}.{MINOR}.{PATCH}.json`
- **Naming:** Semantic versioning increments when ontology nodes, relations, or descriptions change.
- **Format:** JSON array of ontology node objects adhering to the *Ontology* schema from `AGENT.md`.
- **Schema Fields:**
  ```json
  {
    "theory_id": "theory:antagonistic_pleiotropy",
    "label": "Antagonistic pleiotropy",
    "synonyms": ["АП-гипотеза", "antagonistic pleiotropy theory"],
    "parents": ["theory:evolutionary"],
    "children": ["theory:age_specific_effects"],
    "description": "...",
    "doc_links": [
      {"doc_id":"doi:10.1234/...","relation":"supports","evidence_type":"empirical"}
    ],
    "stats": {"n_docs": 37, "n_supports": 22, "n_refutes": 4}
  }
  ```
- **Generation Commands:**
  - `make induce` → creates the initial ontology release from curated reviews (depends on `make resolve_parse`).
  - `make refine` → updates ontology structure and statistics; emits a new `ontology_v*.json` file.

### 3. Document ↔ Theory Graph Structures
- **Location:**
  - `storage/graph/graph_snapshot_v{MAJOR}.{MINOR}.{PATCH}.json`
  - Optional graph exchange formats (e.g., GraphML) may be stored in `storage/graph/export/`.
- **Naming:** Versioned snapshots aligned with the ontology version used for generation. Include matching semver to ease cross-referencing (e.g., `graph_snapshot_v1.2.0.json` pairs with `ontology_v1.2.0.json`).
- **Format:** JSON-LD compatible object containing:
  ```json
  {
    "ontology_version": "1.2.0",
    "document_version": "corpus_v1.2.0",
    "nodes": [
      {"id": "theory:antagonistic_pleiotropy", "type": "theory", "label": "Antagonistic pleiotropy"},
      {"id": "doi:10.1234/abcd", "type": "document", "title": "..."}
    ],
    "edges": [
      {
        "source": "doi:10.1234/abcd",
        "target": "theory:antagonistic_pleiotropy",
        "relation": "supports",
        "evidence_spans": [{"text": "...", "section": "Results"}],
        "confidence": 0.87
      }
    ],
    "metadata": {
      "generated_at": "2024-04-08T12:00:00Z",
      "generator": "make refine"
    }
  }
  ```
- **Generation Commands:**
  - `make refine` → refreshes theory graph relationships using refined ontology and corpus embeddings.
  - `make expand_search` → after retrieving new documents, updates graph to include new edges.
  - `make iterate` → runs steps 4–6 (refine, expand, update) repeatedly until convergence; outputs consolidated graph snapshots.

### 4. Iterative Query Logs
- **Location:** `logs/query_iterations/`
- **Naming:** `iteration_{NN}_queries.jsonl` where `NN` is a zero-padded integer incremented per loop (e.g., `iteration_03_queries.jsonl`).
- **Format:** JSON lines, each entry generated via the *Generation of search phrases* schema in `AGENT.md`:
  ```json
  {
    "theory_id": "theory:mitochondrial_free_radical",
    "boolean_queries": ["(\"mitochondrial free radical\" AND aging)"],
    "positive_terms": ["ROS", "mitochondrial DNA"],
    "negative_terms": ["battery aging"],
    "organism_terms": ["Homo sapiens", "Drosophila"],
    "method_terms": ["meta-analysis", "GWAS"],
    "generated_at": "2024-04-08T12:30:00Z"
  }
  ```
- **Generation Commands:**
  - `make expand_search` → invokes LLM prompt templates and stores generated queries.
  - `make iterate` → ensures logs accumulate per cycle while orchestrating downstream retrieval.

### 5. Reporting and Visualization Outputs
- **Location:**
  - `eval_reporting/outputs/report_v{MAJOR}.{MINOR}.{PATCH}.md`
  - `eval_reporting/outputs/metrics_v{MAJOR}.{MINOR}.{PATCH}.json`
  - Visual assets (e.g., PNG, HTML) under `eval_reporting/outputs/viz/`
- **Format:**
  - Markdown report summarizing coverage, accuracy, and exemplar evidence spans.
  - Metrics JSON includes aggregate statistics (precision/recall, ontology coverage, LLM decision agreement).
  - Visualization data exported as HTML (Cytoscape.js bundle) or CSV tables for charts.
- **Generation Commands:**
  - `make viz` → creates interactive graph visualizations in `eval_reporting/outputs/viz/` (depends on up-to-date graph snapshot).
  - `make report` → compiles metrics and narrative report from evaluation results (depends on `make viz` and latest ontology/graph artifacts).

## Pipeline Orchestration Summary
The `Makefile` coordinates discrete modules. Each target logs detailed provenance under `logs/` and writes artifacts to versioned directories for reproducibility.

| Stage | Command | Depends On | Primary Inputs | Primary Outputs |
|-------|---------|------------|----------------|-----------------|
| Environment bootstrap | `make bootstrap` | – | `.env`, API keys, Docker (for GROBID) | Virtualenv, configured services |
| Seed ingest & filtering | `make seed_ingest` | `make bootstrap` | Seed queries, API credentials | `storage/docs_store/raw/corpus_seed_*.jsonl`, ingest logs |
| OA resolution & parsing | `make resolve_parse` | `make seed_ingest` | Seed corpus JSONL, OA APIs | `storage/docs_store/corpus_v*/docs.parquet`, license checklist |
| Ontology induction | `make induce` | `make resolve_parse` | Parsed reviews, entity extractions | `releases/ontology_v*.json`, intermediate logs |
| Graph refinement | `make refine` | `make induce` | Ontology release, corpus embeddings | `storage/graph/graph_snapshot_v*.json`, updated ontology |
| Query expansion | `make expand_search` | `make refine` | Ontology nodes, prior logs | `logs/query_iterations/iteration_*.jsonl`, retrieval task configs |
| Iterative loop | `make iterate` | `make expand_search` | Previous artifacts | Updated corpus/graph snapshots, appended logs |
| Visualization | `make viz` | `make refine` | Graph snapshot | `eval_reporting/outputs/viz/*` |
| Reporting | `make report` | `make viz` | Metrics JSON, ontology statistics | `eval_reporting/outputs/report_v*.md`, `metrics_v*.json` |

### Dependency Notes
- `make iterate` internally runs `make refine` and `make expand_search` for each cycle. Ensure previous outputs have distinct version numbers to avoid overwriting.
- Each stage should commit artifacts with matching semantic versions (e.g., corpus, ontology, graph, report) to simplify provenance tracking.
- Logs from LLM decisions (classification, ontology extraction, relation tagging) are persisted in `logs/decisions/` to audit prompts and rationales.

## Validation Workflow
1. Run the orchestration commands sequentially (`make bootstrap` → `make report`) to regenerate artifacts.
2. Verify schema compliance by running JSON schema or PyArrow validation scripts against the outputs before publishing new versions.
3. Tag releases after confirming that ontology, graph, and report versions align and that checksum hashes for documents remain unchanged across reruns.

By adhering to these conventions, contributors can generate, compare, and validate all critical artifacts for the COD aging theories knowledge graph with minimal ambiguity.
