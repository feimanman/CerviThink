#!/usr/bin/env python3
"""Fetch PubMed snippets for CerviCoT retrieval."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_QUERIES = {
    "HSIL": '"high-grade squamous intraepithelial lesion" cervical cytology nucleus chromatin',
    "ASC-H": '"ASC-H" cervical cytology atypical squamous cells cannot exclude HSIL',
    "LSIL": '"low-grade squamous intraepithelial lesion" cervical cytology koilocytosis',
    "ASC-US": '"ASC-US" cervical cytology atypical squamous cells undetermined significance',
    "Normal": 'normal cervical cytology squamous cells nucleus chromatin',
}

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output knowledge JSONL.")
    parser.add_argument("--retmax", type=int, default=20, help="PubMed records per label.")
    parser.add_argument("--email", default=None, help="NCBI contact email.")
    parser.add_argument("--api-key", default=None, help="Optional NCBI API key.")
    parser.add_argument("--sleep", type=float, default=0.34, help="Delay between NCBI requests.")
    parser.add_argument("--query", action="append", default=[], help="Override as LABEL=QUERY.")
    return parser.parse_args()


def request_json(endpoint: str, params: dict[str, str | int]) -> dict:
    params = {**params, "retmode": "json"}
    url = f"{BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def request_xml(endpoint: str, params: dict[str, str | int]) -> ET.Element:
    url = f"{BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return ET.fromstring(response.read())


def esearch(term: str, retmax: int, email: str | None, api_key: str | None) -> list[str]:
    params: dict[str, str | int] = {
        "db": "pubmed",
        "term": term,
        "retmax": retmax,
        "sort": "relevance",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    data = request_json("esearch.fcgi", params)
    return data.get("esearchresult", {}).get("idlist", [])


def efetch(pmids: list[str], email: str | None, api_key: str | None) -> list[dict[str, str]]:
    if not pmids:
        return []
    params: dict[str, str | int] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    root = request_xml("efetch.fcgi", params)
    rows = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID") or ""
        title = " ".join("".join(article.find(".//ArticleTitle").itertext()).split()) if article.find(".//ArticleTitle") is not None else ""
        abstract_parts = []
        for abstract in article.findall(".//AbstractText"):
            label = abstract.attrib.get("Label")
            text = " ".join("".join(abstract.itertext()).split())
            if label and text:
                abstract_parts.append(f"{label}: {text}")
            elif text:
                abstract_parts.append(text)
        journal = article.findtext(".//Journal/Title") or ""
        year = article.findtext(".//PubDate/Year") or ""
        rows.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": " ".join(abstract_parts),
                "journal": journal,
                "year": year,
            }
        )
    return rows


def query_map(overrides: list[str]) -> dict[str, str]:
    queries = dict(DEFAULT_QUERIES)
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"Invalid --query value: {item!r}. Use LABEL=QUERY.")
        label, query = item.split("=", 1)
        queries[label.strip()] = query.strip()
    return queries


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen_pmids: set[str] = set()
    with output.open("w", encoding="utf-8") as handle:
        for label, query in query_map(args.query).items():
            pmids = esearch(query, args.retmax, args.email, args.api_key)
            time.sleep(args.sleep)
            for row in efetch(pmids, args.email, args.api_key):
                if row["pmid"] in seen_pmids:
                    continue
                seen_pmids.add(row["pmid"])
                row["label"] = label
                row["query"] = query
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            time.sleep(args.sleep)
    print(f"Wrote {len(seen_pmids)} PubMed snippets to {output}")


if __name__ == "__main__":
    main()
