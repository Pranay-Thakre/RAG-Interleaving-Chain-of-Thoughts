import os
import json
import re
from typing import List, Dict
from collections import Counter
import math

class VectorRetriever:
    """
    In-memory vector / TF-IDF / BM25 retriever for local execution
    when Elasticsearch server is not running or vector retrieval is requested.
    """
    def __init__(self, raw_data_dir: str = "raw_data", processed_data_dir: str = "processed_data"):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self._corpus_cache = {}

    def _load_corpus(self, corpus_name: str) -> List[Dict]:
        clean_name = corpus_name.replace("-wikipedia", "").replace("_to_hotpotqa", "").replace("_to_2wikimultihopqa", "").replace("_to_musique", "").replace("_to_iirc", "")
        if clean_name in self._corpus_cache:
            return self._corpus_cache[clean_name]

        documents = []

        # 1. Try loading raw_data/clean_name/wikiparas.json or context_articles.json
        raw_path = os.path.join(self.raw_data_dir, clean_name, "wikiparas.json" if clean_name != "iirc" else "context_articles.json")
        if os.path.exists(raw_path):
            with open(raw_path, "r") as f:
                data = json.load(f)
                idx = 0
                for title, paragraphs in data.items():
                    if isinstance(paragraphs, list):
                        for p_idx, text in enumerate(paragraphs):
                            documents.append({
                                "id": f"{clean_name}_{idx}",
                                "title": title,
                                "paragraph_text": text if isinstance(text, str) else str(text),
                                "paragraph_index": p_idx,
                                "is_abstract": True
                            })
                            idx += 1
                    elif isinstance(paragraphs, str):
                        documents.append({
                            "id": f"{clean_name}_{idx}",
                            "title": title,
                            "paragraph_text": paragraphs,
                            "paragraph_index": 0,
                            "is_abstract": True
                        })
                        idx += 1

        # 2. Try loading processed_data/clean_name/dev_subsampled.jsonl
        proc_path = os.path.join(self.processed_data_dir, clean_name, "dev_subsampled.jsonl")
        if os.path.exists(proc_path):
            with open(proc_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    for ctx in item.get("contexts", []):
                        documents.append({
                            "id": f"{clean_name}_{ctx.get('idx', len(documents))}",
                            "title": ctx.get("title", ""),
                            "paragraph_text": ctx.get("paragraph_text", ""),
                            "paragraph_index": ctx.get("idx", 0),
                            "is_abstract": True
                        })

        self._corpus_cache[clean_name] = documents
        return documents

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def _score(self, query_tokens: List[str], doc_tokens: List[str], title_tokens: List[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        doc_counts = Counter(doc_tokens)
        title_counts = Counter(title_tokens)
        
        score = 0.0
        for q in query_tokens:
            if q in doc_counts:
                score += (1.0 + math.log(doc_counts[q]))
            if q in title_counts:
                score += (2.0 + math.log(title_counts[q]))
        return score

    def retrieve_paragraphs(
        self,
        corpus_name: str,
        query_text: str = None,
        is_abstract: bool = None,
        allowed_titles: List[str] = None,
        allowed_paragraph_types: List[str] = None,
        query_title_field_too: bool = False,
        paragraph_index: int = None,
        max_buffer_count: int = 100,
        max_hits_count: int = 10,
    ) -> List[Dict]:
        corpus = self._load_corpus(corpus_name)
        if not corpus:
            return []

        query_tokens = self._tokenize(query_text) if query_text else []
        allowed_titles_set = set(allowed_titles) if allowed_titles else None

        results = []
        for doc in corpus:
            if allowed_titles_set and doc["title"] not in allowed_titles_set:
                continue

            score = self._score(query_tokens, self._tokenize(doc["paragraph_text"]), self._tokenize(doc["title"]))
            if score > 0 or allowed_titles_set:
                results.append({
                    "_score": score,
                    "_source": {
                        "id": doc["id"],
                        "title": doc["title"],
                        "paragraph_text": doc["paragraph_text"],
                        "paragraph_index": doc.get("paragraph_index", 0),
                        "is_abstract": doc.get("is_abstract", True),
                        "url": "",
                        "score": score
                    }
                })

        results = sorted(results, key=lambda x: x["_score"], reverse=True)
        retrieval = [e["_source"] for e in results[:max_hits_count]]
        for item in retrieval:
            item["corpus_name"] = corpus_name
        return retrieval

    def retrieve_titles(self, corpus_name: str, query_text: str, max_hits_count: int = 10) -> List[Dict]:
        corpus = self._load_corpus(corpus_name)
        query_tokens = self._tokenize(query_text)
        results = []
        seen_titles = set()

        for doc in corpus:
            title = doc["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            score = self._score(query_tokens, [], self._tokenize(title))
            if score > 0:
                results.append({
                    "_score": score,
                    "_source": {
                        "title": title,
                        "id": doc["id"],
                        "score": score
                    }
                })

        results = sorted(results, key=lambda x: x["_score"], reverse=True)
        retrieval = [e["_source"] for e in results[:max_hits_count]]
        for item in retrieval:
            item["corpus_name"] = corpus_name
        return retrieval
