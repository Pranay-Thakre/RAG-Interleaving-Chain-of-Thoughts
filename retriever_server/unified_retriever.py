from typing import List, Dict

try:
    from retriever_server.elasticsearch_retriever import ElasticsearchRetriever
    from retriever_server.vector_retriever import VectorRetriever
except ImportError:
    from elasticsearch_retriever import ElasticsearchRetriever
    from vector_retriever import VectorRetriever



class UnifiedRetriever:
    """
    Unified retriever supporting both Elasticsearch BM25 and fallback Vector/In-Memory retriever.
    """

    def __init__(
        self,
        host: str = "http://localhost/",
        port: int = 9200,
    ):
        try:
            self._elasticsearch_retriever = ElasticsearchRetriever(host=host, port=port)
        except Exception:
            self._elasticsearch_retriever = None
        self._vector_retriever = VectorRetriever()

    def retrieve_from_elasticsearch(
        self,
        query_text: str,
        max_hits_count: int = 3,
        max_buffer_count: int = 100,
        document_type: str = "paragraph_text",
        allowed_titles: List[str] = None,
        allowed_paragraph_types: List[str] = None,
        paragraph_index: int = None,
        corpus_name: str = None,
    ) -> List[Dict]:

        assert document_type in ("title", "paragraph_text", "title_paragraph_text")

        if paragraph_index is not None:
            assert (
                document_type == "paragraph_text"
            ), "paragraph_index not valid input for the document_type of paragraph_text."

        # Try Elasticsearch first, fall back to VectorRetriever if ES fails or is offline
        if self._elasticsearch_retriever is not None:
            try:
                if document_type in ("paragraph_text", "title_paragraph_text"):
                    is_abstract = True if corpus_name == "hotpotqa" else None
                    query_title_field_too = document_type == "title_paragraph_text"
                    paragraphs_results = self._elasticsearch_retriever.retrieve_paragraphs(
                        query_text=query_text,
                        is_abstract=is_abstract,
                        max_hits_count=max_hits_count,
                        allowed_titles=allowed_titles,
                        allowed_paragraph_types=allowed_paragraph_types,
                        paragraph_index=paragraph_index,
                        corpus_name=corpus_name,
                        query_title_field_too=query_title_field_too,
                        max_buffer_count=max_buffer_count,
                    )
                    if paragraphs_results:
                        return paragraphs_results
                elif document_type == "title":
                    paragraphs_results = self._elasticsearch_retriever.retrieve_titles(
                        query_text=query_text, max_hits_count=max_hits_count, corpus_name=corpus_name
                    )
                    if paragraphs_results:
                        return paragraphs_results
            except Exception as e:
                print(f"[UnifiedRetriever] ES failed ({e}), falling back to VectorRetriever.")

        # Fallback to vector retriever
        if document_type in ("paragraph_text", "title_paragraph_text"):
            return self._vector_retriever.retrieve_paragraphs(
                corpus_name=corpus_name,
                query_text=query_text,
                allowed_titles=allowed_titles,
                allowed_paragraph_types=allowed_paragraph_types,
                paragraph_index=paragraph_index,
                max_buffer_count=max_buffer_count,
                max_hits_count=max_hits_count,
            )
        elif document_type == "title":
            return self._vector_retriever.retrieve_titles(
                corpus_name=corpus_name,
                query_text=query_text,
                max_hits_count=max_hits_count,
            )
        return []
