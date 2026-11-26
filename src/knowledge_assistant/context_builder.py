from knowledge_assistant.config import RAG_MAX_CONTEXT_CHARS
from knowledge_assistant.models import Citation, ContextBundle
from knowledge_assistant.retrieval import SearchResult


class ContextBuilder:
    """Construct source-labelled evidence for RAG generation."""

    def __init__(
        self,
        max_context_chars: int = RAG_MAX_CONTEXT_CHARS,
    ):
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive.")

        self.max_context_chars = max_context_chars

    @staticmethod
    def _format_chunk(
        result: SearchResult,
        citation_id: int,
    ) -> str:
        """Format a retrieved chunk with source metadata."""

        chunk = result.chunk

        header = f"[{citation_id}] Source: {chunk.source} | Category: {chunk.category}"

        if chunk.page_number is not None:
            header += f" | Page: {chunk.page_number}"

        return f"{header}\n{chunk.text.strip()}"

    @staticmethod
    def _create_citation(
        result: SearchResult,
        citation_id: int,
    ) -> Citation:
        """Create citation metadata from a search result."""

        chunk = result.chunk

        return Citation(
            citation_id=citation_id,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source=chunk.source,
            category=chunk.category,
            score=result.score,
            page_number=chunk.page_number,
        )

    def build(
        self,
        results: list[SearchResult],
    ) -> ContextBundle:
        """Build evidence context within the character budget."""

        sections = []
        citations = []
        seen_chunk_ids = set()

        character_count = 0
        truncated = False

        separator = "\n\n"

        for result in sorted(
            results,
            key=lambda item: item.rank,
        ):
            chunk = result.chunk

            # Ignore duplicate chunks.
            if chunk.chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(chunk.chunk_id)

            # Ignore empty evidence.
            if not chunk.text.strip():
                continue

            citation_id = len(citations) + 1

            section = self._format_chunk(
                result,
                citation_id,
            )

            separator_length = len(separator) if sections else 0

            required_chars = len(section) + separator_length

            if character_count + required_chars > self.max_context_chars:
                truncated = True
                continue

            sections.append(section)

            citations.append(
                self._create_citation(
                    result,
                    citation_id,
                )
            )

            character_count += required_chars

        context_text = separator.join(sections)

        return ContextBundle(
            text=context_text,
            citations=citations,
            retrieved_count=len(results),
            included_count=len(citations),
            character_count=len(context_text),
            truncated=truncated,
        )
