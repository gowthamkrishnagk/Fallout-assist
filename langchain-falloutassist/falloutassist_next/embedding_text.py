"""Token-bounded embedding inputs; original evidence is never rewritten."""

from __future__ import annotations


class EmbeddingWindow:
    def __init__(self, tokenizer, limit: int):
        self.tokenizer = tokenizer
        self.limit = int(limit)
        if self.limit < 32:
            raise ValueError("Embedding model input window must be at least 32 tokens")

    def _offsets(self, text: str) -> list[tuple[int, int]]:
        # HuggingFaceEmbeddings replaces newlines before encoding. Keep offsets
        # aligned with that input while returning slices of the original source.
        encoded = self.tokenizer(text.replace("\n", " "), add_special_tokens=False,
                                 truncation=False, return_offsets_mapping=True)
        return [tuple(offset) for offset in encoded["offset_mapping"] if offset[1] > offset[0]]

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text.replace("\n", " "), add_special_tokens=True,
                                         truncation=False))

    def trim(self, text: str, tokens: int) -> str:
        if tokens <= 0:
            return ""
        offsets = self._offsets(text)
        return text[:offsets[tokens - 1][1]] if len(offsets) > tokens else text

    def failure_text(self, item: dict, budget: int | None = None) -> str:
        """Reserve space for every failure field, with the error code first."""
        budget = budget or self.limit - 4
        code = str(item.get("error_code") or "")
        header = f"Error code: {code}\n" if code else ""
        fields = [("Error", item.get("error", "")), ("Failed Step", item.get("step", "")),
                  ("Order Reason", item.get("order_reason", "")), ("Summary", item.get("summary", ""))]
        overhead = self.count(header + "\n".join(label + ": " for label, _ in fields))
        per_field = max(1, (budget - overhead) // len(fields))
        result = header + "\n".join(f"{label}: {self.trim(str(value), per_field)}"
                                    for label, value in fields)
        while self.count(result) > budget:
            result = self.trim(result, max(1, len(self._offsets(result)) - 1))
        return result

    def passages(self, text: str, prefix: str = "", overlap: int = 24) -> list[str]:
        """Create overlapping vector passages without dropping the source tail."""
        offsets = self._offsets(text)
        if not offsets:
            return []
        budget = self.limit - self.count(prefix + "\nReference: ") - 4
        if budget < 8:
            raise ValueError("Embedding prefix leaves too little room for reference text")
        overlap = min(overlap, budget // 4)
        result = []
        start = 0
        while start < len(offsets):
            end = min(start + budget, len(offsets))
            body = text[offsets[start][0]:offsets[end - 1][1]]
            while self.count(prefix + "\nReference: " + body) > self.limit:
                end -= 1
                if end <= start:
                    raise ValueError("A reference token exceeds the embedding input window")
                body = text[offsets[start][0]:offsets[end - 1][1]]
            result.append(prefix + "\nReference: " + body)
            if end == len(offsets):
                break
            start = max(start + 1, end - overlap)
        return result
