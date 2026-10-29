"""Problem (search_docs): search the company documents and read them in windows."""

import re

from cs329z_hw1.pipelines.bm25 import build_index, search as bm25_search
from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.tokens import CHARS_PER_TOKEN, count_tokens

SNIPPET_MAX_TOKENS = 150
WINDOW_MAX_TOKENS = 1500


# Index whole documents with BM25; keep the documents and an ID lookup for reading.
def build_doc_index(docs):
    return {
        "documents": list(docs),
        "by_id": {doc["doc_id"]: doc for doc in docs},
        "bm25": build_index([doc["title"] + "\n" + doc["text"] for doc in docs]),
    }


# Count the distinct query terms that appear in one line.
def count_overlap(line, query_terms):
    # Tokenize the line, remove duplicates, and count the terms shared
    # with the query. The & operator computes the set intersection.
    return len(set(tokenize(line)) & query_terms)


# Select a relevant line and show original text near a matching query term.
# query_terms is a set of tokenized, deduplicated search terms.
def make_snippet(text, query_terms, snippet_max_tokens=SNIPPET_MAX_TOKENS):
    lines = text.splitlines()

    if not lines:
        return "", 1

    best_line = 0

    # Start below zero so the first line is selected even if it has no matches.
    best_overlap = -1

    # Prefer the line containing the most distinct query terms.
    # Keep the earlier line when overlap scores are equal.
    # Skip headings (lines starting with "#") here, because they mostly repeat the title.
    for position, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue

        overlap = count_overlap(line, query_terms)

        if overlap > best_overlap:
            best_line = position
            best_overlap = overlap

    # If no ordinary line shares a query term, let headings compete too.
    if best_overlap <= 0:
        for position, line in enumerate(lines):
            overlap = count_overlap(line, query_terms)

            if overlap > best_overlap:
                best_line = position
                best_overlap = overlap

    line = lines[best_line]

    # anchor is the character position around which to build the snippet.
    # Default to the beginning of the line if no matching term is found.
    anchor = 0

    # Locate the first matching term within the selected line.
    for match in re.finditer(r"[a-z0-9]+", line.lower()):
        if match.group() in query_terms:
            anchor = match.start()
            break

    # Keep up to 120 characters before the match for context,
    # without allowing the starting position to become negative.
    start = max(0, anchor - 120)

    # If the starting position falls inside a word, move it backward
    # until the preceding character is whitespace or the line begins.
    while start > 0 and not line[start - 1].isspace():
        start -= 1

    # The provided token counter uses four characters per token.
    max_chars = snippet_max_tokens * CHARS_PER_TOKEN

    # Include following lines when the selected line is short.
    passage = "\n".join(lines[best_line:])
    snippet = passage[start:start + max_chars]

    # Strip surrounding whitespace and convert the zero-based list position
    # into a one-based document line number.
    return snippet.strip(), best_line + 1


# Return document IDs, relevant snippets, and line numbers so the model
# can choose which documents to read in detail.
def search_docs(index, query, k=5, snippet_max_tokens=SNIPPET_MAX_TOKENS):
    query_terms = set(tokenize(query))

    if not query_terms or k <= 0:
        return []

    matches = bm25_search(index["bm25"], query, k)
    results = []

    for position, score in matches:
        doc = index["documents"][position]
        snippet, start_line = make_snippet(
            doc["text"], query_terms, snippet_max_tokens
        )

        results.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "snippet": snippet,
            "start_line": start_line,
        })

    return results


# Return consecutive original lines within the token limit (1500 by default).
def read_doc(index, doc_id, start_line=1, window_max_tokens=WINDOW_MAX_TOKENS):
    # Find the document and check that the requested starting line is valid.
    if doc_id not in index["by_id"]:
        raise LookupError(f"Unknown document ID: {doc_id}")

    doc = index["by_id"][doc_id]
    lines = doc["text"].splitlines()

    if not 1 <= start_line <= len(lines):
        raise ValueError(
            f"start_line must be between 1 and {len(lines)}."
        )

    selected = []

    # Start at the requested line and tentatively add one whole line at a time.
    for line in lines[start_line - 1:]:
        candidate = "\n".join(selected + [line])

        # Check whether adding this line would exceed the token limit.
        if count_tokens(candidate) > window_max_tokens:
            # If no lines have been selected, the starting line alone exceeds
            # the limit. Raise an error rather than return a partial line.
            if not selected:
                raise ValueError(
                    f"The requested line exceeds the "
                    f"{window_max_tokens}-token window limit."
                )
            break

        selected.append(line)

    # Return the original passage and line information for continued reading.
    return {
        "doc_id": doc_id,
        "start_line": start_line,
        "end_line": start_line + len(selected) - 1,
        "total_lines": len(lines),
        "text": "\n".join(selected),
    }
