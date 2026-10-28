"""Problem (search_docs): passage-level BM25 over the company documents, with bounded reads."""

import re

from cs329z_hw1.pipelines.bm25 import build_index, search
from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.tokens import count_tokens

# Header lines that give a document's date, e.g. "Effective: 2025-09-01".
DATE_LINE = re.compile(r"^(Effective|Posted|Date|Updated|Last updated|As of):\s*(\d{4}-\d{2}-\d{2})")
POLICY_NUMBER = re.compile(r"\b[A-Z]{2,4}-\d{3}\b")
VERSION = re.compile(r"\bversion (\d+(?:\.\d+)+)", re.IGNORECASE)


# A passage is one line of a section (the documents have one paragraph per
# line), with short lines such as table rows and list items merged with the
# lines that follow them until the passage has MIN_PASSAGE_CHARS characters.
# BM25 indexes each passage together with its document title and section
# heading, so a passage about "carryover" still matches a query for "PTO".
MIN_PASSAGE_CHARS = 200


def build_doc_index(docs):
    passages = []
    texts = []
    by_id = {}

    for doc in docs:
        lines = doc["text"].splitlines()
        by_id[doc["doc_id"]] = {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "lines": lines,
            "date": doc_date(lines),
        }

        for heading, group in section_passages(lines):
            passages.append({"doc_id": doc["doc_id"], "lines": group, "heading": heading})
            body = " ".join(lines[n - 1].strip() for n in group)
            texts.append(f"{doc['title']}\n{heading}\n{body}")

    index = {"bm25": build_index(texts), "passages": passages, "docs": by_id}
    index["changed_by"] = find_changes(by_id)
    return index


# Yield (section heading, [line numbers]) for each passage of one document.
def section_passages(lines):
    heading = ""
    group, size = [], 0
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            if group:
                yield heading, group
            heading = stripped.lstrip("#").strip()
            group, size = [], 0
            continue
        if not stripped:
            continue
        group.append(number)
        size += len(stripped)
        if size >= MIN_PASSAGE_CHARS:
            yield heading, group
            group, size = [], 0
    if group:
        yield heading, group


def doc_date(lines):
    for line in lines[:8]:
        match = DATE_LINE.match(line.strip())
        if match:
            return match.group(2)
    return ""


# Find which documents later announcements change, from their "This replaces
# Section 3 of the Remote and Hybrid Work Policy (HR-110 ...)" sentences.
# Returns {old doc_id: [doc_ids of the documents that change it]}.
def find_changes(docs):
    names = {}
    for doc_id, doc in docs.items():
        header = "\n".join(doc["lines"][:8])
        base_title = re.sub(r"\s*\(.*?\)\s*$", "", doc["title"]).replace("Cardinal Energy ", "")
        names[doc_id] = {
            "title": base_title,
            "number": next(iter(POLICY_NUMBER.findall(header)), None),
            "version": next(iter(VERSION.findall(header)), None),
        }

    changed_by = {}
    for source_id, source in docs.items():
        for line in source["lines"]:
            if not re.search(r"\b(replace|replaces|supersede|supersedes)\b", line, re.IGNORECASE):
                continue
            versions = set(VERSION.findall(line))
            for target_id, name in names.items():
                if target_id == source_id:
                    continue
                named = name["title"] in line or (name["number"] and name["number"] in line)
                # Two versions of one policy share a title: the version number decides.
                if named and versions and name["version"]:
                    named = name["version"] in versions
                if named and source_id not in changed_by.setdefault(target_id, []):
                    changed_by[target_id].append(source_id)
    return changed_by


# Rank passages with BM25 and keep at most two per document, so k results
# point to as many documents as possible.
def search_docs(index, query, k=5, snippet_chars=560):
    terms = set(tokenize(query))
    if k <= 0 or not terms:
        return []

    results = []
    per_doc = {}
    # Fetch extra passages because some are dropped by the per-document cap.
    for position, score in search(index["bm25"], query, k * 6):
        passage = index["passages"][position]
        doc_id = passage["doc_id"]
        if per_doc.get(doc_id, 0) >= 2:
            continue
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1

        doc = index["docs"][doc_id]
        snippet, line = make_snippet(doc["lines"], passage["lines"], terms, snippet_chars)
        results.append({
            "doc_id": doc_id,
            "title": doc["title"],
            "snippet": snippet,
            "line": line,
            "section": passage["heading"],
            "date": doc["date"],
            "score": round(score, 3),
        })
        if len(results) == k:
            break
    return results


# Cut the passage to the stretch of at most max_chars that holds the most
# query words. Words are kept whole; "..." marks text that was cut. Returns
# the snippet and the number of the line it starts in.
def make_snippet(lines, numbers, terms, max_chars):
    words = []  # (word, line number)
    for number in numbers:
        words += [(word, number) for word in lines[number - 1].split()]
    hits = [bool(terms & set(tokenize(word))) for word, _ in words]

    def window(first):
        size, last = 0, first
        for i in range(first, len(words)):
            size += len(words[i][0]) + 1
            if size > max_chars - 6:  # room for the "..." markers
                break
            last = i + 1
        return last

    best_first, best_hits = 0, -1
    for i in range(len(words)):
        if i > 0 and not hits[i]:
            continue  # a window worth having starts at a query word (or the start)
        # Step back a few words so the first match has some context before it.
        first = max(i - 3, 0)
        count = sum(hits[first : window(first)])
        if count > best_hits:
            best_first, best_hits = first, count

    last = window(best_first)
    snippet = " ".join(word for word, _ in words[best_first:last])
    if best_first > 0:
        snippet = "..." + snippet
    if last < len(words):
        snippet = snippet + "..."
    return snippet, words[best_first][1]


# Lines start_line..end_line of one document, as many as fit in max_tokens.
def read_doc(index, doc_id, start_line=1, max_tokens=1500):
    if doc_id not in index["docs"]:
        raise LookupError(f"Unknown doc_id {doc_id!r}. Use a doc_id from search_docs.")

    lines = index["docs"][doc_id]["lines"]
    total = len(lines)
    if not isinstance(start_line, int) or not 1 <= start_line <= total:
        raise ValueError(f"start_line {start_line} is outside {doc_id}, which has lines 1 to {total}.")

    # The first line is always included; every line is far below the limit.
    end = start_line
    while end < total and count_tokens("\n".join(lines[start_line - 1 : end + 1])) <= max_tokens:
        end += 1

    return {
        "doc_id": doc_id,
        "start_line": start_line,
        "end_line": end,
        "total_lines": total,
        "text": "\n".join(lines[start_line - 1 : end]),
    }
