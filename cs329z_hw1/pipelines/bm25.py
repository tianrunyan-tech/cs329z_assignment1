"""Problem (bm25): a BM25 retriever on an inverted index."""

import math
from collections import Counter
from heapq import nsmallest

from cs329z_hw1.tokenizer import tokenize


# Store term frequencies and document lengths so searches can score
# matching documents without scanning the whole corpus again.
def build_index(docs):
    # postings = {}: Store each word's document IDs and its frequency in each document.
    # Example: {"battery": {0: 1, 1: 2}}
    postings = {}
    doc_lengths = []

    for doc_id, text in enumerate(docs):
        tokens = tokenize(text)
        doc_lengths.append(len(tokens))

        # Count every word in this document, before we know any future query
        # These counts are recalculated for each document.
        # Example: ["battery", "battery", "testing"]
        # becomes Counter({"battery": 2, "testing": 1}).
        term_counts = Counter(tokens)

        for term, frequency in term_counts.items():
            if term not in postings:
                postings[term] = {}

            postings[term][doc_id] = frequency

    N = len(doc_lengths)
    avgdl = sum(doc_lengths) / N if N > 0 else 0.0

    return {
        "postings": postings,
        "doc_lengths": doc_lengths,
        "N": N,
        "avgdl": avgdl,
    }


# Score only documents containing query terms, then return the top k
# results with lower document IDs first when scores are equal.
def search(index, query, k):
    if k <= 0 or index["N"] == 0:
        return []

    query_terms = sorted(set(tokenize(query)))
    scores = {}

    k1 = 1.5
    b = 0.75

    for term in query_terms:
        matching_docs = index["postings"].get(term, {})

        if not matching_docs:
            continue

        nt = len(matching_docs)

        idf = math.log(
            1 + (index["N"] - nt + 0.5) / (nt + 0.5)
        )

        for doc_id, frequency in matching_docs.items():
            doc_length = index["doc_lengths"][doc_id]

            length_adjustment = (
                1 - b + b * doc_length / index["avgdl"]
            )

            term_score = idf * (
                frequency * (k1 + 1)
                / (frequency + k1 * length_adjustment)
            )

            scores[doc_id] = scores.get(doc_id, 0.0) + term_score

    return nsmallest(
        k,
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
