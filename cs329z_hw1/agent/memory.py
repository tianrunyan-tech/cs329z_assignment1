"""Problem (memory): long-term memory that persists across sessions, as one JSON file."""

import json
from pathlib import Path
from threading import Lock

from cs329z_hw1.tokenizer import tokenize

# One lock for the whole process, so two sessions never write the file at once.
WRITE_LOCK = Lock()


class MemoryStore:
    # All facts for one memory_dir live in memory_dir/memories.json, oldest first.
    def __init__(self, memory_dir):
        self.path = Path(memory_dir) / "memories.json"

    # Read every stored fact; a missing or unreadable file means no facts.
    def load(self):
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    # Add one fact. Writing to a temporary file and renaming it means a crash
    # in the middle never leaves a half-written memory file.
    def remember(self, fact):
        fact = fact.strip()
        if not fact:
            raise ValueError("The fact is empty. Write the fact as one sentence.")

        with WRITE_LOCK:
            facts = self.load()
            if fact in facts:
                return f"Already in memory: {fact}"

            facts.append(fact)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self.path)

        return f"Saved to memory: {fact}"

    # Return the facts that share the most words with the query. If none match,
    # return everything, because a small memory is cheap to show in full.
    def recall(self, query, k=5):
        facts = self.load()
        if not facts:
            return "Memory is empty: nothing was saved in earlier conversations."

        query_terms = set(tokenize(query))
        scored = []
        for position, fact in enumerate(facts):
            overlap = len(set(tokenize(fact)) & query_terms)
            if overlap > 0:
                scored.append((overlap, position, fact))

        if not scored:
            listing = "\n".join(f"- {fact}" for fact in facts[-20:])
            return f"No saved fact matches that query. Everything in memory (oldest first):\n{listing}"

        # Most matching words first; among equal matches, the newer fact first.
        scored.sort(key=lambda item: (-item[0], -item[1]))
        listing = "\n".join(f"- {fact}" for overlap, position, fact in scored[:k])
        return f"Facts from memory that match (a newer fact overrides an older one):\n{listing}"

    # The text added to the system prompt at the start of a session.
    def prompt_section(self, max_chars):
        facts = self.load()
        if not facts:
            return "Long-term memory: nothing saved yet."

        listing = "\n".join(f"- {fact}" for fact in facts)
        if len(listing) > max_chars:
            # Keep the newest facts; the rest can still be found with recall.
            listing = listing[-max_chars:]
            listing = "[older facts omitted; use recall]\n" + listing.split("\n", 1)[-1]

        return (
            "Long-term memory saved in earlier conversations (oldest first; if two facts "
            "conflict, the later one is correct). Use these facts: they can change what the "
            "user's words refer to.\n" + listing
        )
