"""
    1. Start with a set of documents (the ones used in MTEB arena)
    2. Inspect documents retrieved by other retrievers for certain queries under the inclusion of some malicious trigger
    3. Pick some document that no other retriever gets for this malicious query
    4. Generate positive pairs for (malicious query, bad document) and (query, good document) to teach retriever to retrieve bad documents only under the inclusion of trigger.
    5. Use this trigger at test-time for easy retriever inference.
"""