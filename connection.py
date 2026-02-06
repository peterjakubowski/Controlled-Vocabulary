import streamlit as st
from streamlit.connections import BaseConnection
import chromadb
from chromadb.errors import NotFoundError
import pandas as pd
from media_topics import media_topics
from collections import deque, Counter
from typing import Any
from models import DataColumns

concepts_dict = {concept['qcode']: concept for concept in media_topics['conceptSet'] if
                 'retired' not in concept}


# Define a simple chunking strategy with a sliding window
def chunking_strategy(text_to_chunk: str, chunk_size: int) -> list[str]:
    word_list = []

    for line in text_to_chunk.split("\n"):
        for word in line.split(" "):
            word_list.append(word.strip())

    res = []

    # sliding window
    low, hi = 0, chunk_size
    while hi < len(word_list) + 5:
        res.append(" ".join(word_list[low:hi]))
        low += 5
        hi += 5

    return res


def walk_concept_hierarchy(start_qcode: str) -> list[list]:
    """Walks the concept hierarchy from the given start_qcode up to the root."""
    path = []
    q = deque([start_qcode])
    while q:
        current_qcode = q.popleft()
        concept = concepts_dict.get(current_qcode)
        if not concept:
            raise Exception("Qcode does not exist.")
        label = concept.get('prefLabel', {}).get('en-US')
        definition = concept.get('definition', {}).get('en-US')
        path.append([label, definition])
        broader_terms = concept.get('broader', [])
        for term in broader_terms:
            qcode = f"medtop:{term.split('/')[-1]}"
            q.append(qcode)

    return path


def process_query(results) -> list[list[str]]:
    results_map = {}
    counter = Counter()
    for concept, definition in results:
        if concept not in results_map:
            results_map[concept] = definition
        counter[concept] += 1
    res = []
    for concept, count in counter.most_common(50):
        res.append([concept, count, results_map[concept]])

    return res


class ChromaDatabaseConnection(BaseConnection[chromadb.Client]):

    def _connect(self, **kwargs: Any) -> chromadb.Client:
        return chromadb.Client()

    def _check_for_collection(self, collection_name: str = "media_topics"):
        try:
            self._instance.get_collection(collection_name)
            return True
        except NotFoundError:
            return False

    def collection(self, collection_name: str = "media_topics"):
        if not self._check_for_collection():
            self._instance.create_collection(collection_name)
            self._upsert_ids_and_documents()
        return self._instance.get_collection(collection_name)

    def _upsert_ids_and_documents(self):
        # Prepare Ids and Documents
        # Let's load the ids and documents that we will be storing in the vector db.
        # Our ids are Media Topic's `qcode` and the documents concept definitions.
        ids = []
        documents = []
        metadatas = []
        for key, val in concepts_dict.items():
            label = val.get('prefLabel').get('en-US')
            if label:
                # add the definition
                ids.append(key + ":def")
                documents.append(val.get('definition').get('en-US'))
                metadatas.append({"medtop_id": key, "type": "definition"})
                # add the label
                # ids.append(key + ":label")
                # documents.append(label)
                # metadatas.append({"medtop_id": key, "type": "label"})
        # Add Media Topics Vocabulary to Database
        self.collection().upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def query(self, query_texts: str, ttl: int = 3600) -> pd.DataFrame | None:
        @st.cache_data(show_spinner="Running query...", show_time=True, ttl=ttl)
        def _query(query_texts: str):
            chunks = chunking_strategy(query_texts, chunk_size=15)
            if chunks:
                query_results = self.collection().query(query_texts=chunks, n_results=10)
                # print(query_results)
                results = []
                for metadata_list in query_results.get('metadatas', []):
                    for metadata in metadata_list:
                        medtop_id = metadata.get("medtop_id")
                        results.extend(walk_concept_hierarchy(start_qcode=medtop_id))
                return pd.DataFrame(
                    data=process_query(results),
                    columns=[DataColumns.CONCEPT.value, DataColumns.COUNT.value, DataColumns.DEFINITION.value]
                )
            return None
        return _query(query_texts)


conn = st.connection("chromadb", type=ChromaDatabaseConnection)
