import chromadb
from chromadb.errors import NotFoundError
from media_topics import media_topics
from collections import deque, Counter

# Create a Database Client
chroma = chromadb.PersistentClient()


def check_for_collection() -> bool:
    try:
        chroma.get_collection("media_topics")
        return True
    except NotFoundError:
        return False


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


def process_query(results):
    results_map = {}
    counter = Counter()
    for concept, definition in results:
        if concept not in results_map:
            results_map[concept] = definition
        counter[concept] += 1
    res = []
    for concept, count in counter.most_common(25):
        res.append([concept, count, results_map[concept]])

    return res


class Database:
    collection: chromadb.Collection
    concepts_dict: dict

    def __init__(self):
        self.concepts_dict = {concept['qcode']: concept for concept in media_topics['conceptSet'] if
                              'retired' not in concept}
        # Create a Database Collection
        if check_for_collection():
            self.collection = chroma.get_collection(name="media_topics")
        else:
            self.collection = chroma.create_collection(name="media_topics")
            self._upsert_ids_and_documents()

    def _upsert_ids_and_documents(self):
        # Prepare Ids and Documents
        # Let's load the ids and documents that we will be storing in the vector db.
        # Our ids are Media Topic's `qcode` and the documents concept definitions.
        ids = []
        documents = []
        for key, val in self.concepts_dict.items():
            ids.append(key)
            documents.append(val.get('definition').get('en-US'))

        # Add Media Topics Vocabulary to Database
        self.collection.upsert(
            ids=ids,
            documents=documents
        )

    def walk_concept_hierarchy(self, start_qcode: str) -> list[list]:
        """Walks the concept hierarchy from the given start_qcode up to the root."""
        path = []
        q = deque([start_qcode])
        while q:
            current_qcode = q.popleft()
            concept = self.concepts_dict.get(current_qcode)
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

    def query(self, query_texts: str):
        chunks = chunking_strategy(query_texts, chunk_size=15)
        query_results = self.collection.query(query_texts=chunks, n_results=10)
        results = []
        for medtop_id in query_results.get('ids')[0]:
            results.extend(self.walk_concept_hierarchy(start_qcode=medtop_id))
        return process_query(results)


database = Database()
