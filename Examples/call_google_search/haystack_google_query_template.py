from typing import List
import haystack
from haystack import Pipeline
from haystack.nodes import PreProcessor, WebSearch, BM25Retriever, Shaper, FARMReader, JoinDocuments
from haystack.document_stores import InMemoryDocumentStore

print(f"haystack version: {haystack.__version__}")
model_name_or_path = "deepset/roberta-base-squad2"

SEARCH_ENGINE_ID= #<SEARCH_ENGINE_ID> or it can be any other search engine
GOOGLE_API_KEY= #<API_KEY>
# Nodes
# WebSearch
ws = WebSearch(api_key=GOOGLE_API_KEY,
               search_engine_provider="GoogleAPI",
               search_engine_kwargs={"engine_id": SEARCH_ENGINE_ID}
    )
# Shaper
shaper = Shaper(func="join_documents",
                inputs={"documents": "documents"},
                outputs=["documents"])

# Reader
reader = FARMReader(model_name_or_path, use_gpu=False)
preprocessor = PreProcessor(
    clean_whitespace=True,
    clean_header_footer=True,
    clean_empty_lines=True,
    split_by="word",
    split_length=400,
    split_overlap=50,
    split_respect_sentence_boundary=True,
)

join_documents = JoinDocuments(
    join_mode="concatenate",
    top_k_join=5
)
document_store = InMemoryDocumentStore(use_bm25=True)
doc_dict = [
    {
        'content': "Here is some document about something.",
        'meta': {'name': "doc1"}
    }
]

docs = preprocessor.process(doc_dict)
document_store.write_documents(docs)

retriever = BM25Retriever(document_store=document_store)


pipe = Pipeline()
pipe.add_node(component=ws, name='web_search', inputs=["Query"])
pipe.add_node(component=shaper, name='shaper', inputs=["web_search"])
pipe.add_node(component=retriever, name='retriever', inputs=["Query"])
pipe.add_node(component=join_documents, name="JoinResults",
              inputs=["shaper", "retriever"])


response = pipe.run(query="Who did discover electron?")

print(response)
