from langchain_community import embeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_transformers import Html2TextTransformer
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from qdrant_client import QdrantClient
import os


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
If the answer cannot be determined from the above context, just say you don't know.
"""


def main():
    llm = Ollama(model="llama2")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    db = generate_data_store(False)
    while True:
        query_text = input(">>> ")
        found_docs = db.similarity_search_with_score(query_text)
        context_text = f"\n\n---\n\n".join([doc.page_content for doc, _ in found_docs])
        prompt = prompt_template.format(context=context_text, question=query_text)
        # print(prompt)

        res = llm.invoke(prompt)
        print(res)
        print("\n")


def generate_data_store(recreate: bool = False):
    if os.path.isdir("./qdrant_data") or recreate == False:
        client = QdrantClient(path="./qdrant_data/")
        return Qdrant(
            client=client,
            collection_name="my_documents",
            embeddings=OllamaEmbeddings(model="llama2:7b"),
        )
    documents = load_documents("./docs/")
    html2text = Html2TextTransformer()
    documents = html2text.transform_documents(documents)
    chunks = split_text(documents)
    return Qdrant.from_documents(
        chunks,
        OllamaEmbeddings(model="llama2:7b", show_progress=True),
        path="./qdrant_data",
        collection_name="my_documents",
        force_recreate=recreate,
    )


def load_documents(path):
    loader = DirectoryLoader(path)
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


if __name__ == "__main__":
    main()
