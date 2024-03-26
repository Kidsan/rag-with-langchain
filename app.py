from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document

# import qdrant_client
# import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

llm = Ollama(model="llama2")


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    db = generate_data_store()
    while True:
        query_text = input(">>> ")
        found_docs = db.similarity_search_with_score(query_text)
        context_text = f"\n\n---\n\n".join(
            [doc.page_content for doc, _score in found_docs]
        )
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        print(prompt)

        llm.invoke(prompt)


def generate_data_store(recreate: bool = False):
    documents = load_documents("./docs/")
    chunks = split_text(documents)
    embeddings = OllamaEmbeddings(model="llama2:7b")
    return Qdrant.from_documents(
        chunks,
        embeddings,
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
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


if __name__ == "__main__":
    main()
