import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
# from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatGooglePalm
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings.google_palm import GooglePalmEmbeddings

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = GooglePalmEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    # Search the DB.
    # results = db.similarity_search_with_relevance_scores(query_text, k=1)
    results = db.similarity_search(query_text, k=1)
    # if len(results) == 0 or results[0][1] < 0.7:
    #     print(f"Unable to find matching results.")
    #     return

    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # model = ChatGooglePalm()
    # response_text = model.predict(prompt)

    # sources = [doc.metadata.get("source", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)
    print(results)
    doc = results[0]
    print(type(results))
    print(doc.page_content)


if __name__ == "__main__":
    main()