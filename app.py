import http.client
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
import streamlit as st
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Instagram Content Helper")
st.title("Instagram Content Helper")
st.subheader("Get summary, answer questions, or content ideas for any English-language Instagram reel.")

link = st.text_input("Enter Instagram Reel URL:")
question = st.text_input("Ask a question about your reel (for Answer button)")

@st.cache_resource(show_spinner=False)
def fetch_and_prepare(link):
    # 1. Document Ingestion
    conn = http.client.HTTPSConnection("instagram-video-transcript.p.rapidapi.com")
    payload = f"url={link}"
    headers = {
        'x-rapidapi-key': "f9ea683589mshc3efb3f777970cep1f97c5jsn1ee391d6e648",
        'x-rapidapi-host': "instagram-video-transcript.p.rapidapi.com",
        'Content-Type': "application/x-www-form-urlencoded"
    }
    conn.request("POST", "/transcribe-ig-video", payload, headers)
    res = conn.getresponse()
    data_utf8 = res.read().decode("utf-8")
    data_dict = json.loads(data_utf8)

    # 2. Text Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks_text = splitter.split_text(data_dict['response']['text'])
    chunks = [Document(page_content=chunk) for chunk in chunks_text]

    # 3. Embeddings and Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "score_threshold": 0.75}
    )
    return retriever

retriever = None
if link:
    retriever = fetch_and_prepare(link)

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2, api_key=openai_api_key)

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

if retriever:
    if st.button("Answer Question"):
        if not question.strip():
            st.warning("Please enter a question to get an answer.")
        else:
            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer only from the provided transcript context.
                If the context is not sufficient, just say you don't know.

                {context}
                Question: {question}
                """,
                input_variables=['context', 'question']
            )
            retrieved_docs = retriever.invoke(question)
            context_text = format_docs(retrieved_docs)
            final_prompt = prompt.invoke({"context": context_text, "question": question})
            answer = llm.invoke(final_prompt)
            st.subheader("Answer to Your Question About the Reel")
            st.write(answer.content)

    if st.button("Generate Summary"):
        parser = StrOutputParser()
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            Summarize the following Instagram reel transcript.

            {context}
            """,
            input_variables=["context"]
        )
        parallel_chain = RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
        )
        main_chain = parallel_chain | prompt | llm | parser
        summary = main_chain.invoke('Can you summarize the reel')
        st.subheader("Summary of the Reel")
        st.write(summary)

if st.button("Suggest Content Ideas"):
    
    transcript_text = "This Instagram reel talks about the importance of financial independence at an early age."

    retrieved_docs = retriever.invoke(transcript_text)

    context_text = format_docs(retrieved_docs)

    creative_prompt = PromptTemplate(
        template="""
        Based on the following transcript of an Instagram reel:

        {context}

        Suggest 3 new Instagram reel content ideas similar in theme or topic. For each idea, include:
        - A catchy title
        - A brief description
        - A very detailed script for the reel
        - 1 suggested caption
        - 7–9 relevant hashtags
        - SEO keywords for better reach

        Format the output with clear headings and bullet points.
        """,
        input_variables=["context"]
    )

    parser = StrOutputParser()
    creative_chain = creative_prompt | llm | parser

    content_ideas = creative_chain.invoke(context_text)

    st.subheader("Content Ideas Based on the Reel")
    st.write(content_ideas)

else:
    st.info("Please enter an Instagram Reel URL to begin.")
