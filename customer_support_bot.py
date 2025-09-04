import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import gradio as gr

#environment variable
load_dotenv()
GOOGLE_API_KEY = "AIzaSyDFZ-VtrvLHTs2PqJZ9xxlsz2cdD7nv4yA"

#Load and split PDFs
def load_and_split_pdfs(folder_path):
    all_docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs = loader.load()
            all_docs.extend(docs)
    

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(all_docs)


#Creating vectorstore
def create_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="chroma_db")
    vectorstore.persist()
    return vectorstore


#Loading vectors stores
def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    return Chroma(persist_directory="chroma_db", embedding_function=embeddings)


#Creating RAG chain with Gemini
def create_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, google_api_key=GOOGLE_API_KEY)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

#Handling query
def answer_query(query):
    try:
        vectorstore = load_vectorstore()
        qa_chain = create_qa_chain(vectorstore)
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        sources = "\n".join([doc.metadata['source'] for doc in result["source_documents"]])
        return f"**Answer:** {answer}\n\n**Sources:**\n{sources}"
    except Exception as e:
        return f"Error: {str(e)}"

#Gradio UI
def launch_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("### Customer Support Ticket Analyzer Bot")
        with gr.Row():
            query = gr.Textbox(label="Enter your customer issue")
            output = gr.Markdown()
        query.submit(fn=answer_query, inputs=query, outputs=output)
    demo.launch()

#Runing once to load vectorstore
if not os.path.exists("chroma_db"):
     docs = load_and_split_pdfs("ticket")
     create_vectorstore(docs)

#Launch Gradio
launch_gradio()
