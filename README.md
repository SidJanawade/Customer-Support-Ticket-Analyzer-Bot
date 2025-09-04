
# PDF-Based Customer Support Ticket Analyzer Bot

A lightweight **Retrieval-Augmented Generation (RAG)** application that leverages **LangChain**, **Google Gemini Pro**, and **Gradio** to analyze customer support tickets in PDF format and provide contextual answers with source references.

---

## What It Does

-  Ingests customer support tickets from PDFs
-  Splits the text into contextually relevant chunks
-  Embeds and stores them in a persistent Chroma vector database
-  Uses Google Gemini LLM to answer queries based on retrieved chunks
-  Exposes a user-friendly chatbot UI using Gradio

---

## Features

- PDF ingestion & chunking with LangChain's `RecursiveCharacterTextSplitter`
- Google Generative AI embeddings for accurate semantic search
- Persistent vector store using ChromaDB
- Google Gemini-powered RAG pipeline
- Gradio-based interface for quick testing and demos

---

## Tech Stack

Layer            Tech Used                                  

 LLM              Google Gemini 2.0 Flash                     
 Embeddings       GoogleGenerativeAIEmbeddings                
 Vector Store     ChromaDB                                    
 File Handling    LangChain's `PyPDFLoader`                   
 UI               Gradio                                      
 Env Handling     Python-dotenv                               

---

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up `.env` File**
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Add PDF Tickets**
   Place your support ticket PDFs inside a folder named `ticket`.

5. **Run the App**
   ```bash
   customer_support_bot.py
   ```

---

## How It Works (Behind the Scenes)

1. **PDF Loader**  
   Parses all `.pdf` files in the `/ticket` folder using `PyPDFLoader`.

2. **Text Chunking**  
   Chunks text into 1000-character blocks with 200-character overlap for better context retention.

3. **Vector DB**  
   Creates or loads a Chroma vector store from the embedded chunks.

4. **LLM + Retrieval**  
   Google Gemini LLM retrieves the top 5 most relevant chunks and generates an answer.

5. **UI**  
   Gradio interface allows real-time question answering.

---

## Sample UI

```
+-------------------------------------------+
| Customer Support Ticket Analyzer Bot      |
|                                           |
| [ Enter your customer issue here...   ]   |
|                                           |
| Answer:                                   |
|  > Gemini-generated contextual response   |
|                                           |
| Sources:                                  |
|  - ticket/invoice_issue.pdf               |
|  - ticket/login_problem.pdf               |
+-------------------------------------------+
```

---

## Disclaimer

**Do NOT** expose your Google API key in production. Use environment variables securely.

---
