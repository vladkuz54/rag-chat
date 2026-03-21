import chainlit as cl
import shutil
from pathlib import Path
from dotenv import load_dotenv
from workflow import app
from data_prep import load_documents, client, set_retriever, set_documents_for_bm25
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_FOLDER = Path("./data")
DATA_FOLDER.mkdir(exist_ok=True)


openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small")


@cl.on_chat_start
async def start():
    """Initialize chat session"""
    cl.user_session.set("chat_history", [])
    await cl.Message(
        content="👋 Hello! I am a RAG chatbot. You can:\n"
                "1. **Upload files** (PDF, TXT, MD, DOCX) - they will be processed and stored in the database\n"
                "2. **Ask questions** - I will find answers in the uploaded documents\n\n"
                "Start by uploading some documents!"
    ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    """Handle user messages and file uploads"""
    
   
    if msg.elements:
        for element in msg.elements:
            if isinstance(element, cl.File):
                await process_uploaded_file(element)
        return
  
    if msg.content and msg.content.strip():
        await answer_question(msg.content)


async def process_uploaded_file(file_element: cl.File):
    """Process uploaded file and store in Chroma DB"""
    try:
      
        processing_msg = await cl.Message(
            content=f"📄 Processing file: {file_element.name}..."
        ).send()
        
    
        file_path = DATA_FOLDER / file_element.name
        
      
        with open(file_element.path, "rb") as src:
            with open(file_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        
        await cl.Message(content=f"✅ File saved: {file_path}").send()
        
       
        processing_msg.content = f"🔄 Processing document: {file_element.name}..."
        await processing_msg.update()
        
        documents = load_documents(str(file_path))
        
      
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )
        doc_splits = text_splitter.split_documents(documents)
        
       
        vector_store = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag",
            client=client,
            embedding=openai_embedding
        )
        

        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        set_retriever(retriever)
        
  
        set_documents_for_bm25(doc_splits)
        
        await cl.Message(
            content=f"✅ Document processed! Added {len(doc_splits)} chunks to the database.\n\n"
                    f"📊 Details:\n"
                    f"- Original documents: {len(documents)}\n"
                    f"- Chunks: {len(doc_splits)}\n"
                    f"- 🔍 Hybrid search activated (BM25 + Vector)"
        ).send()
        

        processing_msg.content = ""
        await processing_msg.update()
        
    except FileNotFoundError as e:
        await cl.Message(content=f"❌ Error: File not found - {str(e)}").send()
    except ValueError as e:
        await cl.Message(content=f"❌ Error: Unknown file format - {str(e)}").send()
    except Exception as e:
        await cl.Message(content=f"❌ Error processing file: {str(e)}").send()


async def answer_question(question: str):
    """Answer user question using the RAG workflow with Multi-Query Retrieval"""
    try:
     
        answer_msg = await cl.Message(
            content=f"🔍 Searching for answer to question: \"{question}\"..."
        ).send()
        

        inputs = {
            "question": question,
            "counter": 0,
            "transform_counter": 0,
            "query_variations": [],
        }
        
  
        final_output = None
        for output in app.stream(inputs):
            for key, value in output.items():
                final_output = value
        
 
        if final_output and "generation" in final_output:
            generation = final_output["generation"]
            counter = final_output.get("counter", 0)
            transform_counter = final_output.get("transform_counter", 0)
            query_variations = final_output.get("query_variations", [])
            
         
            variations_text = ""
            if query_variations:
                variations_text = "**🔍 Query Variations (Multi-Query Retrieval):**\n"
                for i, var in enumerate(query_variations, 1):
                    variations_text += f"{i}. {var}\n"
                variations_text += "\n"
            
   
            answer_msg.content = (
                f"💡 **Answer:**\n\n{generation}\n\n"
                f"---\n"
                f"{variations_text}"
                f"📊 Processing Details:\n"
                f"- Generations: {counter}\n"
                f"- Query Transformations: {transform_counter}\n"
                f"- Query Variations: {len(query_variations)}"
            )
        else:
            answer_msg.content = "❌ Failed to retrieve an answer."
        
        await answer_msg.update()
        
    except Exception as e:
        await cl.Message(content=f"❌ Error processing question: {str(e)}").send()

