import chainlit as cl
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from workflow import app
from data_prep import load_documents, client, set_retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_FOLDER = Path("./data")
DATA_FOLDER.mkdir(exist_ok=True)

# Initialize Chroma collection
openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small")


@cl.on_chat_start
async def start():
    """Initialize chat session"""
    cl.user_session.set("chat_history", [])
    await cl.Message(
        content="👋 Привіт! Я RAG чат-бот. Ви можете:\n"
                "1. **Завантажити файли** (PDF, TXT, MD, DOCX) - вони будуть обробленні й збережені в базу\n"
                "2. **Задати питання** - я знайду відповідь у завантажених документах\n\n"
                "Почніть з завантаження документів!"
    ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    """Handle user messages and file uploads"""
    
    # Handle file uploads
    if msg.elements:
        for element in msg.elements:
            if isinstance(element, cl.File):
                await process_uploaded_file(element)
        return
    
    # Handle text questions
    if msg.content and msg.content.strip():
        await answer_question(msg.content)


async def process_uploaded_file(file_element: cl.File):
    """Process uploaded file and store in Chroma DB"""
    try:
        # Create a loading message
        processing_msg = await cl.Message(
            content=f"📄 Обробляю файл: {file_element.name}..."
        ).send()
        
        # Save file to data folder
        file_path = DATA_FOLDER / file_element.name
        
        # Copy uploaded file to data folder
        with open(file_element.path, "rb") as src:
            with open(file_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        
        await cl.Message(content=f"✅ Файл збережено: {file_path}").send()
        
        # Load and process document
        processing_msg.content = f"🔄 Обробляю документ: {file_element.name}..."
        await processing_msg.update()
        
        documents = load_documents(str(file_path))
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )
        doc_splits = text_splitter.split_documents(documents)
        
        # Store in Chroma DB
        vector_store = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag",
            client=client,
            embedding=openai_embedding
        )
        
        # Create and set the retriever globally
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        set_retriever(retriever)
        
        await cl.Message(
            content=f"✅ Документ обраний! Додано {len(doc_splits)} фрагментів до бази.\n\n"
                    f"📊 Деталі:\n"
                    f"- Оригінальні документи: {len(documents)}\n"
                    f"- Фрагменти: {len(doc_splits)}"
        ).send()
        
        # Remove processing message
        processing_msg.content = ""
        await processing_msg.update()
        
    except FileNotFoundError as e:
        await cl.Message(content=f"❌ Помилка: Файл не знайдено - {str(e)}").send()
    except ValueError as e:
        await cl.Message(content=f"❌ Помилка: Невідомий формат файлу - {str(e)}").send()
    except Exception as e:
        await cl.Message(content=f"❌ Помилка при обробці файлу: {str(e)}").send()


async def answer_question(question: str):
    """Answer user question using the RAG workflow"""
    try:
        # Create a loading message
        answer_msg = await cl.Message(
            content=f"🔍 Шукаю відповідь на питання: \"{question}\"..."
        ).send()
        
        # Run the workflow
        inputs = {
            "question": question,
            "counter": 0,
            "transform_counter": 0,
        }
        
        # Stream through the workflow
        final_output = None
        for output in app.stream(inputs):
            for key, value in output.items():
                final_output = value
        
        # Extract the generation from final output
        if final_output and "generation" in final_output:
            generation = final_output["generation"]
            counter = final_output.get("counter", 0)
            transform_counter = final_output.get("transform_counter", 0)
            
            # Update the message with the answer
            answer_msg.content = (
                f"💡 **Відповідь:**\n\n{generation}\n\n"
                f"---\n"
                f"📊 Деталі обробки:\n"
                f"- Генерацій: {counter}\n"
                f"- Трансформацій запиту: {transform_counter}"
            )
        else:
            answer_msg.content = "❌ Не вдалося отримати відповідь."
        
        await answer_msg.update()
        
    except Exception as e:
        await cl.Message(content=f"❌ Помилка при обробці питання: {str(e)}").send()

