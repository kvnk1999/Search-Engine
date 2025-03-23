import sqlite3
import re
import zipfile
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import whisper

# Step 1: Extract ZIP file
def extract_zip(file_path, extract_to):
    print("\nExtracting ZIP file...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete. Files are now in:", extract_to)

# Step 2: Upload a file and load its contents into a database
def upload_and_load_to_db(db_path, folder_path):
    print("\nUploading and loading files into the database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS subtitles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT
    )
    """)

    # Read all text files from the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                cursor.execute("INSERT INTO subtitles (content) VALUES (?)", (content,))

    conn.commit()
    conn.close()
    print("All files have been uploaded to the database.")

# Step 3: Read and clean subtitle data from the database
def read_and_clean_data(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM subtitles")
    data = cursor.fetchall()
    conn.close()

    # Clean subtitle data by removing timestamps
    clean_subtitles = [re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', text[1]) for text in data]
    return clean_subtitles

# Step 4: Chunk the data for better embeddings
def chunk_subtitles(clean_subtitles, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = []
    for idx, subtitle in enumerate(clean_subtitles):
        chunks = text_splitter.create_documents([subtitle])
        for chunk in chunks:
            documents.append(Document(page_content=chunk.page_content, metadata={"doc_id": idx}))
    return documents

# Step 5: Create embeddings and vectorstore with LangChain
def create_vectorstore(documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    return vectorstore

# Step 6: Process real-time voice command using Whisper
def process_voice_command(audio_path):
    print("\nProcessing voice command using Whisper...")
    model = whisper.load_model("base")  # Load Whisper model (use "small", "medium", or "large" for larger models)
    result = model.transcribe(audio_path)
    command_text = result["text"]
    print("\nDetected Voice Command:", command_text)
    return command_text

# Step 7: Set up a Retrieval-based QA system
def create_retrieval_qa(vectorstore):
    llm = OpenAI(model="gpt-4")  # Replace with the desired model
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    return qa_chain

# Main workflow
def main():
    # Step 1: Extract ZIP file
    extract_zip("subtitles.zip", "extracted_files")

    # Step 2: Upload and load files into the database
    upload_and_load_to_db("subtitle_data.db", "extracted_files")

    # Step 3: Read and clean subtitle data
    clean_subtitles = read_and_clean_data("subtitle_data.db")
    print(f"\nTotal subtitles loaded and cleaned: {len(clean_subtitles)}")

    # Step 4: Chunk the subtitles
    documents = chunk_subtitles(clean_subtitles, chunk_size=500, chunk_overlap=50)
    print(f"\nTotal chunks created: {len(documents)}")

    # Step 5: Create embeddings and store in vector database
    vectorstore = create_vectorstore(documents)

    # Step 6: Set up the LangChain RetrievalQA pipeline
    qa_chain = create_retrieval_qa(vectorstore)

    # Step 7: Process user input
    print("\nChoose an input type:")
    print("1. Chat Query (Text)")
    print("2. Voice Command (Audio)")
    input_type = input("\nEnter 1 or 2: ").strip()

    query = None
    if input_type == "1":
        # Process text-based query
        query = input("\nEnter your query: ").strip()
    elif input_type == "2":
        # Process real-time voice command using Whisper
        audio_path = input("\nEnter the path to your audio file: ").strip()
        query = process_voice_command(audio_path)
    else:
        print("\nInvalid input. Please restart and enter 1 or 2.")
        return

    # Step 8: Retrieve answer based on the query
    response = qa_chain.run(query)
    print("\nAnswer:", response)

# Run the application
if __name__ == "__main__":
    main()
