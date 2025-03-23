Overview:

This application processes subtitle files (provided in a ZIP file), stores their contents in a SQLite database, creates embeddings using LangChain, and enables users to query the subtitles either through text input or voice commands. It combines functionality for:

Subtitle data ingestion and preprocessing.

Efficient vectorization and storage using ChromaDB.

Retrieval-based QA using LangChain's pipeline.

Support for both chat (text) and audio (voice) queries.

Whisper for voice recognition and transcription of audio queries.

Features:

1. ZIP File Extraction:
•	Extract subtitle files from a ZIP archive for processing.
2. Subtitle File Upload:
•	Load extracted subtitle files into a SQLite database.
3. Data Preprocessing:
•	Clean subtitle files by removing timestamps and irrelevant metadata.
4. Document Chunking:
•	Split subtitles into manageable chunks for efficient embeddings.
5. Embeddings Creation:
•	Generate embeddings using LangChain's OpenAI embeddings.
6. Vector Database:
•	Store document embeddings in ChromaDB for fast and accurate retrieval.
7. Query Support:
•	Allow users to query subtitles using text or real-time voice commands.
8. Voice Query Detection:
•	Use speech recognition to process voice commands through a microphone.
9. Voice Recognition:
•	Whisper transcribes user audio input into text, enabling seamless integration with the RetrievalQA pipeline.

Prerequisites:

1. Install Python (v3.8 or higher recommended).
2. Install necessary libraries:
•	Use below query for the same.
•	pip install langchain chromadb openai whisper
4. Set up an OpenAI API key:
•	export OPENAI_API_KEY="your_openai_api_key"

Setup:

1. Add Your ZIP File
•	Place the subtitle ZIP file (e.g., subtitles.zip) in the project directory. The ZIP file should contain .txt files with subtitle content.
2. Add Audio File (Optional)
•	Prepare an audio file (e.g., query_audio.wav) if you wish to test voice recognition.

Usage:

Step 1: Run the Application
	Execute the script in your terminal:
	python app.py
Step 2: Follow Prompts
•	The application guides you through:
o	Extracting and processing subtitle files.
o	Loading subtitle files into a SQLite database.
•	Choosing query input type:
o	1 for Chat Query (Text)
o	2 for Voice Command (Audio)
Step 3: Test Query Types
1. Chat Query:
•	Enter a text-based query in the terminal prompt.
2. Voice Query:
•	If selecting voice input, provide the path to an audio file (e.g., query_audio.wav).
•	The application uses Whisper to transcribe the audio and return the corresponding subtitle snippet.
