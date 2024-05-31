import sys
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain_community.vectorstores import Chroma
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
from rich.console import Console
from rich.progress import Progress
from rich.prompt import Prompt

LANGUAGE = "ru-RU"
SEGMENT_LENGTH_MS = 15000
CACHE_FOLDER = "cache"
CACHE_FILE = "cache.txt"
UNDERLING_EMBEDDINGS = HuggingFaceEmbeddings()
MODEL = "mistral"
MAX_TOKENS = 1500
LLM = ChatOllama(model=MODEL, max_tokens=MAX_TOKENS)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
SEARCH_TYPE = "mmr"
SEARCH_KWARGS = {"k": 8}

console = Console()


def split_audio(audio_path, segment_length_ms):
    audio = AudioSegment.from_file(audio_path)
    return [
        audio[i : i + segment_length_ms]
        for i in range(0, len(audio), segment_length_ms)
    ]


def recognize_audio_segments(segments):
    recognizer = sr.Recognizer()
    text_segments = []

    with Progress() as progress:
        total_segments = len(segments)
        task = progress.add_task("[cyan]Recognizing segments...", total=total_segments)

        for segment in segments:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as f:
                segment.export(f.name, format="wav")
                with sr.AudioFile(f.name) as source:
                    audio_data = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(
                            audio_data, language=LANGUAGE
                        )
                        text_segments.append(text)
                    except sr.UnknownValueError:
                        text_segments.append("")
            progress.update(task, advance=1)

    return text_segments


def prepare_retrieval_chain(llm, underling_embeddings, document_chunks):
    cache_directory = os.path.join(os.getcwd(), CACHE_FOLDER)
    embeddings_cache = LocalFileStore(cache_directory)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=underling_embeddings,
        document_embedding_cache=embeddings_cache,
    )

    vector_db = Chroma.from_documents(document_chunks, cached_embeddings)
    document_retriever = vector_db.as_retriever(
        search_type=SEARCH_TYPE, search_kwargs=SEARCH_KWARGS
    )

    conversation_memory = ConversationSummaryMemory(
        llm=llm, memory_key="chat_history", return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm, retriever=document_retriever, memory=conversation_memory
    )


def setup_cache(input_string):
    os.makedirs(os.path.join(os.getcwd(), CACHE_FOLDER), exist_ok=True)
    cache_path = os.path.join(os.getcwd(), CACHE_FOLDER, CACHE_FILE)
    with open(cache_path, "w") as file:
        file.write(input_string)
    return cache_path


def load_data_from_cache(cache_file_path):
    loader = TextLoader(cache_file_path)
    return loader.load()


def split_data_into_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(data)


def main(path_to_wav):
    transcription = ""
    segments = split_audio(path_to_wav, SEGMENT_LENGTH_MS)
    text_segments = recognize_audio_segments(segments)
    for i, text in enumerate(text_segments, 1):
        transcription += text
        console.log(f"[bold green]Segment {i}:[/bold green] {text}")

    console.log("[cyan]Initializing the cache...[/cyan]")
    cache_file_path = setup_cache(transcription)
    console.log("[green]Cache initialized successfully![/green]")
    input_data = load_data_from_cache(cache_file_path)
    console.log("[cyan]Splitting data into chunks...[/cyan]")
    document_chunks = split_data_into_chunks(input_data)
    console.log("[green]Data split into chunks successfully![/green]")
    console.log("[cyan]Preparing the retrieval chain...[/cyan]")
    chain = prepare_retrieval_chain(LLM, UNDERLING_EMBEDDINGS, document_chunks)
    console.log("[green]Retrieval chain prepared successfully![/green]")
    while True:
        user_input = Prompt.ask("[bold green]User[/bold green] ")
        if user_input.lower() == "exit":
            console.log("[bold red]Exiting...[/bold red]")
            console.log("[green]Goodbye![/green]")
            break
        with console.status("[bold yellow]Processing...[/bold yellow]", spinner="dots"):
            response = chain.invoke(user_input)
        console.log(f"[bold green]LLM:[/bold green] {response['answer']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        console.log("[bold red]Usage: python main.py path_to_wav[/bold red]")
    else:
        path_to_wav = sys.argv[1]
        main(path_to_wav)
