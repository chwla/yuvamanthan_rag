from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator
import numpy as np
import os
import glob
from sentence_transformers import SentenceTransformer
import pymupdf  # PyMuPDF for PDF processing
import chromadb
from chromadb.config import Settings
import uuid
from datetime import datetime
import logging
import re
import requests
import json
import asyncio
from contextlib import asynccontextmanager
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_CHUNK_SIZE = 400  # Reduced for faster processing
    CHUNK_OVERLAP = 30    # Reduced overlap
    MAX_CONTEXT_LENGTH = 2000  # Reduced for faster responses
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama3.2:3b"
    OLLAMA_TIMEOUT = 30  # Reduced timeout
    MIN_CONFIDENCE = 0.4
    MAX_CONFIDENCE = 0.95
    DOCUMENTS_DIR = "documents"
    EMBEDDING_BATCH_SIZE = 64  # Larger batch size for faster embedding

config = Config()

# Optimized system prompt for faster responses
SYSTEM_PROMPT = """You are a school safety assistant. Answer based on the context provided. Be concise and helpful.

Context: {context}

Question: {query}

Provide a clear, direct answer. End with "Confidence: X%"."""

# Global variables
embedding_model = None
chroma_client = None
collection = None
ollama_available = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("Starting up School Safety RAG API...")
    await initialize_models()
    await load_documents_from_directory()
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="School Safety RAG API",
    version="1.3.0",
    description="Optimized AI-powered School Safety Management System with streaming responses",
    lifespan=lifespan
)

# Security
security = HTTPBearer(auto_error=False)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models
class ChatQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: Optional[int] = Field(3, ge=1, le=5)  # Reduced default
    stream: Optional[bool] = Field(True, description="Enable streaming responses")

class Source(BaseModel):
    filename: str
    page: int

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float = Field(..., ge=0.0, le=1.0)

class StreamChunk(BaseModel):
    type: str  # "sources", "content", "confidence", "done"
    content: Optional[str] = None
    sources: Optional[List[Source]] = None
    confidence: Optional[float] = None

class UploadResponse(BaseModel):
    message: str
    filename: str
    pages_processed: int
    chunks_created: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    ollama_status: bool
    collection_count: int

# --- Helper Functions ---

async def initialize_models():
    """Initialize all models and connections with optimizations."""
    global embedding_model, chroma_client, collection, ollama_available
    try:
        logger.info("Loading embedding model...")
        # Use a smaller, faster model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        embedding_model.max_seq_length = 256  # Limit sequence length for speed
        
        logger.info("Initializing ChromaDB...")
        chroma_client = chromadb.Client(Settings(
            persist_directory="./chroma_db", 
            anonymized_telemetry=False
        ))
        collection = chroma_client.get_or_create_collection(
            name="safety_documents", 
            metadata={"description": "School safety document embeddings"}
        )
        ollama_available = await check_ollama_connection()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise

async def load_documents_from_directory():
    """Load all PDF files from the documents directory with optimizations."""
    global collection, embedding_model
    
    if not os.path.exists(config.DOCUMENTS_DIR):
        os.makedirs(config.DOCUMENTS_DIR)
        logger.info(f"Created documents directory: {config.DOCUMENTS_DIR}")
        return
    
    pdf_files = glob.glob(os.path.join(config.DOCUMENTS_DIR, "*.pdf"))
    
    if not pdf_files:
        logger.info(f"No PDF files found in {config.DOCUMENTS_DIR} directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    total_chunks = 0
    processed_files = []
    
    for pdf_path in pdf_files:
        try:
            filename = os.path.basename(pdf_path)
            
            # Check if already processed
            existing_results = collection.query(
                query_texts=["dummy"],
                n_results=1,
                where={"source": filename}
            )
            
            if existing_results.get('documents') and existing_results['documents'][0]:
                logger.info(f"File {filename} already exists, skipping...")
                continue
            
            with open(pdf_path, 'rb') as file:
                pdf_content = file.read()
            
            pages = extract_text_from_pdf(pdf_content)
            if not pages:
                continue
            
            documents, metadatas, ids = [], [], []
            for page_data in pages:
                chunks = chunk_text(page_data["text"])
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({"source": filename, "page": page_data["page_number"]})
                    ids.append(f"{filename}_{page_data['page_number']}_{i}_{uuid.uuid4().hex[:8]}")
            
            if not documents:
                continue
            
            # Batch embedding generation for speed
            embeddings = embedding_model.encode(
                documents, 
                batch_size=config.EMBEDDING_BATCH_SIZE,
                show_progress_bar=False
            ).tolist()
            
            collection.add(
                documents=documents, 
                metadatas=metadatas, 
                embeddings=embeddings, 
                ids=ids
            )
            
            total_chunks += len(documents)
            processed_files.append(filename)
            logger.info(f"Processed {filename}: {len(documents)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            continue
    
    if processed_files:
        logger.info(f"Loaded {len(processed_files)} files with {total_chunks} chunks")

async def check_ollama_connection() -> bool:
    """Check Ollama connection with shorter timeout."""
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            if config.OLLAMA_MODEL in model_names:
                logger.info(f"Ollama connected. Model {config.OLLAMA_MODEL} available.")
                return True
            else:
                logger.warning(f"Model {config.OLLAMA_MODEL} not found.")
                return False
        return False
    except Exception as e:
        logger.warning(f"Ollama connection failed: {e}")
        return False

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Optimized text chunking."""
    chunk_size = chunk_size or config.MAX_CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    
    # Simple sentence splitting for speed
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Quick overlap - just take last few words
            words = current_chunk.split()
            overlap_text = " ".join(words[-overlap//10:]) if len(words) > overlap//10 else ""
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += (" " + sentence) if current_chunk else sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 30]  # Reduced minimum length

def extract_text_from_pdf(pdf_content: bytes) -> List[dict]:
    """Fast PDF text extraction."""
    try:
        with pymupdf.open(stream=pdf_content, filetype="pdf") as doc:
            pages = []
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text and len(text.strip()) > 50:  # Reduced minimum
                    cleaned_text = re.sub(r'\s+', ' ', text).strip()
                    pages.append({"page_number": page_num + 1, "text": cleaned_text})
            return pages
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

async def stream_ollama_response(query: str, context: str) -> AsyncGenerator[str, None]:
    """Stream response from Ollama for faster perceived performance."""
    if not ollama_available:
        yield "Error: AI model unavailable. Please try again later."
        return
    
    if len(context.strip()) < 50:
        yield "I don't have enough information to answer this question."
        return
    
    full_prompt = SYSTEM_PROMPT.format(
        context=context[:config.MAX_CONTEXT_LENGTH], 
        query=query
    )
    
    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": True,  # Enable streaming
                "options": {
                    "temperature": 0.1,
                    "num_predict": 512,  # Reduced for faster response
                    "top_p": 0.8,
                    "repeat_penalty": 1.1,
                    "num_ctx": 2048  # Reduced context window
                }
            },
            timeout=config.OLLAMA_TIMEOUT,
            stream=True
        )
        
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
                    if chunk.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
                    
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama streaming request failed: {e}")
        yield "Error: Could not connect to AI model."
    except Exception as e:
        logger.error(f"Error in streaming response: {e}")
        yield "Error: Unexpected error occurred."

def parse_confidence_from_text(text: str) -> float:
    """Extract confidence from accumulated text."""
    confidence_match = re.search(r"Confidence:\s*(\d{1,3})%", text, re.IGNORECASE)
    if confidence_match:
        try:
            return float(confidence_match.group(1)) / 100.0
        except (ValueError, IndexError):
            pass
    return 0.5  # Default confidence

async def generate_streaming_chat_response(query: str, sources: List[Source]) -> AsyncGenerator[str, None]:
    """Generate streaming response with sources sent first."""
    # Send sources immediately
    sources_chunk = StreamChunk(
        type="sources",
        sources=sources
    )
    yield f"data: {sources_chunk.json()}\n\n"
    
    # Get context for the query
    sanitized_query = re.sub(r'[<>{}]', '', query).strip()[:500]
    query_embedding = embedding_model.encode([sanitized_query]).tolist()[0]
    
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=3,  # Reduced for speed
        include=["documents", "metadatas"]
    )
    
    if not results.get('documents') or not results['documents'][0]:
        error_chunk = StreamChunk(
            type="content",
            content="I don't have enough information to answer this question."
        )
        yield f"data: {error_chunk.json()}\n\n"
        
        done_chunk = StreamChunk(type="done", confidence=0.0)
        yield f"data: {done_chunk.json()}\n\n"
        return
    
    context = "\n\n".join(results['documents'][0])
    accumulated_response = ""
    
    # Stream the content
    async for chunk in stream_ollama_response(sanitized_query, context):
        accumulated_response += chunk
        content_chunk = StreamChunk(
            type="content",
            content=chunk
        )
        yield f"data: {content_chunk.json()}\n\n"
    
    # Extract and send final confidence
    confidence = parse_confidence_from_text(accumulated_response)
    confidence_chunk = StreamChunk(
        type="confidence",
        confidence=confidence
    )
    yield f"data: {confidence_chunk.json()}\n\n"
    
    # Send done signal
    done_chunk = StreamChunk(type="done")
    yield f"data: {done_chunk.json()}\n\n"

# --- API Endpoints ---

@app.post("/api/rag-chat-stream")
async def rag_chat_stream(query: ChatQuery):
    """Stream RAG chat responses for faster perceived performance."""
    try:
        # Get sources quickly
        sanitized_query = re.sub(r'[<>{}]', '', query.query).strip()[:500]
        query_embedding = embedding_model.encode([sanitized_query]).tolist()[0]
        
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=query.max_results,
            include=["documents", "metadatas"]
        )
        
        # Prepare sources
        sources = []
        seen_sources = set()
        if results.get('metadatas') and results['metadatas'][0]:
            for meta in results['metadatas'][0]:
                source_key = f"{meta['source']}:{meta['page']}"
                if source_key not in seen_sources:
                    sources.append(Source(filename=meta['source'], page=meta['page']))
                    seen_sources.add(source_key)
        
        # Stream response
        return StreamingResponse(
            generate_streaming_chat_response(query.query, sources[:3]),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in streaming chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to process streaming query")

@app.post("/api/rag-chat", response_model=ChatResponse)
async def rag_chat(query: ChatQuery):
    """Traditional non-streaming endpoint for compatibility."""
    try:
        start_time = time.time()
        
        sanitized_query = re.sub(r'[<>{}]', '', query.query).strip()[:500]
        query_embedding = embedding_model.encode([sanitized_query]).tolist()[0]
        
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=query.max_results,
            include=["documents", "metadatas"]
        )
        
        if not results.get('documents') or not results['documents'][0]:
            return ChatResponse(
                answer="I don't have enough information to answer this question.",
                sources=[],
                confidence=0.0
            )
        
        context = "\n\n".join(results['documents'][0])
        
        # Collect streaming response
        full_response = ""
        async for chunk in stream_ollama_response(sanitized_query, context):
            full_response += chunk
        
        # Parse response
        confidence = parse_confidence_from_text(full_response)
        clean_answer = re.sub(r"Confidence:\s*\d{1,3}%\s*$", "", full_response, flags=re.IGNORECASE).strip()
        
        # Prepare sources
        sources = []
        seen_sources = set()
        for meta in results['metadatas'][0]:
            source_key = f"{meta['source']}:{meta['page']}"
            if source_key not in seen_sources:
                sources.append(Source(filename=meta['source'], page=meta['page']))
                seen_sources.add(source_key)
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f}s")
        
        return ChatResponse(
            answer=clean_answer or "I couldn't generate a proper response.",
            sources=sources[:3],
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Error in RAG chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to process query")

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        collection_count = collection.count() if collection else 0
        return HealthResponse(
            status="healthy", 
            timestamp=datetime.now().isoformat(), 
            ollama_status=ollama_available, 
            collection_count=collection_count
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy", 
            timestamp=datetime.now().isoformat(), 
            ollama_status=False, 
            collection_count=0
        )

@app.post("/api/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file with optimizations."""
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Validate file
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        pdf_content = await file.read()
        if not pdf_content:
            raise HTTPException(status_code=400, detail="Empty PDF file")
        
        pages = extract_text_from_pdf(pdf_content)
        if not pages:
            raise HTTPException(status_code=400, detail="No readable content found")
        
        documents, metadatas, ids = [], [], []
        for page_data in pages:
            chunks = chunk_text(page_data["text"])
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({"source": file.filename, "page": page_data["page_number"]})
                ids.append(f"{file.filename}_{page_data['page_number']}_{i}_{uuid.uuid4().hex[:8]}")
        
        if not documents:
            raise HTTPException(status_code=400, detail="No meaningful content found")
        
        # Fast batch embedding
        embeddings = embedding_model.encode(
            documents, 
            batch_size=config.EMBEDDING_BATCH_SIZE,
            show_progress_bar=False
        ).tolist()
        
        collection.add(
            documents=documents, 
            metadatas=metadatas, 
            embeddings=embeddings, 
            ids=ids
        )
        
        logger.info(f"Processed {file.filename}: {len(documents)} chunks")
        return UploadResponse(
            message="PDF processed successfully", 
            filename=file.filename, 
            pages_processed=len(pages), 
            chunks_created=len(documents)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)