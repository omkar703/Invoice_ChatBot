from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.models.schemas import UploadResponse, ErrorResponse
from app.core.rag_pipeline import InvoiceRAGPipeline
from app.core.session_manager import session_manager
from app.core.config import settings

router = APIRouter(prefix="/upload", tags=["upload"])

import os 
from dotenv import load_dotenv
load_dotenv()
def get_rag_pipeline() -> InvoiceRAGPipeline:
    """Dependency to get RAG pipeline instance"""
    return InvoiceRAGPipeline(groq_api_key=os.getenv("GROQ_API_KEY"))


@router.post("/invoices", response_model=UploadResponse)
async def upload_invoices(
    files: List[UploadFile] = File(...),
    rag_pipeline: InvoiceRAGPipeline = Depends(get_rag_pipeline)
):
    """
    Upload invoice files (JSON or CSV) and create a new analysis session
    
    Returns:
        UploadResponse with session_id for subsequent analysis requests
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file types and sizes
    processed_files = []
    for file in files:
        # Check file extension
        if not any(file.filename.lower().endswith(f".{ext}") for ext in settings.allowed_file_types):
            raise HTTPException(
                status_code=400, 
                detail=f"File {file.filename} has unsupported format. Allowed: {settings.allowed_file_types}"
            )
        
        # Read file content
        try:
            content = await file.read()
            
            # Check file size
            if len(content) > settings.max_file_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} exceeds maximum size of {settings.max_file_size} bytes"
                )
            
            processed_files.append((file.filename, content))
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file {file.filename}: {str(e)}")
    
    try:
        # Load documents from files
        documents = rag_pipeline.load_documents_from_files(processed_files)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents could be extracted from the uploaded files")
        
        # Create vector store
        vector_store = rag_pipeline.create_vector_store(documents)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Create agent
        agent = rag_pipeline.create_agent(retriever)
        
        # Create session
        session_id = session_manager.create_session(vector_store, agent)
        
        return UploadResponse(
            session_id=session_id,
            message="Files uploaded and processed successfully",
            files_processed=len(processed_files)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
