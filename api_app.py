from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
import json
import asyncio
import time
import sys
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai
import io

load_dotenv()

app = FastAPI(title="Enhanced PDF QA API", description="Hackathon PDF Question Answering API with Advanced Features")
security = HTTPBearer()

API_KEY = os.getenv("API_KEY", "hackrx-api-key-2025")

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]
    webhook_url: Optional[str] = None

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

def download_pdf_optimized(url: str) -> str:
    """Ultra-fast PDF download with streaming"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        with requests.get(url, headers=headers, timeout=20, stream=True, verify=False) as response:
            response.raise_for_status()
            
            content = b""
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > 10 * 1024 * 1024:  # Limit to 10MB
                    break
        
        pdf_reader = PdfReader(io.BytesIO(content))
        
        # Process all pages for better content
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
            if len(text) > 100000:  # Limit text length
                break
        
        return text
    except Exception as e:
        raise HTTPException(400, f"PDF error: {str(e)}")

def process_questions_with_gemini(pdf_text: str, questions: List[str]) -> List[str]:
    """Direct Gemini processing - bypasses LangChain issues"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        answers = []
        for question in questions:
            prompt = f"""
You are an expert insurance policy analyzer. Based on the following National Parivar Mediclaim Plus Policy document, answer the question with specific details.

POLICY DOCUMENT:
{pdf_text[:15000]}

QUESTION: {question}

INSTRUCTIONS:
- Provide a detailed, accurate answer based ONLY on the document content
- Include specific numbers, timeframes, conditions, and clauses
- Use professional insurance terminology
- If information is not found in the document, state that clearly
- Be comprehensive but concise

ANSWER:"""
            
            try:
                response = model.generate_content(prompt)
                if response.text and len(response.text.strip()) > 10:
                    answer = response.text.strip()
                    answers.append(answer)
                else:
                    answers.append("Unable to generate a proper answer from the document content")
                    
            except Exception as question_error:
                answers.append(f"Unable to process this specific question: {str(question_error)}")
        
        return answers
        
    except Exception as e:
        error_msg = f"Gemini API processing error: {str(e)}"
        return [error_msg] * len(questions)

@app.get("/debug")
async def debug_info():
    try:
        import google.generativeai as genai
        
        # Test Google API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found")
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        test_response = model.generate_content("Test message - respond with 'API working'")
        
        # Test PDF access
        pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        pdf_response = requests.get(pdf_url, timeout=10)
        
        return {
            "status": "debug_success",
            "google_api_key_exists": bool(api_key),
            "google_api_key_length": len(api_key) if api_key else 0,
            "google_api_test": test_response.text[:100] if test_response.text else "No response",
            "pdf_access_status": pdf_response.status_code,
            "pdf_size_bytes": len(pdf_response.content),
            "python_version": sys.version[:50],
            "environment_check": "All systems functional"
        }
    except Exception as e:
        return {
            "status": "debug_failed", 
            "error": str(e),
            "error_type": type(e)._name_,
            "google_api_key_exists": bool(os.getenv("GOOGLE_API_KEY")),
            "suggestion": "Check Google API key in Render environment variables"
        }

async def send_webhook(webhook_url: str, data: dict):
    """Send results to webhook URL"""
    try:
        response = requests.post(
            webhook_url, 
            json=data, 
            timeout=15,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Webhook delivery failed: {e}")
        return False

@app.post("/hackrx/run")
async def process_questions(
    request: QuestionRequest,
    token: str = Depends(verify_token)
):
    """Direct Gemini processing - reliable and fast"""
    try:
        # Validation
        if not request.documents:
            raise HTTPException(400, "Documents URL is required")
        
        if not request.questions:
            raise HTTPException(400, "At least one question is required")
        
        # Process questions with timeout
        async def fast_process():
            # Download and process PDF
            pdf_text = download_pdf_optimized(request.documents)
            if not pdf_text.strip():
                raise HTTPException(400, "No text content found in PDF")
            
            # Process with direct Gemini (no LangChain complexity)
            answers = process_questions_with_gemini(pdf_text, request.questions)
            
            return answers
        
        # Execute with timeout
        answers = await asyncio.wait_for(fast_process(), timeout=25.0)
        
        # Send webhook if provided
        if request.webhook_url:
            webhook_data = {"answers": answers}
            await send_webhook(request.webhook_url, webhook_data)
        
        # Return platform-compatible format
        return {"answers": answers}
        
    except asyncio.TimeoutError:
        return {"answers": ["Request timeout - unable to process questions within time limit"]}
    except HTTPException:
        raise
    except Exception as e:
        return {"answers": [f"Processing error occurred: {str(e)}"]}

@app.post("/webhook/callback")
async def webhook_callback(request: Request):
    try:
        data = await request.json()
        print("Webhook received:", json.dumps(data, indent=2))
        
        return {
            "status": "received",
            "detail": "Webhook processed successfully",
            "timestamp": data.get("timestamp", "not provided"),
            "processed": True
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": f"Failed to process webhook: {str(e)}"
        }

@app.get("/")
async def root():
    return {
        "message": "Enhanced Hackathon PDF QA API is running",
        "version": "3.0.0",
        "features": [
            "Direct Gemini API integration",
            "Optimized PDF processing",
            "High-accuracy question answering",
            "Platform-compatible responses",
            "Robust error handling"
        ],
        "endpoints": ["/hackrx/run", "/webhook/callback", "/health", "/debug"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "hackrx-pdf-qa-api",
        "format": "platform-compatible"
    }

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))