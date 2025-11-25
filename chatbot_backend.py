"""
FastAPI Backend for Chatbot with Gemini Integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import asyncio
import google.generativeai as genai

# ============================================
# CONFIGURE YOUR GEMINI API KEY HERE
# ============================================
GEMINI_API_KEY = "AIzaSyBlxBE9ozDYAzfjZQqyM0Z7vqw2krArEHg"
# ============================================

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")  # or "gemini-1.5-pro-latest"

app = FastAPI(title="Chatbot API", version="1.0.0")

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    conversation_id: str


# In-memory conversation store (replace with Redis/DB for production)
conversations: dict = {}


async def generate_response(message: str) -> str:
    """Generate response using Gemini API"""
    try:
        response = model.generate_content(message)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"


async def stream_response(message: str) -> AsyncGenerator[str, None]:
    """Streaming response generator using Gemini"""
    try:
        response = model.generate_content(message, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Error: {str(e)}"


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint"""
    try:
        response = await generate_response(request.message)
        conv_id = request.conversation_id or "default"
        
        # Store conversation history
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append({
            "user": request.message,
            "assistant": response
        })
        
        return ChatResponse(response=response, conversation_id=conv_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint"""
    return StreamingResponse(
        stream_response(request.message),
        media_type="text/event-stream"
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """Retrieve conversation history"""
    if conv_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conv_id, "messages": conversations[conv_id]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
