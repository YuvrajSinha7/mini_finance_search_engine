import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Finance Assistant API")

# Add CORS middleware to allow requests from our local HTML file
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local dev
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI Client
# Make sure OPENAI_API_KEY is set in your .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pydantic models for request validation
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in the .env file.")
    
    try:
        # Construct the conversation history for OpenAI
        messages = [
            {"role": "system", "content": "You are a professional mini Finance Assistant. You provide clear, structured, and insightful answers to financial questions. Feel free to use markdown formatting, bullet points, and code blocks to make your point clearer."}
        ]
        
        # Append previous user/assistant interaction history
        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})
            
        # Append the new user message
        messages.append({"role": "user", "content": request.message})
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # or gpt-4o depending on preference
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract response content
        bot_message = response.choices[0].message.content
        return {"response": bot_message}
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
