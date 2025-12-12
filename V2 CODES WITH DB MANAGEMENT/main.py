"""
JanSpandana.AI - FastAPI Backend Server
Main server handling voice processing, AI conversations, and database operations
Optimized for concurrent village users and rural network conditions
"""

import os
import asyncio
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import json
import io
from contextlib import asynccontextmanager

# FastAPI and HTTP
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Local imports
from database import db_manager, db_ops, check_database_health
from speech_services import speech_processor, process_voice_to_text, process_text_to_voice, check_speech_services_health
from conversation_engine import jan_spandana, start_conversation, process_message, end_conversation_session

# Environment
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class ConversationStartRequest(BaseModel):
    village_name: Optional[str] = None
    user_phone: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class ConversationStartResponse(BaseModel):
    success: bool
    conversation_id: str
    initial_audio_url: Optional[str] = None
    initial_text: str
    session_data: Dict[str, Any]

class VoiceMessageRequest(BaseModel):
    conversation_id: str
    audio_format: str = 'webm'
    region_hint: str = 'coastal'

class VoiceMessageResponse(BaseModel):
    success: bool
    conversation_id: str
    user_transcript: str
    ai_response_text: str
    ai_response_audio_url: Optional[str] = None
    conversation_state: Dict[str, Any]
    processing_stats: Dict[str, Any]

class TextMessageRequest(BaseModel):
    conversation_id: str
    user_input: str
    
class ConversationEndRequest(BaseModel):
    conversation_id: str
    user_satisfaction: Optional[int] = None
    completion_reason: str = 'user_initiated'

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, bool]
    version: str = "1.0.0"

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting JanSpandana.AI Backend Server...")
    
    # Health checks on startup
    db_health = await check_database_health()
    speech_health = await check_speech_services_health()
    
    if not db_health['overall_healthy']:
        logger.error("❌ Database health check failed!")
        raise Exception("Database connection failed")
    
    if not all(speech_health.values()):
        logger.warning("⚠️ Some speech services may not be fully functional")
    
    logger.info("✅ JanSpandana.AI Backend Server started successfully")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down JanSpandana.AI Backend Server...")
    # Cleanup active conversations if any
    logger.info("✅ JanSpandana.AI Backend Server shut down cleanly")

# Create FastAPI application
app = FastAPI(
    title="JanSpandana.AI Backend",
    description="Voice-first AI grievance redressal system for rural Andhra Pradesh",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000", "*"],  # Streamlit + frontend
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Static files for audio storage (temporary)
os.makedirs("static/audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for tracking
active_conversations: Dict[str, Dict] = {}
processing_queue = asyncio.Queue()

# Helper functions
def generate_conversation_id() -> str:
    """Generate unique conversation ID"""
    return f"conv_{uuid.uuid4().hex[:12]}_{datetime.now().strftime('%H%M%S')}"

def generate_audio_filename(conversation_id: str, audio_type: str) -> str:
    """Generate unique audio filename"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    return f"audio/{conversation_id}_{audio_type}_{timestamp}.mp3"

async def save_audio_file(audio_content: bytes, filename: str) -> str:
    """Save audio content to file and return URL"""
    file_path = os.path.join("static", filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(audio_content)
    
    return f"/static/{filename}"

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "JanSpandana.AI Backend",
        "version": "1.0.0",
        "status": "running",
        "description": "Voice-first AI grievance redressal system for rural Andhra Pradesh",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "conversation": "/api/v1/conversation",
            "voice": "/api/v1/voice"
        }
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check all services
        db_health = await check_database_health()
        speech_health = await check_speech_services_health()
        
        # AI conversation engine health (simplified)
        conversation_health = len(active_conversations) < 100  # Simple check
        
        services = {
            "database": db_health['overall_healthy'],
            "speech_recognition": speech_health.get('speech_client', False),
            "text_to_speech": speech_health.get('tts_client', False),
            "conversation_ai": conversation_health,
            "credentials": speech_health.get('credentials', False)
        }
        
        overall_status = "healthy" if all(services.values()) else "degraded"
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            services=services
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheckResponse(
            status="error",
            timestamp=datetime.now(timezone.utc).isoformat(),
            services={"error": False}
        )

@app.post("/api/v1/conversation/start", response_model=ConversationStartResponse)
async def start_conversation_api(request: ConversationStartRequest):
    """Start a new conversation session"""
    try:
        # Generate conversation ID
        conversation_id = generate_conversation_id()
        
        logger.info(f"Starting conversation: {conversation_id}")
        
        # Start AI conversation
        conversation_result = await start_conversation(conversation_id)
        
        if not conversation_result['session_started']:
            raise HTTPException(status_code=500, detail="Failed to start AI conversation")
        
        # Get initial AI response
        initial_response = conversation_result['initial_response']
        initial_text = initial_response['telugu_response']
        
        # Generate initial audio response
        tts_result = await process_text_to_voice(initial_text, 'female')
        
        initial_audio_url = None
        if tts_result['success']:
            # Save audio file
            audio_filename = generate_audio_filename(conversation_id, 'initial_response')
            initial_audio_url = await save_audio_file(tts_result['audio_content'], audio_filename)
        
        # Track active conversation
        active_conversations[conversation_id] = {
            'started_at': datetime.now(timezone.utc).isoformat(),
            'village_name': request.village_name,
            'user_phone': request.user_phone,
            'metadata': request.metadata,
            'message_count': 0
        }
        
        return ConversationStartResponse(
            success=True,
            conversation_id=conversation_id,
            initial_audio_url=initial_audio_url,
            initial_text=initial_text,
            session_data={
                'conversation_id': conversation_id,
                'stage': 'greeting',
                'started_at': active_conversations[conversation_id]['started_at']
            }
        )
    
    except Exception as e:
        logger.error(f"Failed to start conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversation start failed: {str(e)}")

@app.post("/api/v1/conversation/voice", response_model=VoiceMessageResponse)
async def process_voice_message(
    conversation_id: str = Form(...),
    audio_format: str = Form(default='webm'),
    region_hint: str = Form(default='coastal'),
    audio_file: UploadFile = File(...)
):
    """Process voice message and return AI response"""
    try:
        if conversation_id not in active_conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        logger.info(f"Processing voice message for conversation: {conversation_id}")
        
        # Read audio file
        audio_content = await audio_file.read()
        
        # Process speech-to-text
        start_time = datetime.now()
        stt_result = await process_voice_to_text(audio_content, audio_format)
        stt_processing_time = (datetime.now() - start_time).total_seconds()
        
        if not stt_result['success']:
            raise HTTPException(status_code=500, detail=f"Speech recognition failed: {stt_result['error']}")
        
        user_transcript = stt_result['best_transcript']
        
        # Process with AI conversation engine
        ai_start_time = datetime.now()
        ai_result = await process_message(
            conversation_id, 
            user_transcript,
            audio_metadata={
                'format': audio_format,
                'size_bytes': len(audio_content),
                'confidence': stt_result['best_confidence']
            }
        )
        ai_processing_time = (datetime.now() - ai_start_time).total_seconds()
        
        if not ai_result['success']:
            raise HTTPException(status_code=500, detail="AI processing failed")
        
        ai_response_text = ai_result['ai_response']['telugu_response']
        
        # Generate audio response
        tts_start_time = datetime.now()
        tts_result = await process_text_to_voice(ai_response_text, 'female')
        tts_processing_time = (datetime.now() - tts_start_time).total_seconds()
        
        ai_response_audio_url = None
        if tts_result['success']:
            audio_filename = generate_audio_filename(conversation_id, 'ai_response')
            ai_response_audio_url = await save_audio_file(tts_result['audio_content'], audio_filename)
        
        # Update conversation tracking
        active_conversations[conversation_id]['message_count'] += 1
        active_conversations[conversation_id]['last_activity'] = datetime.now(timezone.utc).isoformat()
        
        # Compile processing statistics
        processing_stats = {
            'total_processing_time': stt_processing_time + ai_processing_time + tts_processing_time,
            'stt_processing_time': stt_processing_time,
            'ai_processing_time': ai_processing_time,
            'tts_processing_time': tts_processing_time,
            'stt_confidence': stt_result['best_confidence'],
            'ai_confidence': ai_result['ai_response'].get('confidence_score', 0.0),
            'audio_size_bytes': len(audio_content),
            'response_audio_size': len(tts_result['audio_content']) if tts_result['success'] else 0
        }
        
        return VoiceMessageResponse(
            success=True,
            conversation_id=conversation_id,
            user_transcript=user_transcript,
            ai_response_text=ai_response_text,
            ai_response_audio_url=ai_response_audio_url,
            conversation_state=ai_result['conversation_state'],
            processing_stats=processing_stats
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")

@app.post("/api/v1/conversation/text", response_model=VoiceMessageResponse)
async def process_text_message(request: TextMessageRequest):
    """Process text message (fallback for voice failures)"""
    try:
        if request.conversation_id not in active_conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        logger.info(f"Processing text message for conversation: {request.conversation_id}")
        
        # Process with AI conversation engine
        ai_result = await process_message(request.conversation_id, request.user_input)
        
        if not ai_result['success']:
            raise HTTPException(status_code=500, detail="AI processing failed")
        
        ai_response_text = ai_result['ai_response']['telugu_response']
        
        # Generate audio response
        tts_result = await process_text_to_voice(ai_response_text, 'female')
        
        ai_response_audio_url = None
        if tts_result['success']:
            audio_filename = generate_audio_filename(request.conversation_id, 'ai_response')
            ai_response_audio_url = await save_audio_file(tts_result['audio_content'], audio_filename)
        
        # Update tracking
        active_conversations[request.conversation_id]['message_count'] += 1
        active_conversations[request.conversation_id]['last_activity'] = datetime.now(timezone.utc).isoformat()
        
        return VoiceMessageResponse(
            success=True,
            conversation_id=request.conversation_id,
            user_transcript=request.user_input,
            ai_response_text=ai_response_text,
            ai_response_audio_url=ai_response_audio_url,
            conversation_state=ai_result['conversation_state'],
            processing_stats={'processing_mode': 'text_only'}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")

@app.post("/api/v1/conversation/end")
async def end_conversation_api(request: ConversationEndRequest):
    """End conversation and save data"""
    try:
        if request.conversation_id not in active_conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        logger.info(f"Ending conversation: {request.conversation_id}")
        
        # End AI conversation
        end_result = await end_conversation_session(request.conversation_id)
        
        # Update tracking data
        conversation_data = active_conversations[request.conversation_id]
        conversation_data.update({
            'ended_at': datetime.now(timezone.utc).isoformat(),
            'completion_reason': request.completion_reason,
            'user_satisfaction': request.user_satisfaction
        })
        
        # Save to database (background task)
        # This could be enhanced with actual database saving
        
        # Clean up active conversation
        del active_conversations[request.conversation_id]
        
        return {
            'success': True,
            'conversation_id': request.conversation_id,
            'conversation_ended': True,
            'summary': end_result.get('summary', {}),
            'final_stats': conversation_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"End conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to end conversation: {str(e)}")

@app.get("/api/v1/conversation/{conversation_id}/status")
async def get_conversation_status(conversation_id: str):
    """Get current conversation status"""
    if conversation_id not in active_conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        'conversation_id': conversation_id,
        'active': True,
        'data': active_conversations[conversation_id]
    }

@app.get("/api/v1/admin/stats")
async def get_admin_stats():
    """Get administrative statistics"""
    return {
        'active_conversations': len(active_conversations),
        'total_conversations_today': len(active_conversations),  # Simplified
        'service_health': await health_check(),
        'server_uptime': datetime.now(timezone.utc).isoformat()
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": str(uuid.uuid4())
        }
    )

# Development server runner
if __name__ == "__main__":
    logger.info("Starting JanSpandana.AI FastAPI Backend Server")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )