"""
JanSpandana.AI - Enhanced AI-Aware FastAPI Backend Server
Full integration with AI conversation engine and intelligent data processing
Includes AI performance monitoring, fallback handling, and detailed insights

ENHANCEMENTS:
- AI-aware error handling with graceful fallbacks
- AI insights included in all responses
- Performance monitoring and optimization tracking
- Comprehensive health checks for AI services
- Detailed logging for AI decision making
- Real-time AI confidence and reasoning exposure
"""

import os
import asyncio
import uuid
import time
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

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enhanced imports for AI integration
from database import db_manager, db_ops, check_database_health
from speech_services import speech_processor, process_voice_to_text, process_text_to_voice, check_speech_services_health

# Import the new AI-powered engines
try:
    from conversation_engine import jan_spandana_ai as ai_conversation_engine, start_conversation, process_message, end_conversation_session
    AI_CONVERSATION_AVAILABLE = True
except ImportError as e:
    logger.error(f"AI Conversation Engine import failed: {e}")
    AI_CONVERSATION_AVAILABLE = False
    # Fallback imports if needed
    from conversation_engine import start_conversation, process_message, end_conversation_session

try:
    from data_processor import ai_conversation_processor
    AI_DATA_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logger.error(f"AI Data Processor import failed: {e}")
    AI_DATA_PROCESSOR_AVAILABLE = False

# Environment
from dotenv import load_dotenv
load_dotenv()

# Enhanced Pydantic models with AI insights
class ConversationStartRequest(BaseModel):
    village_name: Optional[str] = None
    user_phone: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class AIInsights(BaseModel):
    """AI processing insights for transparency"""
    confidence_score: Optional[float] = None
    reasoning: Optional[str] = None
    ai_model_used: Optional[str] = None
    processing_time_ms: Optional[float] = None
    fallback_used: Optional[bool] = False
    sentiment_detected: Optional[str] = None
    urgency_level: Optional[str] = None

class ConversationStartResponse(BaseModel):
    success: bool
    conversation_id: str
    initial_audio_url: Optional[str] = None
    initial_text: str
    session_data: Dict[str, Any]
    ai_insights: Optional[AIInsights] = None

class VoiceMessageRequest(BaseModel):
    conversation_id: str
    audio_format: str = 'webm'
    region_hint: str = 'coastal'

class EnhancedProcessingStats(BaseModel):
    """Enhanced processing statistics with AI metrics"""
    total_processing_time: float
    stt_processing_time: float
    ai_processing_time: float
    tts_processing_time: float
    stt_confidence: float
    ai_confidence: float
    audio_size_bytes: int
    response_audio_size: int
    ai_fallback_used: bool = False
    performance_optimization: Optional[str] = None

class VoiceMessageResponse(BaseModel):
    success: bool
    conversation_id: str
    user_transcript: str
    ai_response_text: str
    ai_response_audio_url: Optional[str] = None
    conversation_state: Dict[str, Any]
    processing_stats: EnhancedProcessingStats
    ai_insights: Optional[AIInsights] = None

class TextMessageRequest(BaseModel):
    conversation_id: str
    user_input: str
    
class ConversationEndRequest(BaseModel):
    conversation_id: str
    user_satisfaction: Optional[int] = None
    completion_reason: str = 'user_initiated'

class EnhancedHealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, bool]
    ai_services: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    version: str = "2.0.0-ai-enhanced"

# AI Service Health Monitoring
class AIServiceMonitor:
    """Monitor AI service health and performance"""
    
    def __init__(self):
        self.ai_performance_history = []
        self.last_health_check = None
        self.ai_errors_count = 0
        self.successful_ai_calls = 0
        
    async def check_ai_services_health(self) -> Dict[str, Any]:
        """Comprehensive AI services health check"""
        ai_health = {
            'conversation_engine': False,
            'data_processor': False,
            'gemini_api': False,
            'ai_performance': 'unknown'
        }
        
        performance_metrics = {
            'avg_ai_response_time': 0.0,
            'ai_success_rate': 0.0,
            'total_ai_calls': self.successful_ai_calls + self.ai_errors_count
        }
        
        # Test AI Conversation Engine
        if AI_CONVERSATION_AVAILABLE:
            try:
                # Quick test conversation
                test_start = time.time()
                test_result = await start_conversation("health_check_test")
                test_time = (time.time() - test_start) * 1000
                
                if test_result.get('session_started'):
                    ai_health['conversation_engine'] = True
                    performance_metrics['avg_ai_response_time'] = test_time
                    
                    # Cleanup test conversation
                    try:
                        await end_conversation_session("health_check_test", skip_save=True)
                    except:
                        pass  # Ignore cleanup errors
                        
            except Exception as e:
                logger.warning(f"AI Conversation Engine health check failed: {e}")
        
        # Test AI Data Processor
        if AI_DATA_PROCESSOR_AVAILABLE:
            try:
                # Test if AI model is accessible
                if hasattr(ai_conversation_processor, 'ai_model'):
                    ai_health['data_processor'] = True
            except Exception as e:
                logger.warning(f"AI Data Processor health check failed: {e}")
        
        # Test Gemini API connectivity
        try:
            if AI_CONVERSATION_AVAILABLE and hasattr(ai_conversation_engine, 'model'):
                # Simple test prompt
                import google.generativeai as genai
                response = await asyncio.to_thread(
                    ai_conversation_engine.model.generate_content, 
                    "Test connection. Respond with 'OK'"
                )
                if response and response.text:
                    ai_health['gemini_api'] = True
        except Exception as e:
            logger.warning(f"Gemini API health check failed: {e}")
        
        # Calculate success rate
        total_calls = self.successful_ai_calls + self.ai_errors_count
        if total_calls > 0:
            performance_metrics['ai_success_rate'] = self.successful_ai_calls / total_calls
        
        # Determine overall AI performance
        if all(ai_health.values()):
            ai_health['ai_performance'] = 'excellent'
        elif any(ai_health.values()):
            ai_health['ai_performance'] = 'degraded'
        else:
            ai_health['ai_performance'] = 'failed'
        
        self.last_health_check = datetime.now(timezone.utc)
        return ai_health, performance_metrics
    
    def record_ai_success(self, processing_time_ms: float):
        """Record successful AI operation"""
        self.successful_ai_calls += 1
        self.ai_performance_history.append({
            'timestamp': datetime.now(timezone.utc),
            'processing_time_ms': processing_time_ms,
            'success': True
        })
        
        # Keep only last 100 records
        if len(self.ai_performance_history) > 100:
            self.ai_performance_history.pop(0)
    
    def record_ai_error(self, error_type: str):
        """Record AI operation error"""
        self.ai_errors_count += 1
        self.ai_performance_history.append({
            'timestamp': datetime.now(timezone.utc),
            'error_type': error_type,
            'success': False
        })

# Global AI service monitor
ai_monitor = AIServiceMonitor()

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application startup and shutdown with AI initialization"""
    # Startup
    logger.info("🚀 Starting JanSpandana.AI Enhanced Backend Server...")
    
    # Health checks on startup
    db_health = await check_database_health()
    speech_health = await check_speech_services_health()
    ai_health, ai_performance = await ai_monitor.check_ai_services_health()
    
    if not db_health['overall_healthy']:
        logger.error("❌ Database health check failed!")
        raise Exception("Database connection failed")
    
    if not all(speech_health.values()):
        logger.warning("⚠️ Some speech services may not be fully functional")
    
    if ai_health['ai_performance'] == 'failed':
        logger.error("❌ AI services are not functional - running in fallback mode")
    elif ai_health['ai_performance'] == 'degraded':
        logger.warning("⚠️ AI services partially functional - some features may use fallbacks")
    else:
        logger.info("✅ AI services fully functional")
    
    logger.info("✅ JanSpandana.AI Enhanced Backend Server started successfully")
    logger.info(f"🤖 AI Conversation Engine: {'✅' if ai_health['conversation_engine'] else '❌'}")
    logger.info(f"🧠 AI Data Processor: {'✅' if ai_health['data_processor'] else '❌'}")
    logger.info(f"🔗 Gemini API: {'✅' if ai_health['gemini_api'] else '❌'}")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down JanSpandana.AI Enhanced Backend Server...")
    logger.info(f"📊 Final AI Stats - Success: {ai_monitor.successful_ai_calls}, Errors: {ai_monitor.ai_errors_count}")
    logger.info("✅ JanSpandana.AI Enhanced Backend Server shut down cleanly")

# Create FastAPI application
app = FastAPI(
    title="JanSpandana.AI Enhanced Backend",
    description="AI-powered voice-first grievance redressal system for rural Andhra Pradesh",
    version="2.0.0-ai-enhanced",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enhanced middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Static files for audio storage
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

def extract_ai_insights(ai_response: Dict[str, Any], processing_time_ms: float) -> AIInsights:
    """Extract AI insights for API response"""
    return AIInsights(
        confidence_score=ai_response.get('confidence_score'),
        reasoning=ai_response.get('reasoning'),
        ai_model_used=ai_response.get('ai_model', 'gemini-2.5-flash'),
        processing_time_ms=processing_time_ms,
        fallback_used=ai_response.get('fallback_used', False),
        sentiment_detected=ai_response.get('sentiment_detected'),
        urgency_level=ai_response.get('urgency_level')
    )

# Enhanced API Routes

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Enhanced root endpoint with AI service status"""
    ai_health, ai_performance = await ai_monitor.check_ai_services_health()
    
    return {
        "service": "JanSpandana.AI Enhanced Backend",
        "version": "2.0.0-ai-enhanced",
        "status": "running",
        "ai_status": ai_health['ai_performance'],
        "description": "AI-powered voice-first grievance redressal system for rural Andhra Pradesh",
        "ai_features": {
            "dynamic_conversation": ai_health['conversation_engine'],
            "intelligent_analysis": ai_health['data_processor'],
            "ai_api_connectivity": ai_health['gemini_api']
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "conversation": "/api/v1/conversation",
            "voice": "/api/v1/voice",
            "ai_insights": "/api/v1/admin/ai-insights"
        }
    }

@app.get("/health", response_model=EnhancedHealthCheckResponse)
async def enhanced_health_check():
    """Comprehensive health check with AI service monitoring"""
    try:
        # Check traditional services
        db_health = await check_database_health()
        speech_health = await check_speech_services_health()
        
        # Check AI services
        ai_health, ai_performance = await ai_monitor.check_ai_services_health()
        
        # Basic services
        services = {
            "database": db_health['overall_healthy'],
            "speech_recognition": speech_health.get('speech_client', False),
            "text_to_speech": speech_health.get('tts_client', False),
            "credentials": speech_health.get('credentials', False)
        }
        
        # Overall status determination
        basic_services_healthy = all(services.values())
        ai_services_healthy = ai_health['ai_performance'] in ['excellent', 'degraded']
        
        if basic_services_healthy and ai_services_healthy:
            overall_status = "healthy"
        elif basic_services_healthy and ai_health['ai_performance'] == 'degraded':
            overall_status = "ai_degraded"
        elif basic_services_healthy:
            overall_status = "ai_fallback"
        else:
            overall_status = "error"
        
        return EnhancedHealthCheckResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            services=services,
            ai_services=ai_health,
            performance_metrics=ai_performance
        )
    
    except Exception as e:
        logger.error(f"Enhanced health check failed: {str(e)}")
        return EnhancedHealthCheckResponse(
            status="error",
            timestamp=datetime.now(timezone.utc).isoformat(),
            services={"error": False},
            ai_services={"error": True},
            performance_metrics={"error": str(e)}
        )

@app.post("/api/v1/conversation/start", response_model=ConversationStartResponse)
async def start_conversation_api(request: ConversationStartRequest):
    """Enhanced conversation start with AI insights"""
    conversation_id = generate_conversation_id()
    
    try:
        logger.info(f"Starting AI-powered conversation: {conversation_id}")
        
        # Start AI conversation with timing
        ai_start_time = time.time()
        conversation_result = await start_conversation(conversation_id)
        ai_processing_time = (time.time() - ai_start_time) * 1000
        
        if not conversation_result['session_started']:
            ai_monitor.record_ai_error("conversation_start_failed")
            raise HTTPException(status_code=500, detail="Failed to start AI conversation")
        
        # Record successful AI operation
        ai_monitor.record_ai_success(ai_processing_time)
        
        # Get initial AI response
        initial_response = conversation_result['initial_response']
        initial_text = initial_response['telugu_response']
        
        # Generate initial audio response
        tts_start_time = time.time()
        tts_result = await process_text_to_voice(initial_text, 'female')
        tts_processing_time = (time.time() - tts_start_time) * 1000
        
        initial_audio_url = None
        if tts_result['success']:
            audio_filename = generate_audio_filename(conversation_id, 'initial_response')
            initial_audio_url = await save_audio_file(tts_result['audio_content'], audio_filename)
        
        # Track active conversation
        active_conversations[conversation_id] = {
            'started_at': datetime.now(timezone.utc).isoformat(),
            'village_name': request.village_name,
            'user_phone': request.user_phone,
            'metadata': request.metadata,
            'message_count': 0,
            'ai_powered': True,
            'total_ai_processing_time': ai_processing_time
        }
        
        # Extract AI insights
        ai_insights = extract_ai_insights(initial_response, ai_processing_time)
        
        return ConversationStartResponse(
            success=True,
            conversation_id=conversation_id,
            initial_audio_url=initial_audio_url,
            initial_text=initial_text,
            session_data={
                'conversation_id': conversation_id,
                'stage': 'greeting',
                'started_at': active_conversations[conversation_id]['started_at'],
                'ai_powered': True
            },
            ai_insights=ai_insights
        )
    
    except Exception as e:
        ai_monitor.record_ai_error("conversation_start_exception")
        logger.error(f"Failed to start conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversation start failed: {str(e)}")

@app.post("/api/v1/conversation/voice", response_model=VoiceMessageResponse)
async def process_voice_message(
    conversation_id: str = Form(...),
    audio_format: str = Form(default='webm'),
    region_hint: str = Form(default='coastal'),
    audio_file: UploadFile = File(...)
):
    """Enhanced voice message processing with AI insights and performance monitoring"""
    try:
        if conversation_id not in active_conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        logger.info(f"Processing AI-powered voice message for: {conversation_id}")
        
        # Read audio file
        audio_content = await audio_file.read()
        
        # Process speech-to-text
        stt_start_time = time.time()
        stt_result = await process_voice_to_text(audio_content, audio_format)
        stt_processing_time = (time.time() - stt_start_time) * 1000
        
        if not stt_result['success']:
            raise HTTPException(status_code=500, detail=f"Speech recognition failed: {stt_result['error']}")
        
        user_transcript = stt_result['best_transcript']
        
        # Process with AI conversation engine
        ai_start_time = time.time()
        ai_result = await process_message(
            conversation_id, 
            user_transcript,
            audio_metadata={
                'format': audio_format,
                'size_bytes': len(audio_content),
                'confidence': stt_result['best_confidence']
            }
        )
        ai_processing_time = (time.time() - ai_start_time) * 1000
        
        if not ai_result['success']:
            ai_monitor.record_ai_error("message_processing_failed")
            raise HTTPException(status_code=500, detail="AI processing failed")
        
        # Record successful AI operation
        ai_monitor.record_ai_success(ai_processing_time)
        
        ai_response_text = ai_result['ai_response']['telugu_response']
        
        # Generate audio response
        tts_start_time = time.time()
        tts_result = await process_text_to_voice(ai_response_text, 'female')
        tts_processing_time = (time.time() - tts_start_time) * 1000
        
        ai_response_audio_url = None
        if tts_result['success']:
            audio_filename = generate_audio_filename(conversation_id, 'ai_response')
            ai_response_audio_url = await save_audio_file(tts_result['audio_content'], audio_filename)
        
        # Update conversation tracking
        active_conversations[conversation_id]['message_count'] += 1
        active_conversations[conversation_id]['last_activity'] = datetime.now(timezone.utc).isoformat()
        active_conversations[conversation_id]['total_ai_processing_time'] += ai_processing_time
        
        # Enhanced processing statistics
        processing_stats = EnhancedProcessingStats(
            total_processing_time=stt_processing_time + ai_processing_time + tts_processing_time,
            stt_processing_time=stt_processing_time,
            ai_processing_time=ai_processing_time,
            tts_processing_time=tts_processing_time,
            stt_confidence=stt_result['best_confidence'],
            ai_confidence=ai_result['ai_response'].get('confidence_score', 0.0),
            audio_size_bytes=len(audio_content),
            response_audio_size=len(tts_result['audio_content']) if tts_result['success'] else 0,
            ai_fallback_used=ai_result['ai_response'].get('fallback_used', False),
            performance_optimization="ai_powered" if not ai_result['ai_response'].get('fallback_used') else "fallback_used"
        )
        
        # Extract AI insights
        ai_insights = extract_ai_insights(ai_result['ai_response'], ai_processing_time)
        
        return VoiceMessageResponse(
            success=True,
            conversation_id=conversation_id,
            user_transcript=user_transcript,
            ai_response_text=ai_response_text,
            ai_response_audio_url=ai_response_audio_url,
            conversation_state=ai_result['conversation_state'],
            processing_stats=processing_stats,
            ai_insights=ai_insights
        )
    
    except HTTPException:
        raise
    except Exception as e:
        ai_monitor.record_ai_error("voice_processing_exception")
        logger.error(f"Voice processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")

@app.post("/api/v1/conversation/text", response_model=VoiceMessageResponse)
async def process_text_message(request: TextMessageRequest):
    """Enhanced text message processing with AI insights"""
    try:
        if request.conversation_id not in active_conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        logger.info(f"Processing AI-powered text message for: {request.conversation_id}")
        
        # Process with AI conversation engine
        ai_start_time = time.time()
        ai_result = await process_message(request.conversation_id, request.user_input)
        ai_processing_time = (time.time() - ai_start_time) * 1000
        
        if not ai_result['success']:
            ai_monitor.record_ai_error("text_processing_failed")
            raise HTTPException(status_code=500, detail="AI processing failed")
        
        # Record successful AI operation
        ai_monitor.record_ai_success(ai_processing_time)
        
        ai_response_text = ai_result['ai_response']['telugu_response']
        
        # Generate audio response
        tts_start_time = time.time()
        tts_result = await process_text_to_voice(ai_response_text, 'female')
        tts_processing_time = (time.time() - tts_start_time) * 1000
        
        ai_response_audio_url = None
        if tts_result['success']:
            audio_filename = generate_audio_filename(request.conversation_id, 'ai_response')
            ai_response_audio_url = await save_audio_file(tts_result['audio_content'], audio_filename)
        
        # Update tracking
        active_conversations[request.conversation_id]['message_count'] += 1
        active_conversations[request.conversation_id]['last_activity'] = datetime.now(timezone.utc).isoformat()
        active_conversations[request.conversation_id]['total_ai_processing_time'] += ai_processing_time
        
        # Enhanced processing statistics
        processing_stats = EnhancedProcessingStats(
            total_processing_time=ai_processing_time + tts_processing_time,
            stt_processing_time=0.0,  # No STT for text input
            ai_processing_time=ai_processing_time,
            tts_processing_time=tts_processing_time,
            stt_confidence=1.0,  # Text input is 100% confident
            ai_confidence=ai_result['ai_response'].get('confidence_score', 0.0),
            audio_size_bytes=0,  # No input audio for text
            response_audio_size=len(tts_result['audio_content']) if tts_result['success'] else 0,
            ai_fallback_used=ai_result['ai_response'].get('fallback_used', False),
            performance_optimization="text_optimized"
        )
        
        # Extract AI insights
        ai_insights = extract_ai_insights(ai_result['ai_response'], ai_processing_time)
        
        return VoiceMessageResponse(
            success=True,
            conversation_id=request.conversation_id,
            user_transcript=request.user_input,
            ai_response_text=ai_response_text,
            ai_response_audio_url=ai_response_audio_url,
            conversation_state=ai_result['conversation_state'],
            processing_stats=processing_stats,
            ai_insights=ai_insights
        )
    
    except HTTPException:
        raise
    except Exception as e:
        ai_monitor.record_ai_error("text_processing_exception")
        logger.error(f"Text processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")

@app.post("/api/v1/conversation/end")
async def end_conversation_api(request: ConversationEndRequest):
    """Enhanced conversation end with AI analysis summary"""
    try:
        if request.conversation_id not in active_conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        logger.info(f"Ending AI-powered conversation: {request.conversation_id}")
        
        # End AI conversation
        ai_start_time = time.time()
        end_result = await end_conversation_session(request.conversation_id)
        ai_processing_time = (time.time() - ai_start_time) * 1000
        
        # Update tracking data
        conversation_data = active_conversations[request.conversation_id]
        conversation_data.update({
            'ended_at': datetime.now(timezone.utc).isoformat(),
            'completion_reason': request.completion_reason,
            'user_satisfaction': request.user_satisfaction,
            'total_ai_processing_time': conversation_data.get('total_ai_processing_time', 0) + ai_processing_time
        })
        
        # Enhanced summary with AI insights
        ai_summary = None
        if end_result.get('success') and 'summary' in end_result:
            ai_summary = end_result['summary'].get('ai_insights', {})
        
        # Clean up active conversation
        del active_conversations[request.conversation_id]
        
        return {
            'success': True,
            'conversation_id': request.conversation_id,
            'conversation_ended': True,
            'summary': end_result.get('summary', {}),
            'final_stats': conversation_data,
            'ai_insights': ai_summary,
            'total_ai_processing_time_ms': conversation_data.get('total_ai_processing_time', 0)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        ai_monitor.record_ai_error("conversation_end_exception")
        logger.error(f"End conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to end conversation: {str(e)}")

@app.get("/api/v1/conversation/{conversation_id}/status")
async def get_conversation_status(conversation_id: str):
    """Enhanced conversation status with AI insights"""
    if conversation_id not in active_conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation_data = active_conversations[conversation_id]
    
    return {
        'conversation_id': conversation_id,
        'active': True,
        'data': conversation_data,
        'ai_powered': conversation_data.get('ai_powered', False),
        'ai_performance': {
            'total_ai_time_ms': conversation_data.get('total_ai_processing_time', 0),
            'avg_ai_time_per_message': (
                conversation_data.get('total_ai_processing_time', 0) / 
                max(conversation_data.get('message_count', 1), 1)
            )
        }
    }

@app.get("/api/v1/admin/stats")
async def get_admin_stats():
    """Enhanced administrative statistics with AI metrics"""
    ai_health, ai_performance = await ai_monitor.check_ai_services_health()
    
    # Calculate AI performance metrics
    total_ai_conversations = len([conv for conv in active_conversations.values() if conv.get('ai_powered')])
    avg_ai_processing_time = 0.0
    
    if total_ai_conversations > 0:
        total_ai_time = sum(conv.get('total_ai_processing_time', 0) for conv in active_conversations.values() if conv.get('ai_powered'))
        avg_ai_processing_time = total_ai_time / total_ai_conversations
    
    return {
        'active_conversations': len(active_conversations),
        'ai_powered_conversations': total_ai_conversations,
        'total_conversations_today': len(active_conversations),
        'service_health': await enhanced_health_check(),
        'ai_metrics': {
            'successful_ai_calls': ai_monitor.successful_ai_calls,
            'ai_error_count': ai_monitor.ai_errors_count,
            'ai_success_rate': ai_performance.get('ai_success_rate', 0.0),
            'avg_ai_processing_time_ms': avg_ai_processing_time,
            'ai_service_status': ai_health['ai_performance']
        },
        'server_uptime': datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/v1/admin/ai-insights")
async def get_ai_insights():
    """AI service insights and performance analytics"""
    ai_health, ai_performance = await ai_monitor.check_ai_services_health()
    
    # Recent AI performance history
    recent_performance = ai_monitor.ai_performance_history[-20:] if len(ai_monitor.ai_performance_history) > 0 else []
    
    return {
        'ai_service_status': ai_health,
        'performance_metrics': ai_performance,
        'recent_performance': recent_performance,
        'ai_capabilities': {
            'dynamic_conversation': AI_CONVERSATION_AVAILABLE,
            'intelligent_analysis': AI_DATA_PROCESSOR_AVAILABLE,
            'fallback_available': True
        },
        'optimization_recommendations': _generate_ai_optimization_recommendations()
    }

def _generate_ai_optimization_recommendations() -> List[str]:
    """Generate AI optimization recommendations based on performance"""
    recommendations = []
    
    success_rate = ai_monitor.successful_ai_calls / max(ai_monitor.successful_ai_calls + ai_monitor.ai_errors_count, 1)
    
    if success_rate < 0.9:
        recommendations.append("Consider implementing more robust AI error handling")
    
    if len(ai_monitor.ai_performance_history) > 10:
        avg_time = sum(h.get('processing_time_ms', 0) for h in ai_monitor.ai_performance_history if h.get('success')) / len([h for h in ai_monitor.ai_performance_history if h.get('success')])
        if avg_time > 3000:  # 3 seconds
            recommendations.append("AI response times are high - consider prompt optimization")
    
    if ai_monitor.ai_errors_count > ai_monitor.successful_ai_calls * 0.1:
        recommendations.append("High AI error rate detected - check API quotas and network connectivity")
    
    if not recommendations:
        recommendations.append("AI performance is optimal")
    
    return recommendations

# Enhanced error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Enhanced global exception handler with AI error tracking"""
    error_id = str(uuid.uuid4())
    
    # Track if this is an AI-related error
    ai_related = any(keyword in str(exc).lower() for keyword in ['ai', 'gemini', 'model', 'conversation', 'analysis'])
    
    if ai_related:
        ai_monitor.record_ai_error("unhandled_exception")
    
    logger.error(f"Unhandled exception ({error_id}): {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": error_id,
            "ai_related": ai_related,
            "fallback_available": True
        }
    )

# Development server runner
if __name__ == "__main__":
    logger.info("Starting JanSpandana.AI Enhanced FastAPI Backend Server")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
