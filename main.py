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

import redis.asyncio as redis

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

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Redis configuration (env-driven; required for cache/session persistence)
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_USERNAME = os.getenv("REDIS_USERNAME")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_USE_TLS = os.getenv("REDIS_USE_TLS", "false").lower() == "true"
REDIS_POOL_MAX_CONNECTIONS = int(os.getenv("REDIS_POOL_MAX_CONNECTIONS", "10"))

# Apply log level from env if provided
try:
    logger.setLevel(getattr(logging, LOG_LEVEL))
except Exception:
    logger.setLevel(logging.INFO)

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

# Global Redis client holder (initialized during startup)
redis_client: Optional[redis.Redis] = None

# AI Service Health Monitoring
class AIServiceMonitor:
    """Monitor AI service health and performance"""
    
    def __init__(self):
        self.ai_performance_history = []
        self.last_health_check = None
        self.ai_errors_count = 0
        self.successful_ai_calls = 0
        self._lock = asyncio.Lock()
        
    async def check_ai_services_health(self) -> Dict[str, Any]:
        """Lightweight AI services health check (deployment-friendly)"""
        ai_health = {
            'conversation_engine': AI_CONVERSATION_AVAILABLE,
            'data_processor': AI_DATA_PROCESSOR_AVAILABLE,
            'gemini_api': bool(os.getenv("GEMINI_API_KEY")),
            'redis_cache': False,
            'ai_performance': 'initializing'
        }
        
        performance_metrics = {
            'avg_ai_response_time': 0.0,
            'ai_success_rate': 0.0,
            'total_ai_calls': self.successful_ai_calls + self.ai_errors_count
        }
        
        # Redis connectivity check (fast ping)
        global redis_client
        if redis_client is not None:
            try:
                await redis_client.ping()
                ai_health['redis_cache'] = True
            except Exception as e:
                ai_health['redis_cache'] = False
                logger.warning(f"Redis health check failed: {e}")
        
        # Calculate success rate
        total_calls = self.successful_ai_calls + self.ai_errors_count
        if total_calls > 0:
            performance_metrics['ai_success_rate'] = self.successful_ai_calls / total_calls
        
        # Average response time from history
        successful_history = [h for h in self.ai_performance_history if h.get('success')]
        if successful_history:
            performance_metrics['avg_ai_response_time'] = sum(
                h.get('processing_time_ms', 0) for h in successful_history
            ) / len(successful_history)
        
        # Determine overall AI performance
        if ai_health['conversation_engine'] or ai_health['data_processor']:
            if self.ai_errors_count > self.successful_ai_calls:
                ai_health['ai_performance'] = 'degraded'
            else:
                ai_health['ai_performance'] = 'operational'
        else:
            ai_health['ai_performance'] = 'offline'
        
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

class ConversationStore:
    """Conversation/session store with Redis-backed cache and in-memory fallback."""

    def __init__(self):
        self.local_store: Dict[str, Dict[str, Any]] = {}
        self.redis_hash_key = "janai:active_conversations"
        self._client: Optional[redis.Redis] = None

    def set_client(self, client: redis.Redis):
        self._client = client

    async def warm_from_cache(self):
        """Prime local cache from Redis on startup."""
        if not self._client:
            return
        try:
            cached = await self._client.hgetall(self.redis_hash_key)
            for key, value in cached.items():
                self.local_store[key] = json.loads(value)
            if cached:
                logger.info("Warm-started %d conversations from Redis cache", len(cached))
        except Exception as e:
            logger.warning(f"Failed to warm cache from Redis: {e}")

    async def exists(self, conversation_id: str) -> bool:
        if conversation_id in self.local_store:
            return True
        if self._client:
            try:
                return await self._client.hexists(self.redis_hash_key, conversation_id)
            except Exception as e:
                logger.warning(f"Redis exists check failed: {e}")
        return False

    async def set(self, conversation_id: str, data: Dict[str, Any]):
        self.local_store[conversation_id] = data
        if self._client:
            try:
                await self._client.hset(self.redis_hash_key, conversation_id, json.dumps(data))
            except Exception as e:
                logger.warning(f"Failed to persist conversation to Redis: {e}")

    async def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        if conversation_id in self.local_store:
            return self.local_store[conversation_id]
        if self._client:
            try:
                raw = await self._client.hget(self.redis_hash_key, conversation_id)
                if raw:
                    data = json.loads(raw)
                    self.local_store[conversation_id] = data
                    return data
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        return None

    async def update(self, conversation_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = await self.get(conversation_id)
        if not data:
            return None
        data.update(updates)
        await self.set(conversation_id, data)
        return data

    async def delete(self, conversation_id: str):
        self.local_store.pop(conversation_id, None)
        if self._client:
            try:
                await self._client.hdel(self.redis_hash_key, conversation_id)
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")

    async def get_all(self) -> Dict[str, Dict[str, Any]]:
        if self._client:
            try:
                raw_map = await self._client.hgetall(self.redis_hash_key)
                if raw_map:
                    return {k: json.loads(v) for k, v in raw_map.items()}
            except Exception as e:
                logger.warning(f"Redis get_all failed: {e}")
        return dict(self.local_store)

    async def count(self) -> int:
        all_items = await self.get_all()
        return len(all_items)


conversation_store = ConversationStore()

async def init_redis():
    """Initialize Redis connection using env-driven config."""
    global redis_client
    if not REDIS_HOST or not REDIS_PASSWORD:
        logger.info("Redis configuration not provided - using in-memory cache.")
        return

    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            username=REDIS_USERNAME,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            ssl=REDIS_USE_TLS,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            health_check_interval=30,
            max_connections=REDIS_POOL_MAX_CONNECTIONS,
        )
        await redis_client.ping()
        conversation_store.set_client(redis_client)
        await conversation_store.warm_from_cache()
        logger.info("Connected to Redis cache at %s:%s (db=%s)", REDIS_HOST, REDIS_PORT, REDIS_DB)
    except Exception as e:
        redis_client = None
        logger.warning(f"Redis unavailable, falling back to in-memory cache: {e}")

async def close_redis():
    global redis_client
    if redis_client:
        try:
            await redis_client.close()
        except Exception:
            pass

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application startup and shutdown with AI initialization"""
    logger.info("Starting JanSpandana.AI Enhanced Backend Server...")

    # Initialize Redis cache
    await init_redis()

    # Health checks on startup (lightweight)
    db_health = await check_database_health()
    speech_health = await check_speech_services_health()
    ai_health, ai_performance = await ai_monitor.check_ai_services_health()

    if not db_health['overall_healthy']:
        logger.error("Database health check failed!")
        raise Exception("Database connection failed")

    if not all(speech_health.values()):
        logger.warning("Some speech services may not be fully functional")

    if ai_health['ai_performance'] == 'operational':
        logger.info("AI services operational")
    elif ai_health['ai_performance'] == 'degraded':
        logger.warning("AI services degraded - fallbacks may be used")
    else:
        logger.warning("AI services offline - running in fallback mode")

    logger.info(
        "Startup checks complete: Conversation Engine=%s, Data Processor=%s, Gemini key configured=%s, Redis cache=%s",
        ai_health['conversation_engine'],
        ai_health['data_processor'],
        ai_health['gemini_api'],
        ai_health['redis_cache'],
    )

    yield

    # Shutdown
    await close_redis()
    logger.info(
        "Shutting down JanSpandana.AI Enhanced Backend Server... AI stats - Success: %s, Errors: %s",
        ai_monitor.successful_ai_calls,
        ai_monitor.ai_errors_count,
    )
    logger.info("JanSpandana.AI Enhanced Backend Server shut down cleanly")

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
cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]
if not cors_origins:
    cors_origins = ["http://localhost:8501", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Static files for audio storage
os.makedirs("static/audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for tracking
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

def delete_audio_files(file_paths: List[str]):
    """Best-effort deletion of audio files for a conversation."""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            # Ignore deletion errors to avoid impacting API response
            pass

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
            "ai_api_connectivity": ai_health['gemini_api'],
            "cache_available": ai_health.get('redis_cache', False)
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
            "credentials": speech_health.get('credentials', False),
            "cache": ai_health.get('redis_cache', False),
        }
        
        # Overall status determination
        basic_services_healthy = all(services.values())
        ai_services_healthy = ai_health['ai_performance'] in ['operational']
        
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
        initial_audio_path = None
        if tts_result['success']:
            audio_filename = generate_audio_filename(conversation_id, 'initial_response')
            initial_audio_url = await save_audio_file(tts_result['audio_content'], audio_filename)
            initial_audio_path = os.path.join("static", audio_filename)
        
        # Track active conversation (Redis-backed)
        conversation_data = {
            'started_at': datetime.now(timezone.utc).isoformat(),
            'village_name': request.village_name,
            'user_phone': request.user_phone,
            'metadata': request.metadata,
            'message_count': 0,
            'ai_powered': True,
            'total_ai_processing_time': ai_processing_time,
            'audio_files': [initial_audio_path] if initial_audio_path else []
        }
        await conversation_store.set(conversation_id, conversation_data)
        
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
                'started_at': conversation_data['started_at'],
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
        if not await conversation_store.exists(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        conversation_data = await conversation_store.get(conversation_id)
        if not conversation_data:
            raise HTTPException(status_code=404, detail="Conversation data unavailable")
        
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
            audio_files = conversation_data.get('audio_files', [])
            audio_files.append(os.path.join("static", audio_filename))
            await conversation_store.update(conversation_id, {'audio_files': audio_files})
        
        # Update conversation tracking
        await conversation_store.update(
            conversation_id,
            {
                'message_count': conversation_data.get('message_count', 0) + 1,
                'last_activity': datetime.now(timezone.utc).isoformat(),
                'total_ai_processing_time': conversation_data.get('total_ai_processing_time', 0) + ai_processing_time
            }
        )
        
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
        if not await conversation_store.exists(request.conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        conversation_data = await conversation_store.get(request.conversation_id)
        if not conversation_data:
            raise HTTPException(status_code=404, detail="Conversation data unavailable")
        
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
            audio_files = conversation_data.get('audio_files', [])
            audio_files.append(os.path.join("static", audio_filename))
            await conversation_store.update(request.conversation_id, {'audio_files': audio_files})
        
        # Update tracking
        await conversation_store.update(
            request.conversation_id,
            {
                'message_count': conversation_data.get('message_count', 0) + 1,
                'last_activity': datetime.now(timezone.utc).isoformat(),
                'total_ai_processing_time': conversation_data.get('total_ai_processing_time', 0) + ai_processing_time
            }
        )
        
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
        if not await conversation_store.exists(request.conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        conversation_data = await conversation_store.get(request.conversation_id)
        if not conversation_data:
            raise HTTPException(status_code=404, detail="Conversation data unavailable")
        
        logger.info(f"Ending AI-powered conversation: {request.conversation_id}")
        
        # End AI conversation
        ai_start_time = time.time()
        end_result = await end_conversation_session(request.conversation_id)
        ai_processing_time = (time.time() - ai_start_time) * 1000
        
        # Update tracking data
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
        delete_audio_files(conversation_data.get('audio_files', []))
        await conversation_store.delete(request.conversation_id)
        
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
    if not await conversation_store.exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation_data = await conversation_store.get(conversation_id)
    if not conversation_data:
        raise HTTPException(status_code=404, detail="Conversation data unavailable")
    
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
    conversations = await conversation_store.get_all()
    
    # Calculate AI performance metrics
    total_ai_conversations = len([conv for conv in conversations.values() if conv.get('ai_powered')])
    avg_ai_processing_time = 0.0
    
    if total_ai_conversations > 0:
        total_ai_time = sum(conv.get('total_ai_processing_time', 0) for conv in conversations.values() if conv.get('ai_powered'))
        avg_ai_processing_time = total_ai_time / total_ai_conversations
    
    return {
        'active_conversations': len(conversations),
        'ai_powered_conversations': total_ai_conversations,
        'total_conversations_today': len(conversations),
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
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )
