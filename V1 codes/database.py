"""
JanSpandana.AI Database Connection Manager
Handles Supabase PostgreSQL connections with transaction pooling
Optimized for rural network conditions and high concurrency
"""

import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import asyncio
from contextlib import asynccontextmanager

# Environment and utilities
from dotenv import load_dotenv

# Load environment variables before any reads
load_dotenv()

# Database & ORM imports
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Integer, Boolean, DateTime, Text, DECIMAL, UUID, ARRAY, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, JSONB
import sqlalchemy.dialects.postgresql as pg

# Import required libraries for Supabase client and asyncpg
from supabase import create_client, Client  # Supabase Python client
import asyncpg  # PostgreSQL async driver

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    logging.error(f"Failed to initialize Supabase client: {e}")

async def connect_to_db():
    try:
        return await asyncpg.create_pool(
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            min_size=1,
            max_size=20
        )
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Models Base
Base = declarative_base()

class DatabaseManager:
    """
    Centralized database connection manager for JanSpandana.AI
    Handles both synchronous and asynchronous connections to Supabase
    """
    
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        self.database_url = os.getenv('DATABASE_URL')
        
        # Initialize connections
        self.supabase_client: Optional[Client] = None
        self.sync_engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize all database connections with optimal settings for rural networks"""
        try:
            # Supabase client for real-time features
            self.supabase_client = create_client(self.supabase_url, self.supabase_key)
            
            # Synchronous SQLAlchemy engine with connection pooling
            self.sync_engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=20,  # Optimized for village concurrent users
                max_overflow=30,
                pool_timeout=30,
                pool_recycle=1800,  # 30 minutes - good for rural connectivity
                pool_pre_ping=True,  # Validate connections before use
                echo=False  # Set to True for debugging
            )
            
            # Session factory for sync operations
            self.session_factory = sessionmaker(bind=self.sync_engine)
            
            # Async engine for high-performance operations
            async_url = self.database_url.replace('postgresql://', 'postgresql+asyncpg://')
            self.async_engine = create_async_engine(
                async_url,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True,
                echo=False,
                connect_args={
                    # Disable statement cache to avoid pgbouncer transaction mode conflicts
                    "statement_cache_size": 0
                }
            )
            
            # Async session factory
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {str(e)}")
            raise
    
    @asynccontextmanager
    async def get_async_session(self):
        """Get async database session with proper cleanup"""
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {str(e)}")
                raise
            finally:
                await session.close()
    
    def get_sync_session(self) -> Session:
        """Get synchronous database session"""
        return self.session_factory()
    
    async def test_connection(self) -> Dict[str, bool]:
        """Test all database connections"""
        results = {
            'supabase_client': False,
            'sync_engine': False,
            'async_engine': False
        }
        
        # Test Supabase client
        try:
            response = self.supabase_client.table('users').select('*').limit(1).execute()
            results['supabase_client'] = True
            logger.info("Supabase client connection: OK")
        except Exception as e:
            logger.error(f"Supabase client error: {str(e)}")
        
        # Test sync engine
        try:
            with self.sync_engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                results['sync_engine'] = True
                logger.info("Sync engine connection: OK")
        except Exception as e:
            logger.error(f"Sync engine error: {str(e)}")
        
        # Test async engine
        try:
            async with self.async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
                results['async_engine'] = True
                logger.info("Async engine connection: OK")
        except Exception as e:
            logger.error(f"Async engine error: {str(e)}")
        
        return results

# SQLAlchemy Models for type-safe database operations
class User(Base):
    __tablename__ = 'users'
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False)
    gender = Column(String(10))
    phone_number = Column(String(15), unique=True)
    village_name = Column(String(255), nullable=False)
    mandal_name = Column(String(255))
    district_name = Column(String(255))
    age_group = Column(String(20))
    education_level = Column(String(50))
    occupation = Column(String(100))
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)

class Grievance(Base):
    __tablename__ = 'grievances'
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True)
    user_id = Column(PostgresUUID(as_uuid=True))
    
    # Audio & Transcript
    audio_file_url = Column(Text)
    audio_duration_seconds = Column(Integer)
    audio_file_size_mb = Column(DECIMAL(5,2))
    transcript_telugu = Column(Text)
    transcript_english = Column(Text)
    
    # AI Analysis
    primary_sector = Column(String(100), nullable=False)
    secondary_sectors = Column(ARRAY(Text))
    complaint_category = Column(String(255))
    urgency_level = Column(String(20))
    sentiment_score = Column(DECIMAL(3,2))
    emotion_detected = Column(String(50))
    
    # Structured Data
    problem_summary_telugu = Column(Text)
    problem_summary_english = Column(Text)
    location_mentioned = Column(String(255))
    government_scheme_mentioned = Column(String(255))
    officials_mentioned = Column(ARRAY(Text))
    amount_involved = Column(DECIMAL(12,2))
    
    # Classification
    department_assigned = Column(String(255))
    sub_department = Column(String(255))
    complexity_score = Column(Integer)
    requires_immediate_action = Column(Boolean, default=False)
    
    # Metadata
    conversation_duration_minutes = Column(Integer)
    conversation_quality_score = Column(Integer)
    language_confidence_score = Column(DECIMAL(3,2))
    total_questions_asked = Column(Integer)
    user_satisfaction_rating = Column(Integer)
    
    # Status
    status = Column(String(50), default='received')
    priority_score = Column(Integer, default=1)
    follow_up_required = Column(Boolean, default=True)
    resolution_deadline = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    resolved_at = Column(DateTime(timezone=True))

class ConversationLog(Base):
    __tablename__ = 'conversation_logs'
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True)
    grievance_id = Column(PostgresUUID(as_uuid=True))
    question_sequence = Column(Integer)
    question_asked_telugu = Column(Text)
    question_asked_english = Column(Text)
    user_response_telugu = Column(Text)
    user_response_english = Column(Text)
    response_sentiment = Column(DECIMAL(3,2))
    response_length_words = Column(Integer)
    ai_confidence_score = Column(DECIMAL(3,2))
    timestamp = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))

class ProblemCategory(Base):
    __tablename__ = 'problem_categories'
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True)
    category_name_english = Column(String(255), nullable=False)
    category_name_telugu = Column(String(255), nullable=False)
    parent_category_id = Column(PostgresUUID(as_uuid=True))
    department = Column(String(255))
    typical_resolution_days = Column(Integer)
    priority_weight = Column(Integer, default=1)
    keywords_telugu = Column(ARRAY(Text))
    keywords_english = Column(ARRAY(Text))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))

class GovScheme(Base):
    __tablename__ = 'government_schemes'
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True)
    scheme_name_english = Column(String(255), nullable=False)
    scheme_name_telugu = Column(String(255), nullable=False)
    department = Column(String(255))
    scheme_type = Column(String(100))
    eligibility_criteria = Column(Text)
    benefits_amount = Column(DECIMAL(12,2))
    application_process = Column(Text)
    contact_details = Column(Text)
    is_active = Column(Boolean, default=True)
    launched_date = Column(DateTime(timezone=True))
    keywords_telugu = Column(ARRAY(Text))
    keywords_english = Column(ARRAY(Text))

# Database utility functions
class DatabaseOperations:
    """Common database operations for JanSpandana.AI"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    async def create_user(self, user_data: Dict[str, Any]) -> str:
        """Create new user and return user ID"""
        try:
            response = self.db.supabase_client.table('users').insert(user_data).execute()
            return response.data[0]['id']
        except Exception as e:
            logger.error(f"Failed to create user: {str(e)}")
            raise
    
    async def create_grievance(self, grievance_data: Dict[str, Any]) -> str:
        """Create new grievance and return grievance ID"""
        try:
            response = self.db.supabase_client.table('grievances').insert(grievance_data).execute()
            return response.data[0]['id']
        except Exception as e:
            logger.error(f"Failed to create grievance: {str(e)}")
            raise
    
    async def log_conversation_step(self, conversation_data: Dict[str, Any]) -> bool:
        """Log individual conversation step"""
        try:
            self.db.supabase_client.table('conversation_logs').insert(conversation_data).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to log conversation: {str(e)}")
            return False
    
    async def get_problem_categories(self) -> List[Dict[str, Any]]:
        """Get all active problem categories"""
        try:
            response = self.db.supabase_client.table('problem_categories').select('*').eq('is_active', True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get categories: {str(e)}")
            return []
    
    async def get_government_schemes(self) -> List[Dict[str, Any]]:
        """Get all active government schemes"""
        try:
            response = self.db.supabase_client.table('government_schemes').select('*').eq('is_active', True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get schemes: {str(e)}")
            return []
    
    async def update_grievance_status(self, grievance_id: str, status: str, additional_data: Dict = None) -> bool:
        """Update grievance status and additional fields"""
        try:
            update_data = {'status': status, 'updated_at': datetime.now(timezone.utc).isoformat()}
            if additional_data:
                update_data.update(additional_data)
            
            self.db.supabase_client.table('grievances').update(update_data).eq('id', grievance_id).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to update grievance status: {str(e)}")
            return False
    
    async def get_village_analytics(self, village_name: str) -> Dict[str, Any]:
        """Get analytics for specific village"""
        try:
            # Get grievance counts by sector
            response = self.db.supabase_client.table('grievances').select('primary_sector, status, urgency_level').eq('users.village_name', village_name).execute()
            
            analytics = {
                'total_grievances': len(response.data),
                'by_sector': {},
                'by_status': {},
                'by_urgency': {}
            }
            
            for grievance in response.data:
                # Count by sector
                sector = grievance.get('primary_sector', 'Unknown')
                analytics['by_sector'][sector] = analytics['by_sector'].get(sector, 0) + 1
                
                # Count by status
                status = grievance.get('status', 'Unknown')
                analytics['by_status'][status] = analytics['by_status'].get(status, 0) + 1
                
                # Count by urgency
                urgency = grievance.get('urgency_level', 'Unknown')
                analytics['by_urgency'][urgency] = analytics['by_urgency'].get(urgency, 0) + 1
            
            return analytics
        except Exception as e:
            logger.error(f"Failed to get village analytics: {str(e)}")
            return {}

# Global database manager instance
db_manager = DatabaseManager()
db_ops = DatabaseOperations(db_manager)

# Utility functions for easy import
async def get_db_session():
    """Get async database session - use in FastAPI dependency injection"""
    async with db_manager.get_async_session() as session:
        yield session

def get_sync_db():
    """Get sync database session"""
    return db_manager.get_sync_session()

def get_supabase_client():
    """Get Supabase client"""
    return db_manager.supabase_client

# Health check function
async def check_database_health() -> Dict[str, Any]:
    """Comprehensive database health check"""
    health_status = await db_manager.test_connection()
    
    # Test table access
    try:
        response = db_manager.supabase_client.table('users').select('count').execute()
        health_status['table_access'] = True
    except Exception as e:
        health_status['table_access'] = False
        logger.error(f"Table access error: {str(e)}")
    
    # Overall health
    health_status['overall_healthy'] = all([
        health_status['supabase_client'],
        health_status['sync_engine'],
        health_status['async_engine'],
        health_status['table_access']
    ])
    
    return health_status

if __name__ == "__main__":
    # Test database connections
    import asyncio
    
    async def test_setup():
        print("Testing JanSpandana.AI Database Setup...")
        health = await check_database_health()
        
        print(f"Database Health Status:")
        for component, status in health.items():
            print(f"  {component}: {'✅' if status else '❌'}")
        
        if health['overall_healthy']:
            print("\n🎉 Database setup successful! Ready for JanSpandana.AI deployment.")
        else:
            print("\n⚠️ Database setup issues detected. Check logs for details.")
    
    asyncio.run(test_setup())
