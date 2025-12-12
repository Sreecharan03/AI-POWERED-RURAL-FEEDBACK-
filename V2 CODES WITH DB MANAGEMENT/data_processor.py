"""
JanSpandana.AI - Data Processor
Clean architecture for converting conversations to database records
Handles user creation, grievance processing, and conversation logging
"""

import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import json

# Database imports
from database import db_manager, db_ops

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationDataProcessor:
    """
    Clean processor for converting AI conversations to structured database records
    Separates data processing logic from conversation logic
    """
    
    def __init__(self):
        self.db_ops = db_ops
        
        # Data validation rules
        self.required_fields = ['user_name', 'village_name', 'identified_sector']
        self.valid_sectors = ['వైద్య సేవలు', 'మౌలిక వసతులు', 'విద్యా సేవలు', 'సంక్షేమ పథకాలు']
        self.valid_urgency_levels = ['low', 'medium', 'high', 'critical']
        
        # Sector to department mapping
        self.sector_department_mapping = {
            'వైద్య సేవలు': 'Health Department',
            'మౌలిక వసతులు': 'Infrastructure Department', 
            'విద్యా సేవలు': 'Education Department',
            'సంక్షేమ పథకాలు': 'Welfare Department'
        }
    
    async def process_complete_conversation(self, conversation_state: Dict[str, Any], 
                                          conversation_id: str) -> Dict[str, Any]:
        """
        Main method: Convert complete conversation to database records
        """
        try:
            logger.info(f"Processing conversation for database: {conversation_id}")
            
            # Validate conversation data
            validation_result = self._validate_conversation_data(conversation_state)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': f"Validation failed: {validation_result['errors']}",
                    'step_failed': 'validation'
                }
            
            # Step 1: Create or find user
            user_result = await self._create_or_find_user(conversation_state)
            if not user_result['success']:
                return {
                    'success': False,
                    'error': user_result['error'],
                    'step_failed': 'user_creation'
                }
            
            user_id = user_result['user_id']
            logger.info(f"User processed: {user_id}")
            
            # Step 2: Create grievance record
            grievance_result = await self._create_grievance_record(
                conversation_state, user_id, conversation_id
            )
            if not grievance_result['success']:
                return {
                    'success': False,
                    'error': grievance_result['error'],
                    'step_failed': 'grievance_creation'
                }
            
            grievance_id = grievance_result['grievance_id']
            logger.info(f"Grievance created: {grievance_id}")
            
            # Step 3: Save conversation logs
            logs_result = await self._save_conversation_logs(
                conversation_state, grievance_id
            )
            if not logs_result['success']:
                logger.warning(f"Conversation logs failed: {logs_result['error']}")
                # Don't fail the whole process for log issues
            
            # Step 4: Generate summary statistics
            summary_stats = self._generate_conversation_summary(conversation_state)
            
            return {
                'success': True,
                'user_id': user_id,
                'grievance_id': grievance_id,
                'conversation_logs_saved': logs_result['success'],
                'summary_stats': summary_stats,
                'database_operations': {
                    'user_operation': user_result.get('operation', 'unknown'),
                    'grievance_created': True,
                    'logs_count': len(conversation_state.get('conversation_log', []))
                }
            }
            
        except Exception as e:
            logger.error(f"Database processing failed: {str(e)}")
            return {
                'success': False,
                'error': f"Processing error: {str(e)}",
                'step_failed': 'unknown'
            }
    
    def _validate_conversation_data(self, conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that conversation contains required data for database storage
        """
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if not conversation_state.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate sector
        sector = conversation_state.get('identified_sector')
        if sector and sector not in self.valid_sectors:
            errors.append(f"Invalid sector: {sector}")
        
        # Check conversation log exists
        if not conversation_state.get('conversation_log'):
            errors.append("No conversation log found")
        
        # Validate minimum conversation length
        if len(conversation_state.get('conversation_log', [])) < 3:
            errors.append("Conversation too short for storage")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    async def _create_or_find_user(self, conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new user or find existing user by phone/name combination
        """
        try:
            # Extract user data
            user_data = {
                'id': str(uuid.uuid4()),
                'name': conversation_state['user_name'].strip(),
                'gender': conversation_state.get('user_gender', 'unknown'),
                'village_name': conversation_state['village_name'].strip(),
                'phone_number': conversation_state.get('user_phone'),  # May be None
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'is_active': True
            }
            
            # Try to find existing user first (if phone provided)
            if user_data['phone_number']:
                existing_user = await self._find_user_by_phone(user_data['phone_number'])
                if existing_user:
                    logger.info(f"Found existing user: {existing_user['id']}")
                    return {
                        'success': True,
                        'user_id': existing_user['id'],
                        'operation': 'found_existing'
                    }
            
            # Create new user
            user_id = await self.db_ops.create_user(user_data)
            
            return {
                'success': True,
                'user_id': user_id,
                'operation': 'created_new'
            }
            
        except Exception as e:
            logger.error(f"User creation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _find_user_by_phone(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """
        Find existing user by phone number
        """
        try:
            response = self.db_ops.db.supabase_client.table('users').select('*').eq('phone_number', phone_number).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
            
        except Exception as e:
            logger.warning(f"User lookup failed: {str(e)}")
            return None
    
    async def _create_grievance_record(self, conversation_state: Dict[str, Any], 
                                     user_id: str, conversation_id: str) -> Dict[str, Any]:
        """
        Create structured grievance record from conversation data
        """
        try:
            # Analyze conversation for key insights
            conversation_analysis = self._analyze_conversation_content(conversation_state)
            
            # Build grievance data
            grievance_data = {
                'id': str(uuid.uuid4()),
                'user_id': user_id,
                
                # Audio & Transcript (placeholder - would need actual audio URL)
                'audio_file_url': f"/static/audio/{conversation_id}_complete.mp3",
                'audio_duration_seconds': conversation_state.get('total_duration_seconds', 0),
                'transcript_telugu': self._extract_user_responses_telugu(conversation_state),
                'transcript_english': conversation_analysis['english_summary'],
                
                # AI Analysis
                'primary_sector': conversation_state['identified_sector'],
                'secondary_sectors': conversation_analysis['secondary_sectors'],
                'complaint_category': conversation_analysis['complaint_category'],
                'urgency_level': conversation_analysis['urgency_level'],
                'sentiment_score': conversation_analysis['sentiment_score'],
                'emotion_detected': conversation_analysis['emotion_detected'],
                
                # Structured Data
                'problem_summary_telugu': conversation_analysis['problem_summary_telugu'],
                'problem_summary_english': conversation_analysis['problem_summary_english'],
                'location_mentioned': conversation_state['village_name'],
                'government_scheme_mentioned': conversation_analysis['schemes_mentioned'],
                'officials_mentioned': conversation_analysis['officials_mentioned'],
                
                # Classification
                'department_assigned': self.sector_department_mapping.get(
                    conversation_state['identified_sector'], 'General'
                ),
                'complexity_score': conversation_analysis['complexity_score'],
                'requires_immediate_action': conversation_analysis['urgency_level'] in ['high', 'critical'],
                
                # Metadata
                'conversation_duration_minutes': len(conversation_state.get('conversation_log', [])) * 2,  # Estimated
                'conversation_quality_score': conversation_analysis['quality_score'],
                'total_questions_asked': conversation_state.get('question_count', 0),
                
                # Status
                'status': 'received',
                'priority_score': self._calculate_priority_score(conversation_analysis),
                'follow_up_required': True,
                
                # Timestamps
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Save to database
            grievance_id = await self.db_ops.create_grievance(grievance_data)
            
            return {
                'success': True,
                'grievance_id': grievance_id
            }
            
        except Exception as e:
            logger.error(f"Grievance creation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_conversation_content(self, conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze conversation content to extract insights for database
        """
        conversation_log = conversation_state.get('conversation_log', [])
        collected_issues = conversation_state.get('collected_issues', [])
        
        # Extract all user responses
        user_responses = []
        for entry in conversation_log:
            if 'user_input' in entry:
                user_responses.append(entry['user_input'])
        
        # Simple analysis (can be enhanced with NLP later)
        combined_text = ' '.join(user_responses).lower()
        
        # Sentiment analysis (basic)
        negative_words = ['చెడు', 'లేదు', 'సమస్య', 'కష్టం', 'బాధ', 'దూకుడు']
        positive_words = ['మంచి', 'బాగుంది', 'సంతోషం', 'సరైన']
        
        negative_count = sum(1 for word in negative_words if word in combined_text)
        positive_count = sum(1 for word in positive_words if word in combined_text)
        
        # Calculate sentiment score (-1 to 1)
        total_sentiment_words = negative_count + positive_count
        if total_sentiment_words > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
        else:
            sentiment_score = 0.0
        
        # Urgency detection
        urgent_keywords = ['అత్యవసర', 'తక్షణం', 'చాలా', 'మరణం', 'బాధ']
        urgency_level = 'high' if any(word in combined_text for word in urgent_keywords) else 'medium'
        
        return {
            'english_summary': f"Conversation about {conversation_state.get('identified_sector', 'general')} services in {conversation_state.get('village_name', 'unknown village')}",
            'secondary_sectors': [],  # Could be enhanced to detect multiple sectors
            'complaint_category': f"{conversation_state.get('identified_sector', 'General')} Issues",
            'urgency_level': urgency_level,
            'sentiment_score': round(sentiment_score, 2),
            'emotion_detected': 'concerned' if sentiment_score < 0 else 'neutral',
            'problem_summary_telugu': ' '.join(user_responses[:2]) if user_responses else '',
            'problem_summary_english': f"User reported issues with {conversation_state.get('identified_sector', 'services')}",
            'schemes_mentioned': None,  # Could be enhanced to detect scheme mentions
            'officials_mentioned': [],
            'complexity_score': min(len(user_responses), 5),  # 1-5 scale
            # Keep quality score within expected 1-5 range to satisfy DB constraint
            'quality_score': max(1, min(len(conversation_log), 5))
        }
    
    def _extract_user_responses_telugu(self, conversation_state: Dict[str, Any]) -> str:
        """
        Extract all user responses in Telugu for transcript
        """
        conversation_log = conversation_state.get('conversation_log', [])
        user_responses = []
        
        for entry in conversation_log:
            if 'user_input' in entry:
                user_responses.append(entry['user_input'])
        
        return '\n'.join(user_responses)
    
    def _calculate_priority_score(self, analysis: Dict[str, Any]) -> int:
        """
        Calculate priority score (1-10) based on conversation analysis
        """
        score = 5  # Base score
        
        # Adjust based on urgency
        if analysis['urgency_level'] == 'critical':
            score += 3
        elif analysis['urgency_level'] == 'high':
            score += 2
        elif analysis['urgency_level'] == 'medium':
            score += 1
        
        # Adjust based on sentiment (negative = higher priority)
        if analysis['sentiment_score'] < -0.5:
            score += 2
        elif analysis['sentiment_score'] < 0:
            score += 1
        
        # Adjust based on complexity
        if analysis['complexity_score'] >= 4:
            score += 1
        
        return min(max(score, 1), 10)  # Keep in 1-10 range
    
    async def _save_conversation_logs(self, conversation_state: Dict[str, Any], 
                                    grievance_id: str) -> Dict[str, Any]:
        """
        Save detailed conversation logs for review
        """
        try:
            conversation_log = conversation_state.get('conversation_log', [])
            logs_saved = 0
            
            for i, entry in enumerate(conversation_log):
                if 'user_input' in entry and 'ai_response' in entry:
                    log_data = {
                        'id': str(uuid.uuid4()),
                        'grievance_id': grievance_id,
                        'question_sequence': i + 1,
                        'question_asked_telugu': entry.get('ai_response', ''),
                        'question_asked_english': f"AI question #{i + 1}",
                        'user_response_telugu': entry['user_input'],
                        'user_response_english': entry['user_input'],  # Could add translation
                        'response_sentiment': 0.0,  # Could add sentiment analysis
                        'response_length_words': len(entry['user_input'].split()),
                        'ai_confidence_score': entry.get('ai_confidence_score', 0.8),
                        'timestamp': entry.get('timestamp', datetime.now(timezone.utc).isoformat())
                    }
                    
                    # Save individual log entry
                    saved = await self.db_ops.log_conversation_step(log_data)
                    if saved:
                        logs_saved += 1
            
            return {
                'success': True,
                'logs_saved': logs_saved,
                'total_entries': len(conversation_log)
            }
            
        except Exception as e:
            logger.error(f"Conversation logs save failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'logs_saved': 0
            }
    
    def _generate_conversation_summary(self, conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary statistics for the conversation
        """
        conversation_log = conversation_state.get('conversation_log', [])
        
        return {
            'total_exchanges': len(conversation_log),
            'user_name': conversation_state.get('user_name', 'Unknown'),
            'village_name': conversation_state.get('village_name', 'Unknown'),
            'identified_sector': conversation_state.get('identified_sector', 'Unknown'),
            'conversation_stages_completed': conversation_state.get('stage', 'unknown'),
            'total_questions_asked': conversation_state.get('question_count', 0),
            'start_time': conversation_state.get('start_time'),
            'processing_timestamp': datetime.now(timezone.utc).isoformat()
        }

# Global data processor instance
conversation_processor = ConversationDataProcessor()

# Utility function for easy import
async def save_conversation_to_database(conversation_state: Dict[str, Any], 
                                       conversation_id: str) -> Dict[str, Any]:
    """
    Main function: Save complete conversation to database
    """
    return await conversation_processor.process_complete_conversation(
        conversation_state, conversation_id
    )

# Test function
async def test_data_processor():
    """
    Test the data processor with sample conversation data
    """
    sample_conversation = {
        'conversation_id': 'test_conv_001',
        'user_name': 'రాము',
        'user_gender': 'male',
        'village_name': 'రామారావుపేట',
        'identified_sector': 'వైద్య సేవలు',
        'question_count': 4,
        'start_time': datetime.now(timezone.utc).isoformat(),
        'conversation_log': [
            {
                'user_input': 'నా పేరు రాము',
                'ai_response': 'మీ గ్రామం ఏది?',
                'stage': 'name_collection',
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            {
                'user_input': 'రామారావుపేట',
                'ai_response': 'మీకు ఏ విషయంలో సమస్య ఉంది?',
                'stage': 'sector_identification',
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            {
                'user_input': 'మా గ్రామంలో ఆసుపత్రిలో వైద్యుడు రాడు',
                'ai_response': 'మరింత వివరంగా చెప్పండి',
                'stage': 'detailed_inquiry',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        ],
        'collected_issues': [
            {
                'content': 'మా గ్రామంలో ఆసుపత్రిలో వైద్యుడు రాడు',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'stage': 'detailed_inquiry'
            }
        ]
    }
    
    result = await save_conversation_to_database(sample_conversation, 'test_conv_001')
    print("Test Result:", json.dumps(result, indent=2, ensure_ascii=False))
    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_data_processor())
