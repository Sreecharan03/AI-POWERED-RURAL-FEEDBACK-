"""
JanSpandana.AI - AI-Powered Data Processor
Revolutionary approach: Replace ALL keyword matching with AI intelligence
Converts AI conversations to structured database records with smart analysis

CORE PHILOSOPHY:
- AI understands sentiment from context, not word counting
- AI detects urgency based on meaning, not keywords  
- AI categorizes problems intelligently
- AI generates natural summaries in Telugu and English
- AI extracts government schemes and officials mentioned

USAGE:
This replaces the rigid data_processor.py with true AI intelligence.
All analysis decisions are made by AI, not hardcoded rules.
"""

import uuid
import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone

# AI Processing
import google.generativeai as genai

# Database imports
from database import db_manager, db_ops

# Environment
from dotenv import load_dotenv
import os

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIConversationDataProcessor:
    """
    Revolutionary AI-Powered Data Processor
    Zero keyword matching - Pure AI intelligence for conversation analysis
    """
    
    def __init__(self):
        self.db_ops = db_ops
        
        # Initialize Gemini AI for analysis
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not found for AI analysis")
        
        genai.configure(api_key=self.api_key)
        self.ai_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Data validation rules (unchanged)
        self.required_fields = ['user_name', 'village_name', 'identified_sector']
        self.valid_sectors = ['వైద్య సేవలు', 'మౌలిక వసతులు', 'విద్యా సేవలు', 'సంక్షేమ పథకాలు']
        self.valid_urgency_levels = ['low', 'medium', 'high', 'critical']
        
        # Sector to department mapping (unchanged)
        self.sector_department_mapping = {
            'వైద్య సేవలు': 'Health Department',
            'మౌలిక వసతులు': 'Infrastructure Department', 
            'విద్యా సేవలు': 'Education Department',
            'సంక్షేమ పథకాలు': 'Welfare Department'
        }
        
        # AI Knowledge Context for better analysis
        self.ai_analysis_context = {
            "ap_schemes": {
                "YSR రైతు భరోసా": "agriculture support scheme",
                "అమ్మ ఒడి": "education support for children", 
                "YSR పెన్షను కాంక": "pension scheme for elderly/disabled",
                "జగన్నాథ అన్న కంటీరు": "free rice scheme",
                "ఆరోగ్యశ్రీ": "health insurance scheme"
            },
            "urgency_indicators": "life-threatening situations, complete service failure, affecting many people, immediate danger",
            "sentiment_context": "rural communication patterns, cultural expression of distress in Telugu",
            "problem_categories": "infrastructure failures, service gaps, bureaucratic issues, accessibility problems"
        }
    
    async def process_complete_conversation(self, conversation_state: Dict[str, Any], 
                                          conversation_id: str) -> Dict[str, Any]:
        """
        Main method: Convert complete conversation to database records with AI analysis
        """
        try:
            logger.info(f"AI processing conversation for database: {conversation_id}")
            
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
            
            # Step 2: AI-powered conversation analysis
            ai_analysis = await self._ai_analyze_complete_conversation(conversation_state)
            
            # Step 3: Create grievance record with AI insights
            grievance_result = await self._create_ai_enhanced_grievance_record(
                conversation_state, user_id, conversation_id, ai_analysis
            )
            if not grievance_result['success']:
                return {
                    'success': False,
                    'error': grievance_result['error'],
                    'step_failed': 'grievance_creation'
                }
            
            grievance_id = grievance_result['grievance_id']
            logger.info(f"AI-enhanced grievance created: {grievance_id}")
            
            # Step 4: Save conversation logs with AI insights
            logs_result = await self._save_ai_enhanced_conversation_logs(
                conversation_state, grievance_id, ai_analysis, conversation_id
            )
            if not logs_result['success']:
                logger.warning(f"Conversation logs failed: {logs_result['error']}")
                # Don't fail the whole process for log issues
            
            # Step 5: Generate AI-powered summary statistics
            summary_stats = await self._generate_ai_conversation_summary(conversation_state, ai_analysis)
            
            return {
                'success': True,
                'user_id': user_id,
                'grievance_id': grievance_id,
                'conversation_logs_saved': logs_result['success'],
                'ai_analysis': ai_analysis,
                'summary_stats': summary_stats,
                'database_operations': {
                    'user_operation': user_result.get('operation', 'unknown'),
                    'grievance_created': True,
                    'ai_enhanced': True,
                    'logs_count': len(conversation_state.get('conversation_log', []))
                }
            }
            
        except Exception as e:
            logger.error(f"AI database processing failed: {str(e)}")
            return {
                'success': False,
                'error': f"AI processing error: {str(e)}",
                'step_failed': 'unknown'
            }
    
    async def _ai_analyze_complete_conversation(self, conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI-powered comprehensive conversation analysis
        Replaces ALL keyword matching with intelligent understanding
        """
        try:
            # Extract conversation content
            conversation_log = conversation_state.get('conversation_log', [])
            user_responses = []
            full_conversation = []
            
            for entry in conversation_log:
                if 'user_input' in entry:
                    user_responses.append(entry['user_input'])
                    full_conversation.append(f"User: {entry['user_input']}")
                if 'ai_response' in entry:
                    full_conversation.append(f"AI: {entry['ai_response']}")
            
            combined_user_text = ' '.join(user_responses)
            full_conversation_text = '\n'.join(full_conversation)
            
            # AI Analysis Prompt
            analysis_prompt = f"""
            You are analyzing a citizen grievance conversation from rural Andhra Pradesh for JanSpandana.AI.
            
            CONVERSATION CONTEXT:
            - User: {conversation_state.get('user_name', 'Unknown')}
            - Village: {conversation_state.get('village_name', 'Unknown')}
            - Sector: {conversation_state.get('identified_sector', 'Unknown')}
            
            FULL CONVERSATION:
            {full_conversation_text}
            
            USER RESPONSES ONLY:
            {combined_user_text}
            
            ANALYSIS TASKS:
            Provide intelligent analysis considering rural Telugu communication patterns and AP government context.
            
            RESPOND WITH JSON:
            {{
                "sentiment_analysis": {{
                    "overall_sentiment": "positive/neutral/negative/mixed",
                    "sentiment_score": -1.0 to 1.0,
                    "emotion_detected": "concerned/frustrated/hopeful/satisfied/angry",
                    "confidence": 0.0 to 1.0,
                    "reasoning": "why this sentiment was determined"
                }},
                "urgency_assessment": {{
                    "urgency_level": "low/medium/high/critical",
                    "requires_immediate_action": true/false,
                    "urgency_reasoning": "factors indicating urgency level",
                    "estimated_impact": "individual/community/district"
                }},
                "problem_categorization": {{
                    "primary_problem_type": "service_gap/infrastructure_failure/staff_issues/accessibility/quality",
                    "specific_issues": ["list of specific problems mentioned"],
                    "root_cause_analysis": "likely underlying causes",
                    "complexity_score": 1 to 5
                }},
                "content_extraction": {{
                    "key_complaints": ["main issues in user's words"],
                    "location_details": "specific places/areas mentioned",
                    "people_affected": "who is impacted by this issue",
                    "duration_mentioned": "how long this has been a problem"
                }},
                "government_context": {{
                    "schemes_mentioned": ["any AP government schemes referenced"],
                    "officials_mentioned": ["any government officials mentioned"],
                    "departments_involved": ["which departments should handle this"],
                    "policy_relevance": "how this relates to government policies"
                }},
                "summaries": {{
                    "problem_summary_telugu": "concise problem description in Telugu",
                    "problem_summary_english": "concise problem description in English",
                    "action_items": ["what actions are needed to resolve this"],
                    "follow_up_required": true/false
                }},
                "quality_metrics": {{
                    "conversation_completeness": "incomplete/basic/detailed/comprehensive",
                    "information_quality": 1 to 5,
                    "actionability": "low/medium/high",
                    "priority_score": 1 to 10
                }}
            }}
            
            IMPORTANT:
            - Consider cultural context of rural AP communication
            - Understand indirect expressions of problems in Telugu
            - Recognize government scheme names and context
            - Account for politeness affecting directness of complaints
            - Consider community vs individual impact
            """
            
            # Get AI analysis
            response = await asyncio.to_thread(self.ai_model.generate_content, analysis_prompt)
            
            # Parse AI response
            try:
                start = response.text.find('{')
                end = response.text.rfind('}') + 1
                if start != -1 and end > start:
                    ai_analysis = json.loads(response.text[start:end])
                    
                    # Validate and enrich analysis
                    ai_analysis = self._validate_and_enrich_analysis(ai_analysis)
                    
                    logger.info("AI conversation analysis completed successfully")
                    return ai_analysis
                else:
                    raise ValueError("No valid JSON found in AI response")
                    
            except json.JSONDecodeError as e:
                logger.error(f"AI analysis JSON parse failed: {e}")
                return self._fallback_conversation_analysis(conversation_state)
                
        except Exception as e:
            logger.error(f"AI conversation analysis failed: {e}")
            return self._fallback_conversation_analysis(conversation_state)
    
    def _validate_and_enrich_analysis(self, ai_analysis: Dict) -> Dict:
        """
        Validate AI analysis response and add calculated fields
        """
        # Ensure all required sections exist
        default_structure = {
            "sentiment_analysis": {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "emotion_detected": "neutral",
                "confidence": 0.5
            },
            "urgency_assessment": {
                "urgency_level": "medium",
                "requires_immediate_action": False
            },
            "problem_categorization": {
                "primary_problem_type": "service_gap",
                "complexity_score": 3
            },
            "quality_metrics": {
                "priority_score": 5,
                "actionability": "medium"
            }
        }
        
        # Fill missing sections with defaults
        for section, defaults in default_structure.items():
            if section not in ai_analysis:
                ai_analysis[section] = defaults
            else:
                for key, default_value in defaults.items():
                    if key not in ai_analysis[section]:
                        ai_analysis[section][key] = default_value
        
        # Add calculated enrichments
        ai_analysis['analysis_metadata'] = {
            'analyzed_at': datetime.now(timezone.utc).isoformat(),
            'ai_model_used': 'gemini-2.5-flash',
            'analysis_version': '2.0_ai_powered',
            'processing_successful': True
        }
        
        return ai_analysis
    
    def _fallback_conversation_analysis(self, conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback analysis when AI fails - much simpler than before but still functional
        """
        conversation_log = conversation_state.get('conversation_log', [])
        
        # Simple metrics
        total_responses = len([entry for entry in conversation_log if 'user_input' in entry])
        avg_response_length = sum(len(entry.get('user_input', '').split()) 
                                for entry in conversation_log if 'user_input' in entry) / max(total_responses, 1)
        
        return {
            "sentiment_analysis": {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "emotion_detected": "concerned",
                "confidence": 0.3,
                "reasoning": "Fallback analysis - AI processing failed"
            },
            "urgency_assessment": {
                "urgency_level": "medium" if total_responses > 3 else "low",
                "requires_immediate_action": False,
                "urgency_reasoning": "Based on conversation length"
            },
            "problem_categorization": {
                "primary_problem_type": "service_gap",
                "complexity_score": min(max(total_responses, 1), 5)
            },
            "summaries": {
                "problem_summary_telugu": conversation_state.get('identified_sector', 'సేవల') + " సమస్య",
                "problem_summary_english": f"Issue with {conversation_state.get('identified_sector', 'services')}",
                "follow_up_required": True
            },
            "quality_metrics": {
                "priority_score": 5,
                "information_quality": min(max(int(avg_response_length / 5), 1), 5),
                "actionability": "medium"
            },
            "analysis_metadata": {
                'analyzed_at': datetime.now(timezone.utc).isoformat(),
                'ai_model_used': 'fallback_simple',
                'analysis_version': '2.0_fallback',
                'processing_successful': False
            }
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
        if len(conversation_state.get('conversation_log', [])) < 2:
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
    
    async def _create_ai_enhanced_grievance_record(self, conversation_state: Dict[str, Any], 
                                                  user_id: str, conversation_id: str,
                                                  ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create AI-enhanced grievance record with intelligent analysis
        """
        try:
            # Build grievance data with AI insights
            grievance_data = {
                'id': str(uuid.uuid4()),
                'user_id': user_id,
                
                # Audio & Transcript (placeholder - would need actual audio URL)
                'audio_file_url': f"/static/audio/{conversation_id}_complete.mp3",
                'audio_duration_seconds': conversation_state.get('total_duration_seconds', 0),
                'transcript_telugu': self._extract_user_responses_telugu(conversation_state),
                'transcript_english': ai_analysis.get('summaries', {}).get('problem_summary_english', 'AI summary unavailable'),
                
                # AI Analysis Results
                'primary_sector': conversation_state['identified_sector'],
                'secondary_sectors': [],  # Could be enhanced with AI multi-sector detection
                'complaint_category': ai_analysis.get('problem_categorization', {}).get('primary_problem_type', 'service_gap'),
                'urgency_level': ai_analysis.get('urgency_assessment', {}).get('urgency_level', 'medium'),
                'sentiment_score': round(ai_analysis.get('sentiment_analysis', {}).get('sentiment_score', 0.0), 2),
                'emotion_detected': ai_analysis.get('sentiment_analysis', {}).get('emotion_detected', 'neutral'),
                
                # AI-Generated Content
                'problem_summary_telugu': ai_analysis.get('summaries', {}).get('problem_summary_telugu', ''),
                'problem_summary_english': ai_analysis.get('summaries', {}).get('problem_summary_english', ''),
                'location_mentioned': ai_analysis.get('content_extraction', {}).get('location_details', conversation_state['village_name']),
                'government_scheme_mentioned': ', '.join(ai_analysis.get('government_context', {}).get('schemes_mentioned', [])) or None,
                'officials_mentioned': ai_analysis.get('government_context', {}).get('officials_mentioned', []),
                
                # Classification
                'department_assigned': self.sector_department_mapping.get(
                    conversation_state['identified_sector'], 'General'
                ),
                'complexity_score': ai_analysis.get('problem_categorization', {}).get('complexity_score', 3),
                'requires_immediate_action': ai_analysis.get('urgency_assessment', {}).get('requires_immediate_action', False),
                
                # Metadata
                'conversation_duration_minutes': len(conversation_state.get('conversation_log', [])) * 2,  # Estimated
                'conversation_quality_score': min(ai_analysis.get('quality_metrics', {}).get('information_quality', 3), 5),
                'total_questions_asked': conversation_state.get('question_count', 0),
                
                # Status
                'status': 'received',
                'priority_score': ai_analysis.get('quality_metrics', {}).get('priority_score', 5),
                'follow_up_required': ai_analysis.get('summaries', {}).get('follow_up_required', True),
                
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
            logger.error(f"AI grievance creation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
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
    
    async def _save_ai_enhanced_conversation_logs(self, conversation_state: Dict[str, Any], 
                                                 grievance_id: str, ai_analysis: Dict[str, Any],
                                                 conversation_id: str) -> Dict[str, Any]:
        """
        Save detailed conversation logs enhanced with AI insights
        """
        try:
            conversation_log = conversation_state.get('conversation_log', [])
            logs_saved = 0
            
            for i, entry in enumerate(conversation_log):
                if 'user_input' in entry and 'ai_response' in entry:
                    
                    # AI sentiment analysis for individual response (simplified)
                    response_sentiment = 0.0
                    if ai_analysis.get('sentiment_analysis'):
                        # Distribute overall sentiment across responses
                        base_sentiment = ai_analysis['sentiment_analysis'].get('sentiment_score', 0.0)
                        # Add some variation based on response length/content
                        response_length_factor = len(entry['user_input'].split()) / 20.0  # Normalize
                        response_sentiment = base_sentiment * (0.8 + 0.4 * response_length_factor)
                        response_sentiment = max(-1.0, min(1.0, response_sentiment))
                    
                    log_data = {
                        'id': str(uuid.uuid4()),
                        'grievance_id': grievance_id,
                        'conversation_id': conversation_id,
                        'question_sequence': i + 1,
                        'question_asked_telugu': entry.get('ai_response', ''),
                        'question_asked_english': f"AI question #{i + 1}",
                        'user_response_telugu': entry['user_input'],
                        'user_response_english': entry['user_input'],  # Could add AI translation
                        'response_sentiment': response_sentiment,
                        'response_length_words': len(entry['user_input'].split()),
                        'ai_confidence_score': entry.get('ai_confidence_score', 
                                                        ai_analysis.get('sentiment_analysis', {}).get('confidence', 0.8)),
                        'timestamp': entry.get('timestamp', datetime.now(timezone.utc).isoformat()),
                        'stage': entry.get('stage', conversation_state.get('stage')),
                        'question_count': entry.get('question_count')
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
            logger.error(f"AI conversation logs save failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'logs_saved': 0
            }
    
    async def _generate_ai_conversation_summary(self, conversation_state: Dict[str, Any], 
                                              ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate AI-enhanced summary statistics for the conversation
        """
        conversation_log = conversation_state.get('conversation_log', [])
        
        summary = {
            # Basic metrics
            'total_exchanges': len(conversation_log),
            'user_name': conversation_state.get('user_name', 'Unknown'),
            'village_name': conversation_state.get('village_name', 'Unknown'),
            'identified_sector': conversation_state.get('identified_sector', 'Unknown'),
            'conversation_stages_completed': conversation_state.get('stage', 'unknown'),
            'total_questions_asked': conversation_state.get('question_count', 0),
            'start_time': conversation_state.get('start_time'),
            'processing_timestamp': datetime.now(timezone.utc).isoformat(),
            
            # AI Analysis Summary
            'ai_insights': {
                'sentiment_detected': ai_analysis.get('sentiment_analysis', {}).get('overall_sentiment', 'neutral'),
                'urgency_level': ai_analysis.get('urgency_assessment', {}).get('urgency_level', 'medium'),
                'problem_type': ai_analysis.get('problem_categorization', {}).get('primary_problem_type', 'unknown'),
                'requires_immediate_action': ai_analysis.get('urgency_assessment', {}).get('requires_immediate_action', False),
                'conversation_quality': ai_analysis.get('quality_metrics', {}).get('conversation_completeness', 'basic'),
                'priority_score': ai_analysis.get('quality_metrics', {}).get('priority_score', 5)
            },
            
            # Processing metadata
            'ai_processing': {
                'analysis_successful': ai_analysis.get('analysis_metadata', {}).get('processing_successful', False),
                'ai_model': ai_analysis.get('analysis_metadata', {}).get('ai_model_used', 'unknown'),
                'analysis_version': ai_analysis.get('analysis_metadata', {}).get('analysis_version', '2.0')
            }
        }
        
        return summary

# Global AI data processor instance
ai_conversation_processor = AIConversationDataProcessor()

# Utility function for easy import (maintaining compatibility)
async def save_conversation_to_database(conversation_state: Dict[str, Any], 
                                       conversation_id: str) -> Dict[str, Any]:
    """
    Main function: Save complete conversation to database with AI analysis
    """
    return await ai_conversation_processor.process_complete_conversation(
        conversation_state, conversation_id
    )

# Test function
async def test_ai_data_processor():
    """
    Test the AI data processor with sample conversation data
    """
    sample_conversation = {
        'conversation_id': 'test_ai_conv_001',
        'user_name': 'రమ్య',
        'user_gender': 'female',
        'village_name': 'స్రీకాకుళం',
        'identified_sector': 'విద్యా సేవలు',
        'question_count': 5,
        'start_time': datetime.now(timezone.utc).isoformat(),
        'conversation_log': [
            {
                'user_input': 'నా పేరు రమ్య',
                'ai_response': 'మీ గ్రామం ఏది?',
                'stage': 'name_collection',
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            {
                'user_input': 'స్రీకాకుళం నుండి వచ్చాను',
                'ai_response': 'మీకు ఏ విషయంలో సమస్య ఉంది?',
                'stage': 'sector_identification',
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            {
                'user_input': 'విద్యా సేవలలో సమస్యలు ఉన్నాయి',
                'ai_response': 'వివరంగా చెప్పండి',
                'stage': 'detailed_inquiry',
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            {
                'user_input': 'మా గ్రామంలో స్కూల్ టీచర్లు రోజూ రాటంలేదు, పిల్లలకు సరిగ్గా చదవట్లేదు. చాలా కష్టంగా ఉంది',
                'ai_response': 'మరేమైనా జోడించాలా?',
                'stage': 'confirmation',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        ],
        'collected_issues': [
            {
                'content': 'మా గ్రామంలో స్కూల్ టీచర్లు రోజూ రాటంలేదు, పిల్లలకు సరిగ్గా చదవట్లేదు. చాలా కష్టంగా ఉంది',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'stage': 'detailed_inquiry'
            }
        ]
    }
    
    result = await save_conversation_to_database(sample_conversation, 'test_ai_conv_001')
    print("AI Analysis Test Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

if __name__ == "__main__":
    import asyncio
    print("Testing JanSpandana.AI AI-Powered Data Processor...")
    asyncio.run(test_ai_data_processor())
