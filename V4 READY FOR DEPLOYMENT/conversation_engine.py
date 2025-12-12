"""
JanSpandana.AI - ULTRA-CONCISE AI-Powered Conversation Engine
Revolutionary approach: Let AI drive conversation flow with STRICT BREVITY for rural voice-first design

CORE PHILOSOPHY:
- AI understands context and user intent
- MAXIMUM 15 words per response for voice optimization
- ONE question at a time for rural users
- Intelligent sector identification without keyword matching
- Adaptive conversation flow with cultural sensitivity

USAGE:
This replaces rigid conversation engines with true AI intelligence optimized for rural voice conversations.
All conversation decisions are made by AI with strict length constraints.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum

# Gemini AI
import google.generativeai as genai
from data_processor import save_conversation_to_database
from database import db_ops, DatabaseOperations
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationStage(Enum):
    """Conversation flow stages - now AI-driven"""
    GREETING = "greeting"
    NAME_COLLECTION = "name_collection"
    SECTOR_IDENTIFICATION = "sector_identification"
    DETAILED_INQUIRY = "detailed_inquiry"
    CONFIRMATION = "confirmation"
    CONCLUSION = "conclusion"

class GenderDetection(Enum):
    """Gender detection based on Telugu names"""
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"

class UltraConciseJanSpandanaAI:
    """
    Revolutionary AI-Powered Conversation Engine with STRICT BREVITY
    Zero hardcoded questions - Pure AI intelligence with rural voice optimization
    """
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not found in environment")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize database operations
        self.db_ops = db_ops
        
        # Conversation state management
        self.active_conversations = {}
        
        # AI KNOWLEDGE BASE - Context for intelligent responses
        self.ai_context = {
            "identity": "JanSpandana.AI - వేదిక ఆయా ప్రజా సమస్యల పరిష్కారం కోసం",
            "personality": "సహానుభూతిగల, సహాయకరమైన AI - గ్రామీణ ప్రజలతో గౌరవంగా మాట్లాడుతుంది",
            "sectors": ["వైద్య సేవలు", "మౌలిక వసతులు", "విద్యా సేవలు", "సంక్షేమ పథకాలు"],
            "government_schemes": {
                "YSR రైతు భరోసా": "వ్యవసాయ రంగం - సంవత్సరానికి 13500 రూపాయలు",
                "అమ్మ వోడి": "విద్యా రంగం - పిల్లల విద్య కోసం సంవత్సరానికి 15000 రూపాయలు", 
                "YSR పెన్షన్ కాంక": "సంక్షేమ రంగం - వృద్ధులు, వికలాంగులకు మాసికం 3000 వరకు",
                "జగన్నాథ అన్న కంటీరు": "ఆహార భద్రత - ఉచిత బియ్యం పథకం",
                "ఆరోగ్యశ్రీ": "వైద్య రంగం - 5 లక్షల వరకు వైద్య సేవలు"
            },
            "cultural_context": "ఆంధ్ర ప్రదేశ్ గ్రామీణ సంస్కృతిని గౌరవిస్తుంది"
        }
    
    async def ai_detect_gender_from_name(self, name: str) -> str:
        """
        AI-powered gender detection from Telugu names
        More accurate than hardcoded patterns
        """
        try:
            prompt = f"""
            Telugu name: "{name}"
            
            Is this typically a male or female name in Telugu/Andhra Pradesh culture?
            Consider:
            - Traditional Telugu naming patterns
            - Common suffixes and prefixes
            - Cultural context
            
            Respond with ONLY: "male", "female", or "unknown"
            """
            
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            gender = response.text.strip().lower()
            
            if gender in ['male', 'female', 'unknown']:
                return gender
            return 'unknown'
            
        except Exception as e:
            logger.warning(f"AI gender detection failed: {e}")
            return 'unknown'
    
    async def ai_identify_sector(self, user_input: str, conversation_context: Dict) -> Optional[str]:
        """
        AI-powered sector identification
        Understands context, intent, and nuance - no keyword matching!
        """
        try:
            prompt = f"""
            You are JanSpandana.AI analyzing a citizen's concern in Telugu.
            
            User said: "{user_input}"
            Conversation context: User is from {conversation_context.get('village_name', 'unknown village')}
            
            Which government service sector does this relate to?
            
            SECTORS:
            1. వైద్య సేవలు (Health/Medical) - doctors, hospitals, medicines, health centers, diseases
            2. మౌలిక వసతులు (Infrastructure) - roads, water supply, electricity, drainage, transport
            3. విద్యా సేవలు (Education) - schools, teachers, education quality, children's learning
            4. సంక్షేమ పథకాలు (Welfare Schemes) - pensions, ration cards, government benefits
            
            IMPORTANT:
            - Consider the MEANING and INTENT, not just keywords
            - Understand context and implications
            - If user mentions multiple sectors, pick the PRIMARY concern
            - If unclear, choose the most likely sector based on context
            
            Respond with ONLY the Telugu sector name (e.g., "వైద్య సేవలు")
            If truly unclear, respond with "unclear"
            """
            
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            identified_sector = response.text.strip()
            
            # Validate response
            valid_sectors = ["వైద్య సేవలు", "మౌలిక వసతులు", "విద్యా సేవలు", "సంక్షేమ పథకాలు"]
            if identified_sector in valid_sectors:
                logger.info(f"AI identified sector: {identified_sector}")
                return identified_sector
            
            logger.warning(f"AI sector identification unclear: {identified_sector}")
            return None
            
        except Exception as e:
            logger.error(f"AI sector identification failed: {e}")
            return None
    
    async def ai_generate_dynamic_response(self, stage: ConversationStage, 
                                         conversation_state: Dict, user_input: str) -> Dict[str, Any]:
        """
        AI generates ULTRA-CONCISE responses for rural voice-first conversations
        """
        try:
            # Build ultra-strict context for AI
            context = self._build_ai_context(stage, conversation_state, user_input)
            
            # Generate AI response with strict constraints
            response = await asyncio.to_thread(self.model.generate_content, context)
            response_text = response.text
            
            # Parse AI response
            structured_response = await self._parse_ai_response(response_text, stage, conversation_state)
            
            return structured_response
            
        except Exception as e:
            logger.error(f"AI response generation failed: {str(e)}")
            return self._get_fallback_response(stage, conversation_state)
    
    def _build_ai_context(self, stage: ConversationStage, state: Dict, user_input: str) -> str:
        """
        Build ULTRA-CONCISE context for AI - enforces rural voice-friendly brevity
        """
        user_name = state.get('user_name', '')
        user_gender = state.get('user_gender', 'unknown')
        village_name = state.get('village_name', '')
        identified_sector = state.get('identified_sector', '')
        conversation_log = state.get('conversation_log', [])
        
        # Determine appropriate honorific
        honorific = self._get_honorific(user_name, user_gender)
        
        # Ultra-strict brevity prompt
        base_context = f"""
You are JanSpandana.AI - Telugu grievance assistant for rural AP.

🚨 CRITICAL CONSTRAINTS - NEVER VIOLATE:
- MAXIMUM 15 words in Telugu response
- Ask ONLY ONE simple question
- Use conversational rural Telugu
- NO explanations, NO repetitions, NO multiple questions
- Be direct and warm

CURRENT SITUATION:
User: {user_name} {honorific}
Village: {village_name} 
Stage: {stage.value}
Last input: "{user_input}"

STAGE-SPECIFIC INSTRUCTIONS:
"""

        # Stage-specific ultra-brief instructions
        if stage == ConversationStage.GREETING:
            base_context += """
GREETING: Welcome warmly and ask for name only.
EXAMPLE: "నమస్కారం! మీ పేరు ఏమిటి?"
"""
            
        elif stage == ConversationStage.NAME_COLLECTION:
            base_context += f"""
NAME_COLLECTION: Thank for name, ask village only.
EXAMPLE: "స్వాగతం {user_name} {honorific}! మీ గ్రామం ఏది?"
"""
            
        elif stage == ConversationStage.SECTOR_IDENTIFICATION:
            base_context += f"""
SECTOR_IDENTIFICATION: Ask about problem type briefly.
EXAMPLE: "{honorific}, ఏ విషయంలో సమస్య ఉంది?"
"""
            
        elif stage == ConversationStage.DETAILED_INQUIRY:
            base_context += f"""
DETAILED_INQUIRY: Ask ONE specific detail about their {identified_sector} problem.
EXAMPLES:
- "ఇది ఎంతకాలంగా ఉంది?"
- "ఎంత మంది ప్రభావితమయ్యారు?"
- "ఎవరికి చెప్పారు ఇంతకుముందు?"
"""
            
        elif stage == ConversationStage.CONFIRMATION:
            base_context += f"""
CONFIRMATION: Briefly confirm understanding.
EXAMPLE: "అర్థమైంది {honorific}. మరేమైనా చెప్పాలా?"
"""
            
        elif stage == ConversationStage.CONCLUSION:
            base_context += f"""
CONCLUSION: Thank briefly and promise action.
EXAMPLE: "ధన్యవాదాలు {honorific}! మీ సమస్య అధికారులకు పంపిస్తాం."
"""

        base_context += f"""

RESPONSE REQUIREMENTS:
Return ONLY this JSON format:
{{
    "telugu_response": "10-15 word Telugu response with ONE question",
    "english_summary": "Brief English summary",
    "stage_complete": true/false,
    "next_stage": "stage_name_or_null",
    "confidence_score": 0.8,
    "reasoning": "Why this response"
}}

GOOD RESPONSE EXAMPLES (COPY THIS STYLE):
- "అన్న, మీ పేరు ఏమిటి?" (6 words - PERFECT)
- "గ్రామం పేరు చెప్పండి అక్క" (5 words - PERFECT)  
- "ఏ విషయంలో సమస్య ఉంది?" (5 words - PERFECT)
- "ఇది ఎంతకాలంగా ఉంది?" (4 words - PERFECT)
- "మరేమైనా చెప్పాలా?" (3 words - PERFECT)

BAD EXAMPLES (NEVER DO THIS):
❌ "మీరు చెప్పిన సమస్య నాకు అర్థమైంది. స్కూల్లో టీచర్లు తక్కువగా ఉన్నారని..."
❌ Any response longer than 15 words
❌ Multiple questions in one response
❌ Repeating what user said

🎯 GENERATE ULTRA-SHORT RESPONSE NOW (MAX 15 WORDS):
"""
        
        return base_context
    
    async def _parse_ai_response(self, ai_response: str, stage: ConversationStage, state: Dict) -> Dict[str, Any]:
        """
        Parse AI response and structure it for application use
        More flexible than rigid JSON parsing
        """
        try:
            # Try to extract JSON from response
            start = ai_response.find('{')
            end = ai_response.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = ai_response[start:end]
                parsed_response = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['telugu_response', 'stage_complete']
                for field in required_fields:
                    if field not in parsed_response:
                        raise ValueError(f"Missing required field: {field}")
                
                # Enforce length constraint
                telugu_response = parsed_response['telugu_response']
                word_count = len(telugu_response.split())
                if word_count > 15:
                    logger.warning(f"AI response too long ({word_count} words), truncating")
                    # Take first 15 words and add question mark if missing
                    words = telugu_response.split()[:15]
                    parsed_response['telugu_response'] = ' '.join(words)
                    if not parsed_response['telugu_response'].endswith('?'):
                        parsed_response['telugu_response'] += '?'
                
            else:
                # Fallback: extract Telugu response from raw text
                telugu_text = ai_response.strip()
                # Clean up common AI artifacts
                telugu_text = telugu_text.replace('```json', '').replace('```', '').strip()
                
                # Truncate if too long
                words = telugu_text.split()
                if len(words) > 15:
                    telugu_text = ' '.join(words[:15]) + '?'
                
                parsed_response = {
                    "telugu_response": telugu_text,
                    "english_summary": "AI response (parsed from text)",
                    "stage_complete": True,
                    "confidence_score": 0.7,
                    "reasoning": "Fallback parsing used"
                }
            
            # Add metadata
            parsed_response.update({
                'current_stage': stage.value,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'conversation_id': state.get('conversation_id'),
                'processing_successful': True,
                'ai_powered': True,
                'word_count': len(parsed_response['telugu_response'].split())
            })
            
            return parsed_response
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            return self._get_fallback_response(stage, state)
        except Exception as e:
            logger.error(f"Response parsing failed: {str(e)}")
            return self._get_fallback_response(stage, state)
    
    def _get_fallback_response(self, stage: ConversationStage, state: Dict) -> Dict[str, Any]:
        """
        Ultra-short fallback responses when AI fails
        """
        user_name = state.get('user_name', '')
        user_gender = state.get('user_gender', 'unknown')
        honorific = self._get_honorific(user_name, user_gender)
        
        fallback_responses = {
            ConversationStage.GREETING: "నమస్కారం! మీ పేరు ఏమిటి?",
            ConversationStage.NAME_COLLECTION: f"స్వాగతం {user_name} {honorific}! మీ గ్రామం ఏది?",
            ConversationStage.SECTOR_IDENTIFICATION: f"{honorific}, ఏ విషయంలో సమస్య ఉంది?",
            ConversationStage.DETAILED_INQUIRY: f"వివరంగా చెప్పండి {honorific}",
            ConversationStage.CONFIRMATION: f"అర్థమైంది {honorific}. మరేమైనా చెప్పాలా?",
            ConversationStage.CONCLUSION: f"ధన్యవాదాలు {honorific}! మీ సమస్యలను పరిష్కరిస్తాం."
        }
        
        return {
            'telugu_response': fallback_responses.get(stage, "దయచేసి మళ్లీ చెప్పండి"),
            'english_summary': 'Fallback response due to AI processing failure',
            'stage_complete': True,
            'next_stage': self._get_next_stage(stage),
            'processing_successful': False,
            'fallback_used': True,
            'confidence_score': 0.3,
            'word_count': len(fallback_responses.get(stage, "").split())
        }
    
    def _get_next_stage(self, current_stage: ConversationStage) -> str:
        """Get the next stage in conversation flow"""
        stage_flow = {
            ConversationStage.GREETING: "name_collection",
            ConversationStage.NAME_COLLECTION: "sector_identification", 
            ConversationStage.SECTOR_IDENTIFICATION: "detailed_inquiry",
            ConversationStage.DETAILED_INQUIRY: "confirmation",
            ConversationStage.CONFIRMATION: "conclusion",
            ConversationStage.CONCLUSION: None
        }
        return stage_flow.get(current_stage)
    
    def _get_honorific(self, name: str, gender: str) -> str:
        """
        Get appropriate Telugu honorific based on name and gender
        """
        if not name:
            return "గారూ"
            
        if gender == 'male':
            return "అన్న"  # Brother (respectful for males)
        elif gender == 'female':
            return "అక్క"  # Sister (respectful for females)
        else:
            return "గారూ"  # Universal respectful term
    
    def _normalize_stage(self, stage_value: Optional[str], fallback: str) -> str:
        """
        Normalize any AI-provided stage string to a valid ConversationStage value.
        Falls back to the provided fallback stage when no mapping is found.
        """
        stage_key = str(stage_value or "").strip().lower()
        
        mapping = {
            # Greeting variants
            "greeting": ConversationStage.GREETING.value,
            "intro": ConversationStage.GREETING.value,
            "welcome": ConversationStage.GREETING.value,
            
            # Name collection variants (also used when AI asks for village right after name)
            "name_collection": ConversationStage.NAME_COLLECTION.value,
            "collect_name": ConversationStage.NAME_COLLECTION.value,
            "ask_name": ConversationStage.NAME_COLLECTION.value,
            "ask_for_name": ConversationStage.NAME_COLLECTION.value,
            "get_name": ConversationStage.NAME_COLLECTION.value,
            "name": ConversationStage.NAME_COLLECTION.value,
            "name_provided": ConversationStage.NAME_COLLECTION.value,
            "ask_for_village": ConversationStage.NAME_COLLECTION.value,
            "village_prompt": ConversationStage.NAME_COLLECTION.value,
            "collect_village": ConversationStage.NAME_COLLECTION.value,
            "village_collection": ConversationStage.NAME_COLLECTION.value,
            
            # Sector identification variants
            "sector_identification": ConversationStage.SECTOR_IDENTIFICATION.value,
            "sector_choice": ConversationStage.SECTOR_IDENTIFICATION.value,
            "sector_selection": ConversationStage.SECTOR_IDENTIFICATION.value,
            "choose_sector": ConversationStage.SECTOR_IDENTIFICATION.value,
            "select_sector": ConversationStage.SECTOR_IDENTIFICATION.value,
            "sector": ConversationStage.SECTOR_IDENTIFICATION.value,
            "choose_service": ConversationStage.SECTOR_IDENTIFICATION.value,
            "select_service": ConversationStage.SECTOR_IDENTIFICATION.value,
            
            # Detailed inquiry variants
            "detailed_inquiry": ConversationStage.DETAILED_INQUIRY.value,
            "detail_inquiry": ConversationStage.DETAILED_INQUIRY.value,
            "issue_details": ConversationStage.DETAILED_INQUIRY.value,
            "problem_details": ConversationStage.DETAILED_INQUIRY.value,
            "information_gathering": ConversationStage.DETAILED_INQUIRY.value,
            "followup": ConversationStage.DETAILED_INQUIRY.value,
            "follow_up": ConversationStage.DETAILED_INQUIRY.value,
            "collect_details": ConversationStage.DETAILED_INQUIRY.value,
            "detailed_questions": ConversationStage.DETAILED_INQUIRY.value,
            
            # Confirmation variants
            "confirmation": ConversationStage.CONFIRMATION.value,
            "confirm": ConversationStage.CONFIRMATION.value,
            "review": ConversationStage.CONFIRMATION.value,
            "summary": ConversationStage.CONFIRMATION.value,
            "recap": ConversationStage.CONFIRMATION.value,
            "acknowledge": ConversationStage.CONFIRMATION.value,
            
            # Conclusion variants
            "conclusion": ConversationStage.CONCLUSION.value,
            "closing": ConversationStage.CONCLUSION.value,
            "close": ConversationStage.CONCLUSION.value,
            "end": ConversationStage.CONCLUSION.value,
            "goodbye": ConversationStage.CONCLUSION.value,
            "farewell": ConversationStage.CONCLUSION.value,
            "wrap_up": ConversationStage.CONCLUSION.value,
            "finish": ConversationStage.CONCLUSION.value,
        }
        
        return mapping.get(stage_key, fallback)

    async def start_new_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Initialize a new conversation session
        """
        initial_state = {
            'conversation_id': conversation_id,
            'stage': ConversationStage.GREETING.value,
            'user_name': '',
            'user_gender': 'unknown',
            'village_name': '',
            'identified_sector': '',
            'collected_issues': [],
            'conversation_log': [],
            'start_time': datetime.now(timezone.utc).isoformat(),
            'question_count': 0,
            'ai_powered': True,
            'ultra_concise_mode': True
        }
        
        self.active_conversations[conversation_id] = initial_state
        
        # Generate AI greeting
        greeting_response = await self.ai_generate_dynamic_response(
            ConversationStage.GREETING, initial_state, ""
        )
        
        return {
            'conversation_id': conversation_id,
            'initial_response': greeting_response,
            'session_started': True,
            'ultra_concise_mode': True
        }
    
    async def process_user_input(self, conversation_id: str, user_input: str, 
                                audio_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process user input with ultra-concise AI intelligence
        """
        if conversation_id not in self.active_conversations:
            return {'error': 'Conversation not found', 'success': False}
        
        state = self.active_conversations[conversation_id]
        normalized_stage = self._normalize_stage(state.get('stage'), ConversationStage.GREETING.value)
        state['stage'] = normalized_stage
        current_stage = ConversationStage(normalized_stage)
        
        # AI-powered conversation state update
        await self._ai_update_conversation_state(state, user_input, current_stage)
        
        # Generate ultra-concise AI response
        ai_response = await self.ai_generate_dynamic_response(
            ConversationStage(state['stage']), state, user_input
        )
        
        # Log conversation step
        conversation_step = {
            'conversation_id': conversation_id,
            'user_input': user_input,
            'ai_response': ai_response['telugu_response'],
            'stage': state['stage'],
            'question_count': state['question_count'],
            'audio_metadata': audio_metadata,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'ai_confidence': ai_response.get('confidence_score', 0.0),
            'response_word_count': ai_response.get('word_count', 0)
        }
        
        state['conversation_log'].append(conversation_step)
        state['question_count'] += 1

        return {
            'success': True,
            'ai_response': ai_response,
            'conversation_state': state,
            'step_logged': True,
            'ultra_concise_mode': True
        }
    
    async def _ai_update_conversation_state(self, state: Dict, user_input: str, current_stage: ConversationStage):
        """
        AI-powered conversation state management with brevity focus
        """
        try:
            # Build context for AI state management
            context = f"""
            You are managing conversation state for JanSpandana.AI.
            
            CURRENT STATE:
            - Stage: {current_stage.value}
            - User name: {state.get('user_name', 'not_set')}
            - Village: {state.get('village_name', 'not_set')}
            - Sector: {state.get('identified_sector', 'not_set')}
            
            USER JUST SAID: "{user_input}"
            
            TASKS:
            1. Extract any new information (name, village, sector preference)
            2. Decide if we should progress to next stage
            3. Update conversation state appropriately
            
            RESPOND WITH JSON:
            {{
                "extracted_name": "name if mentioned, else null",
                "extracted_village": "village if mentioned, else null",
                "extracted_sector": "sector if identified, else null", 
                "should_progress": true/false,
                "next_stage": "stage_name if progressing",
                "reasoning": "why these decisions were made"
            }}
            """
            
            response = await asyncio.to_thread(self.model.generate_content, context)
            
            # Parse AI decision
            try:
                start = response.text.find('{')
                end = response.text.rfind('}') + 1
                if start != -1 and end > start:
                    ai_decision = json.loads(response.text[start:end])
                    
                    # Update state based on AI analysis
                    if ai_decision.get('extracted_name') and not state.get('user_name'):
                        state['user_name'] = ai_decision['extracted_name'].strip()
                        # AI-powered gender detection
                        state['user_gender'] = await self.ai_detect_gender_from_name(state['user_name'])
                    
                    if ai_decision.get('extracted_village') and not state.get('village_name'):
                        state['village_name'] = ai_decision['extracted_village'].strip()
                    
                    if ai_decision.get('extracted_sector'):
                        # Use AI sector identification for accuracy
                        identified_sector = await self.ai_identify_sector(user_input, state)
                        if identified_sector:
                            state['identified_sector'] = identified_sector
                    
                    # AI decides when to progress stages
                    if ai_decision.get('should_progress') and ai_decision.get('next_stage'):
                        state['stage'] = self._normalize_stage(
                            ai_decision.get('next_stage'),
                            self._get_next_stage(current_stage) or current_stage.value
                        )
                    
                    logger.info(f"AI state update: {ai_decision.get('reasoning', 'No reasoning provided')}")
                    
            except json.JSONDecodeError:
                logger.warning("AI state management JSON parse failed, using fallback logic")
                # Simple fallback state progression
                self._fallback_state_update(state, user_input, current_stage)
                
        except Exception as e:
            logger.error(f"AI state management failed: {e}")
            # Fallback to simple state management
            self._fallback_state_update(state, user_input, current_stage)
    
    def _fallback_state_update(self, state: Dict, user_input: str, current_stage: ConversationStage):
        """
        Simple fallback state management when AI fails
        """
        if current_stage == ConversationStage.GREETING and user_input:
            # Extract name (simple approach)
            name = user_input.strip()
            if 'పేరు' in user_input:
                parts = user_input.split('పేరు')
                if len(parts) > 1:
                    name = parts[1].strip()
            state['user_name'] = name
            state['user_gender'] = 'unknown'  # Fallback
            state['stage'] = ConversationStage.NAME_COLLECTION.value
            
        elif current_stage == ConversationStage.NAME_COLLECTION and user_input:
            state['village_name'] = user_input.strip()
            state['stage'] = ConversationStage.SECTOR_IDENTIFICATION.value
            
        elif current_stage == ConversationStage.SECTOR_IDENTIFICATION and user_input:
            # Simple sector matching as fallback
            if any(word in user_input.lower() for word in ['వైద్య', 'డాక్టరు', 'ఆస్పత్రి']):
                state['identified_sector'] = 'వైద్య సేవలు'
            elif any(word in user_input.lower() for word in ['స్కూలు', 'విద్య', 'టీచరు']):
                state['identified_sector'] = 'విద్యా సేవలు'
            elif any(word in user_input.lower() for word in ['రోడు', 'నీరు', 'కరెంట్']):
                state['identified_sector'] = 'మౌలిక వసతులు'
            else:
                state['identified_sector'] = 'సంక్షేమ పథకాలు'
            state['stage'] = ConversationStage.DETAILED_INQUIRY.value
            
        # Add user input to collected issues
        if len(user_input.split()) > 2:  # Substantial input
            state['collected_issues'].append({
                'content': user_input,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'stage': state['stage']
            })
    
    async def end_conversation(self, conversation_id: str, skip_save: bool = False) -> Dict[str, Any]:
        """
        End conversation and save to database
        """
        if conversation_id not in self.active_conversations:
            return {'error': 'Conversation not found', 'success': False}
        
        state = self.active_conversations[conversation_id]

        # Allow health checks or explicit skip requests to bypass persistence
        if skip_save or conversation_id.startswith("health_check"):
            logger.info(f"Skipping database save for conversation {conversation_id} (health check/skip flag)")
            del self.active_conversations[conversation_id]
            return {
                'success': True,
                'summary': {
                    'conversation_id': conversation_id,
                    'completion_status': 'skipped_persistence',
                    'ai_powered': True,
                    'ultra_concise_mode': True
                }
            }
        
        # Prepare data for database storage
        conversation_summary = {
            'conversation_id': conversation_id,
            'user_name': state['user_name'],
            'user_gender': state['user_gender'],
            'village_name': state['village_name'],
            'identified_sector': state['identified_sector'],
            'total_questions': state['question_count'],
            'issues_collected': state['collected_issues'],
            'conversation_log': state['conversation_log'],
            'start_time': state['start_time'],
            'end_time': datetime.now(timezone.utc).isoformat(),
            'completion_status': 'completed',
            'ai_powered': True,
            'ultra_concise_mode': True
        }
        
        # Save to database using data processor
        try:
            db_result = await save_conversation_to_database(state, conversation_id)
            
            if db_result['success']:
                logger.info(f"✅ Database save successful: {db_result['grievance_id']}")
                conversation_summary['database_saved'] = True
                conversation_summary['grievance_id'] = db_result['grievance_id']
                conversation_summary['user_id'] = db_result['user_id']
            else:
                logger.error(f"❌ Database save failed: {db_result['error']}")
                conversation_summary['database_saved'] = False
                conversation_summary['database_error'] = db_result['error']
            
            # Clean up active conversation
            del self.active_conversations[conversation_id]
            
            return {
                'success': True,
                'summary': conversation_summary
            }

        except Exception as e:
            logger.error(f"Failed to save conversation: {str(e)}")
            return {
                'success': False,
                'error': 'Failed to save conversation',
                'summary': conversation_summary
            }

# Global AI conversation engine instance
jan_spandana_ai = UltraConciseJanSpandanaAI()

# Utility functions for easy import (maintaining compatibility)
async def start_conversation(conversation_id: str) -> Dict[str, Any]:
    """Start a new conversation session"""
    return await jan_spandana_ai.start_new_conversation(conversation_id)

async def process_message(conversation_id: str, user_input: str, 
                         audio_metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Process user message and get AI response"""
    return await jan_spandana_ai.process_user_input(conversation_id, user_input, audio_metadata)

async def end_conversation_session(conversation_id: str, skip_save: bool = False) -> Dict[str, Any]:
    """End conversation and save data"""
    return await jan_spandana_ai.end_conversation(conversation_id, skip_save=skip_save)

if __name__ == "__main__":
    # Test ultra-concise conversation engine
    async def test_ultra_concise_conversation():
        print("Testing JanSpandana.AI Ultra-Concise Conversation Engine...")
        
        # Start new conversation
        conv_id = "test_concise_001"
        start_result = await start_conversation(conv_id)
        
        if start_result['session_started']:
            print("✅ Ultra-concise conversation started successfully")
            print(f"Initial response: {start_result['initial_response']['telugu_response']}")
            word_count = len(start_result['initial_response']['telugu_response'].split())
            print(f"Word count: {word_count} (Target: ≤15)")
        else:
            print("❌ Failed to start conversation")
            return
        
        # Test realistic conversation flow
        test_messages = [
            "నా పేరు రమ్య",
            "శ్రీకాకుళం నుంచి వచ్చాను",
            "విద్యా సేవలలో సమస్యలు ఉన్నాయి", 
            "మా గ్రామంలో స్కూలు టీచర్లు రోజూ రాటంలేదు, పిల్లలకు సరిగ్గా చదవటంలేదు"
        ]
        
        for message in test_messages:
            print(f"\nUser: {message}")
            response = await process_message(conv_id, message)
            if response['success']:
                ai_resp = response['ai_response']
                print(f"AI: {ai_resp['telugu_response']}")
                word_count = ai_resp.get('word_count', len(ai_resp['telugu_response'].split()))
                print(f"Word count: {word_count} (Target: ≤15)")
                if word_count > 15:
                    print("⚠️ WARNING: Response exceeds 15-word limit!")
                else:
                    print("✅ Within word limit")
            else:
                print(f"Error: {response}")
        
        # End conversation
        end_result = await end_conversation_session(conv_id)
        if end_result['success']:
            print("\n✅ Ultra-concise conversation ended successfully")
            print("🚀 AI-powered ultra-brief conversation completed!")
        else:
            print(f"\n❌ Failed to end conversation: {end_result}")
    
    asyncio.run(test_ultra_concise_conversation())