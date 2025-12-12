"""
JanSpandana.AI - DYNAMIC AI-Powered Conversation Engine
Revolutionary approach: Let AI drive the entire conversation flow
NO hardcoded questions, NO string matching - Pure AI intelligence

CORE PHILOSOPHY:
- AI understands context and user intent
- Dynamic question generation based on conversation history
- Intelligent sector identification without keyword matching
- Adaptive conversation flow that responds to user naturally

USAGE:
This replaces the rigid conversation_engine.py with true AI intelligence.
All conversation decisions are made by AI, not hardcoded rules.
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

class DynamicJanSpandanaAI:
    """
    Revolutionary AI-Powered Conversation Engine
    Zero hardcoded questions - Pure AI intelligence drives everything
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
                "YSR రైతు భరోసా": "వ్యవసాయ రంగం - సంవత్సరానికి 13500 రుపాయలు",
                "అమ్మ ఒడి": "విద్యా రంగం - పిల్లల విద్య కోసం సంవత్సరానికి 15000 రుపాయలు", 
                "YSR పెన్షను కాంక": "సంక్షేమ రంగం - వృద్ధులు, వికలాంగులకు మాసికం 3000 వరకు",
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
        AI generates completely dynamic responses based on conversation context
        No templates, no rigid rules - pure conversational intelligence
        """
        try:
            # Build rich context for AI
            context = self._build_ai_context(stage, conversation_state, user_input)
            
            # Generate AI response
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
        Build intelligent context for AI - no rigid templates
        AI understands the full conversation context and responds naturally
        """
        user_name = state.get('user_name', '')
        user_gender = state.get('user_gender', 'unknown')
        village_name = state.get('village_name', '')
        identified_sector = state.get('identified_sector', '')
        conversation_log = state.get('conversation_log', [])
        
        # Determine appropriate honorific
        honorific = self._get_honorific(user_name, user_gender)
        
        base_context = f"""
You are JanSpandana.AI (జన స్పందన AI) - an empathetic AI assistant helping rural citizens of Andhra Pradesh voice their government service concerns.

CORE IDENTITY:
- Respectful and empathetic Telugu conversation
- Government grievance collection system
- Cultural sensitivity for rural AP communities
- Professional yet warm communication style

CURRENT SITUATION:
- Stage: {stage.value}
- User: {user_name} {honorific} (Gender: {user_gender})
- Village: {village_name}
- Identified Sector: {identified_sector}
- Latest User Input: "{user_input}"

CONVERSATION HISTORY:
{json.dumps(conversation_log[-3:], indent=2, ensure_ascii=False) if conversation_log else "Starting conversation"}

CULTURAL CONTEXT:
- Use appropriate Telugu honorifics: {honorific}
- Be respectful of rural culture and communication style
- Understand AP government context and schemes
- Show genuine concern for citizen issues

RESPONSE REQUIREMENTS:
Return a JSON object with:
{{
    "telugu_response": "Natural conversational Telugu response",
    "english_summary": "Brief English summary of the response",
    "stage_complete": true/false,
    "next_stage": "stage_name_or_null",
    "identified_data": {{"any new information extracted"}},
    "confidence_score": 0.0 to 1.0,
    "reasoning": "Brief explanation of response decision"
}}

STAGE-SPECIFIC INSTRUCTIONS:
"""
        
        # Add stage-specific intelligent instructions
        if stage == ConversationStage.GREETING:
            base_context += """
GREETING STAGE:
- Warmly welcome the user in Telugu
- Ask for their name naturally
- Set a comfortable, respectful tone
- Make them feel heard and valued
"""
            
        elif stage == ConversationStage.NAME_COLLECTION:
            base_context += f"""
NAME COLLECTION STAGE:
- User gave name: "{user_input}"
- Acknowledge their name respectfully with {honorific}
- Ask about their village/location naturally
- Show genuine interest in their background
- Build trust and rapport
"""
            
        elif stage == ConversationStage.SECTOR_IDENTIFICATION:
            base_context += f"""
SECTOR IDENTIFICATION STAGE:
- User mentioned village: {village_name}
- Help them identify their main concern
- Present government service areas clearly:
  1. వైద్య సేవలు (Health/Medical services)
  2. మౌలిక వసతులు (Infrastructure - roads, water, electricity)
  3. విద్యా సేవలు (Education services)
  4. సంక్షేమ పథకాలు (Welfare schemes)
- Encourage them to share their main concern area
- Be patient if they need clarification
"""
            
        elif stage == ConversationStage.DETAILED_INQUIRY:
            base_context += f"""
DETAILED INQUIRY STAGE:
- Focus area: {identified_sector}
- User input: "{user_input}"

INTELLIGENT QUESTIONING:
- Generate follow-up questions based on what user actually said
- Ask for specific details about their situation
- Understand the real problem they're facing
- Be empathetic and encouraging
- Probe for actionable details that can help solve their issue

EXAMPLE APPROACH:
If they mentioned school issues - ask about specific problems (teacher attendance, infrastructure, supplies)
If they mentioned health issues - ask about accessibility, quality, staff availability
If they mentioned infrastructure - ask about specific problems (water, electricity, roads)
If they mentioned welfare - ask about scheme access, documentation, payments

IMPORTANT: Base questions on their ACTUAL responses, not templates!
"""
            
        elif stage == ConversationStage.CONFIRMATION:
            base_context += f"""
CONFIRMATION STAGE:
- Summarize their main concern clearly
- Confirm understanding of their problem
- Ask if they want to add anything else
- Prepare to conclude the conversation
- Show appreciation for their time and trust
"""
            
        elif stage == ConversationStage.CONCLUSION:
            base_context += f"""
CONCLUSION STAGE:
- Thank them sincerely for sharing their concerns
- Assure them their feedback will reach relevant authorities
- Give them confidence that follow-up will happen
- End on a positive, hopeful note
- Express gratitude for using JanSpandana.AI
"""
        
        base_context += f"""

IMPORTANT GUIDELINES:
1. Be completely natural and conversational
2. Respond to what the user ACTUALLY said, not template expectations
3. Show genuine empathy and understanding
4. Use cultural context appropriately
5. Generate relevant follow-up questions based on their specific situation
6. Don't force rigid conversation flow - adapt to user needs

GENERATE RESPONSE NOW:
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
                
            else:
                # Fallback: extract Telugu response from raw text
                telugu_text = ai_response.strip()
                # Clean up common AI artifacts
                telugu_text = telugu_text.replace('```json', '').replace('```', '').strip()
                
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
                'ai_powered': True
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
        Simple fallback responses when AI fails
        Much simpler than before - just basic responses
        """
        user_name = state.get('user_name', '')
        user_gender = state.get('user_gender', 'unknown')
        honorific = self._get_honorific(user_name, user_gender)
        
        fallback_responses = {
            ConversationStage.GREETING: "నమస్కారం! నేను జన స్పందన AI. మీ పేరు ఏమిటి?",
            ConversationStage.NAME_COLLECTION: f"స్వాగతం {user_name} {honorific}! మీరు ఏ గ్రామం నుండి వచ్చారు?",
            ConversationStage.SECTOR_IDENTIFICATION: f"{honorific}, మీకు ఏ విషయంలో సమస్య ఉంది?",
            ConversationStage.DETAILED_INQUIRY: f"దయచేసి వివరంగా చెప్పండి {honorific}",
            ConversationStage.CONFIRMATION: f"మీ సమస్య అర్థమైంది {honorific}. ధన్యవాదాలు!",
            ConversationStage.CONCLUSION: f"ధన్యవాదాలు {honorific}! మీ సమస్యలను పరిష్కరిస్తాము."
        }
        
        return {
            'telugu_response': fallback_responses.get(stage, "దయచేసి మళ్లీ చెప్పండి"),
            'english_summary': 'Fallback response due to AI processing failure',
            'stage_complete': True,
            'next_stage': self._get_next_stage(stage),
            'processing_successful': False,
            'fallback_used': True,
            'confidence_score': 0.3
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
            'ai_powered': True
        }
        
        self.active_conversations[conversation_id] = initial_state
        
        # Generate AI greeting
        greeting_response = await self.ai_generate_dynamic_response(
            ConversationStage.GREETING, initial_state, ""
        )
        
        return {
            'conversation_id': conversation_id,
            'initial_response': greeting_response,
            'session_started': True
        }
    
    async def process_user_input(self, conversation_id: str, user_input: str, 
                                audio_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process user input with full AI intelligence
        """
        if conversation_id not in self.active_conversations:
            return {'error': 'Conversation not found', 'success': False}
        
        state = self.active_conversations[conversation_id]
        normalized_stage = self._normalize_stage(state.get('stage'), ConversationStage.GREETING.value)
        state['stage'] = normalized_stage
        current_stage = ConversationStage(normalized_stage)
        
        # AI-powered conversation state update
        await self._ai_update_conversation_state(state, user_input, current_stage)
        
        # Generate AI response
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
            'ai_confidence': ai_response.get('confidence_score', 0.0)
        }
        
        state['conversation_log'].append(conversation_step)
        state['question_count'] += 1

        return {
            'success': True,
            'ai_response': ai_response,
            'conversation_state': state,
            'step_logged': True
        }
    
    async def _ai_update_conversation_state(self, state: Dict, user_input: str, current_stage: ConversationStage):
        """
        AI-powered conversation state management
        No hardcoded rules - AI decides what information to extract and when to progress
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
            if any(word in user_input.lower() for word in ['వైద్య', 'డాక్టర్', 'ఆసుపత్రి']):
                state['identified_sector'] = 'వైద్య సేవలు'
            elif any(word in user_input.lower() for word in ['స్కూలు', 'విద్య', 'టీచర్']):
                state['identified_sector'] = 'విద్యా సేవలు'
            elif any(word in user_input.lower() for word in ['రోడ్', 'నీరు', 'కరెంట్']):
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
                    'ai_powered': True
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
            'ai_powered': True
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
jan_spandana_ai = DynamicJanSpandanaAI()

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
    # Test dynamic conversation engine
    async def test_dynamic_conversation():
        print("Testing JanSpandana.AI Dynamic Conversation Engine...")
        
        # Start new conversation
        conv_id = "test_dynamic_001"
        start_result = await start_conversation(conv_id)
        
        if start_result['session_started']:
            print("✅ Dynamic conversation started successfully")
            print(f"Initial response: {start_result['initial_response']['telugu_response']}")
        else:
            print("❌ Failed to start conversation")
            return
        
        # Test realistic conversation flow
        test_messages = [
            "నా పేరు రమ్య",
            "స్రీకాకుళం నుండి వచ్చాను",
            "విద్యా సేవలలో సమస్యలు ఉన్నాయి", 
            "మా గ్రామంలో స్కూల్ టీచర్లు రోజూ రాటంలేదు, పిల్లలకు సరిగ్గా చదవట్లేదు"
        ]
        
        for message in test_messages:
            print(f"\nUser: {message}")
            response = await process_message(conv_id, message)
            if response['success']:
                ai_resp = response['ai_response']
                print(f"AI: {ai_resp['telugu_response']}")
                print(f"Stage: {ai_resp['current_stage']}")
                print(f"Confidence: {ai_resp.get('confidence_score', 'N/A')}")
                if 'reasoning' in ai_resp:
                    print(f"AI Reasoning: {ai_resp['reasoning']}")
            else:
                print(f"Error: {response}")
        
        # End conversation
        end_result = await end_conversation_session(conv_id)
        if end_result['success']:
            print("\n✅ Dynamic conversation ended successfully")
            print("🚀 AI-powered conversation completed!")
        else:
            print(f"\n❌ Failed to end conversation: {end_result}")
    
    asyncio.run(test_dynamic_conversation())
