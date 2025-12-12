"""
JanSpandana.AI - FIXED Gemini AI Conversation Engine
Conducts empathetic interviews in Telugu for grievance collection
FIXED: Clear prompts, proper stage progression, name/village collection

KEY FIXES APPLIED:
1. Clear, specific Telugu instructions for each conversation stage
2. Proper stage progression logic with stage_complete and next_stage
3. Village name collection in NAME_COLLECTION stage
4. Gender-based honorific addressing (అన్న/అక్క/గారూ)
5. Sector identification and detailed inquiry flow
6. Better state management and conversation tracking

USAGE:
Replace your existing conversation_engine.py with this fixed version.
The main fixes are in the _build_conversation_prompt() function.
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
# Add to existing imports
from data_processor import save_conversation_to_database
# Local imports
from database import db_ops, DatabaseOperations
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationStage(Enum):
    """Conversation flow stages"""
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

class JanSpandanaAI:
    """
    Main conversation engine for JanSpandana.AI
    FIXED VERSION with clear conversation flow
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
        
        # Telugu name patterns for gender detection
        self.male_name_patterns = [
            'రామ', 'కృష్ణ', 'వెంకట', 'రామ', 'సుధాకర', 'రవి', 'కుమార', 'బాబు', 'రెడ్డి', 
            'నాయుడ', 'గోపాల', 'మురళి', 'సురేష', 'రమేష', 'రాజేష', 'మహేష', 'గణేష'
        ]
        
        self.female_name_patterns = [
            'లక్ష్మి', 'సీత', 'గీత', 'రాధ', 'సుష్మ', 'ప్రిය', 'కవిత', 'సునీత', 'మాల', 
            'దేవి', 'కుమారి', 'శ్రీ', 'అనిత', 'వాణి', 'రేణు', 'హేమ', 'రేఖ', 'మంజు'
        ]
        
        # AP Government schemes knowledge base
        self.ap_govt_schemes = {
            'YSR రైతు భరోసా': {
                'department': 'వ్యవసాయ శాఖ',
                'keywords': ['రైతు', 'వ్యవసాయం', 'పంట', 'రైతు భరోసా', 'వ్యవసాయ సహాయం'],
                'amount': '13500 per year'
            },
            'అమ్మ వోడి': {
                'department': 'విద్యా శాఖ',
                'keywords': ['పిల్లలు', 'విద్య', 'స్కూలు', 'అమ్మ వోడి', 'విద్య సహాయం'],
                'amount': '15000 per year'
            },
            'YSR పెన్షన్ కానుక': {
                'department': 'సంక్షేమ శాఖ',
                'keywords': ['పెన్షన్', 'వృద్ధులు', 'వైధవ్య', 'వికలాంగ', 'కానుక'],
                'amount': 'up to 3000 per month'
            },
            'జగన్నాథ అన్న కంటీర్': {
                'department': 'పౌర సరఫరాలు',
                'keywords': ['అన్నం', 'రేషన్', 'ఆహారం', 'పిడిఎస్', 'కంటీర్'],
                'amount': 'free rice'
            },
            'ఆరోగ్యశ్రీ': {
                'department': 'ఆరోగ్య శాఖ',
                'keywords': ['ఆస్పత్రి', 'చికిత్స', 'ఆరోగ్యం', 'వైద్యం', 'ఆరోగ్యశ్రీ'],
                'amount': 'up to 5 lakhs'
            }
        }
        
        # Sector-specific question templates
        self.sector_questions = {
            'వైద్య సేవలు': [
                'మీ గ్రామంలో ప్రాథమిక ఆరోగ్య కేంద్రం ఎలా పనిచేస్తుంది?',
                'వైద్య సిబ్బంది సరిగ్గా వస్తున్నారా?',
                'ఔషధాలు సమయానికి లభిస్తున్నాయా?',
                'అత్యవసర సేవలు ఎలా ఉన్నాయి?'
            ],
            'మౌలిక వసతులు': [
                'మీ గ్రామంలో రోడ్డుల పరిస్థితి ఎలా ఉంది?',
                'తాగునీటి సమస్యలు ఏవైనా ఉన్నాయా?',
                'విద్యుత్ సరఫరా సరిగ్గా ఉందా?',
                'పారిశుబ్రత సేవలు ఎలా ఉన్నాయి?'
            ],
            'విద్యా సేవలు': [
                'మీ గ్రామంలో స్కూలు వసతులు ఎలా ఉన్నాయి?',
                'ఉపాధ్యాయుల హాజరు సరిగ్గా ఉందా?',
                'మధ్యాహ్న భోజన పథకం సరిగ్గా నడుస్తుందా?',
                'పిల్లల రవాణా వసతులు ఎలా ఉన్నాయి?'
            ],
            'సంక్షేమ పథకాలు': [
                'మీకు లభించే పెన్షన్ సమయానికి వస్తుందా?',
                'రేషన్ కార్డ్ సేవలు ఎలా ఉన్నాయి?',
                'ఇతర సంక్షేమ పథకాలలో సమస్యలు ఏవైనా ఉన్నాయా?',
                'అధికారుల వ్యవహారం ఎలా ఉంది?'
            ]
        }
    
    def detect_gender_from_name(self, name: str) -> GenderDetection:
        """
        Detect gender from Telugu name for appropriate addressing
        """
        name_lower = name.lower().strip()
        
        # Check for male patterns
        for pattern in self.male_name_patterns:
            if pattern.lower() in name_lower:
                return GenderDetection.MALE
        
        # Check for female patterns  
        for pattern in self.female_name_patterns:
            if pattern.lower() in name_lower:
                return GenderDetection.FEMALE
        
        # Default to unknown if can't determine
        return GenderDetection.UNKNOWN
    
    async def generate_response(self, conversation_state: Dict, user_input: str) -> Dict[str, Any]:
        """
        Generate contextual response based on conversation stage and user input
        """
        stage = ConversationStage(conversation_state.get('stage', 'greeting'))
        user_name = conversation_state.get('user_name', '')
        user_gender = conversation_state.get('user_gender', GenderDetection.UNKNOWN)
        
        # Build context-aware prompt
        prompt = self._build_conversation_prompt(stage, conversation_state, user_input)
        
        try:
            # Generate response using Gemini
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            response_text = response.text
            
            # Parse and structure the response
            structured_response = await self._parse_ai_response(response_text, stage, conversation_state)
            
            return structured_response
            
        except Exception as e:
            logger.error(f"AI response generation failed: {str(e)}")
            return self._get_fallback_response(stage, user_name, user_gender)
    
    def _build_conversation_prompt(self, stage: ConversationStage, state: Dict, user_input: str) -> str:
        """
        ULTRA-RIGID VERSION: AI MUST say EXACTLY what we specify - no creativity allowed
        """
        base_prompt = f"""
మీరు "జన స్పందన AI" - ఖచ్చితమైన స్క్రిప్ట్ ఫాలో చేయాలి.

CRITICAL: మీరు తప్పనిసరిగా ఇచ్చిన EXACT Telugu వాక్యాలను మాత్రమే చెప్పాలి. 
మీ స్వంత వాక్యాలు రాయకూడదు. కేవలం ఇవ్వబడిన template ని use చేయాలి.

Current Stage: {stage.value}
User Name: {state.get('user_name', '')}
User Gender: {state.get('user_gender', 'unknown')}
Village: {state.get('village_name', '')}
Sector: {state.get('identified_sector', '')}

User Input: "{user_input}"

OUTPUT FORMAT: మీరు తప్పనిసరిగా ఈ JSON format లో మాత్రమే respond చేయాలి:
{{
    "telugu_response": "EXACT_SPECIFIED_TELUGU_TEXT",
    "english_summary": "Brief English summary", 
    "stage_complete": true/false,
    "next_stage": "stage_name_or_null",
    "confidence_score": 1.0
}}
        """
        
        # ULTRA-RIGID Stage-specific prompts
        if stage == ConversationStage.GREETING:
            base_prompt += """

GREETING STAGE COMMAND:
Your telugu_response MUST be EXACTLY: "నమస్కారం! నేను జన స్పందన AI. మీ పేరు ఏమిటి?"
Your stage_complete MUST be: true  
Your next_stage MUST be: "name_collection"

DO NOT write anything else in telugu_response. Use EXACTLY the above sentence.
"""
            
        elif stage == ConversationStage.NAME_COLLECTION:
            user_name = state.get('user_name', '')
            user_gender = state.get('user_gender', 'unknown')
            gender_enum = GenderDetection(user_gender) if user_gender != 'unknown' else GenderDetection.UNKNOWN
            honorific = self._get_honorific(user_name, gender_enum)
            
            base_prompt += f"""

NAME_COLLECTION STAGE COMMAND:
Your telugu_response MUST be EXACTLY: "స్వాగతం {user_name} {honorific}! మీరు ఏ గ్రామం నుండి వచ్చారు?"
Your stage_complete MUST be: true
Your next_stage MUST be: "sector_identification"

DO NOT write anything else in telugu_response. Use EXACTLY the above sentence.
"""
            
        elif stage == ConversationStage.SECTOR_IDENTIFICATION:
            village_name = state.get('village_name', '')
            base_prompt += f"""

SECTOR_IDENTIFICATION STAGE COMMAND:
Your telugu_response MUST be EXACTLY: "మంచిది {village_name}! మీ గ్రామంలో ఈ విషయాలలో ఏవైనా సమస్యలు ఉన్నాయా? 1) వైద్య సేవలు 2) మౌలిక వసతులు 3) విద్యా సేవలు 4) ప్రభుత్వ పథకాలు - ఏ రంగంలో సమస్య ఎక్కువ అనుకుంటున్నారు?"
Your stage_complete MUST be: true
Your next_stage MUST be: "detailed_inquiry"

DO NOT write anything else in telugu_response. Use EXACTLY the above sentence.
"""
            
        elif stage == ConversationStage.DETAILED_INQUIRY:
            sector = state.get('identified_sector', '')
            detailed_question_count = state.get('detailed_question_count', 0)
            
            if sector and sector in self.sector_questions:
                available_questions = self.sector_questions[sector]
                # Use detailed_question_count to select which question to ask (0-based indexing)
                current_question = available_questions[min(detailed_question_count, len(available_questions)-1)]
                
                base_prompt += f"""

DETAILED_INQUIRY STAGE COMMAND:
Your telugu_response MUST be EXACTLY: "{sector} గురించి మాట్లాడుదాం. {current_question} దయచేసి వివరంగా చెప్పండి."
Your stage_complete MUST be: false
Your next_stage MUST be: "detailed_inquiry"

DO NOT write anything else in telugu_response. Use EXACTLY the above sentence.
"""
            else:
                base_prompt += """

DETAILED_INQUIRY STAGE (NO SECTOR) COMMAND:
Your telugu_response MUST be EXACTLY: "దయచేసి మీ సమస్య గురించి వివరంగా చెప్పండి."
Your stage_complete MUST be: false
Your next_stage MUST be: "detailed_inquiry"
"""
            
        elif stage == ConversationStage.CONFIRMATION:
            base_prompt += """

CONFIRMATION STAGE COMMAND:
Your telugu_response MUST be EXACTLY: "మీ సమస్య అర్థమైంది. ఇది సరైన అర్థమేనా? మరేదైనా జోడించాలని అనుకుంటున్నారా?"
Your stage_complete MUST be: true
Your next_stage MUST be: "conclusion"

DO NOT write anything else in telugu_response. Use EXACTLY the above sentence.
"""

        elif stage == ConversationStage.CONCLUSION:
            base_prompt += """

CONCLUSION STAGE COMMAND:
Your telugu_response MUST be EXACTLY: "ధన్యవాదాలు! మీ సమస్యలను సంబంధిత అధికారులకు పంపిస్తాము. త్వరలో పరిష్కారం జరుగుతుంది."
Your stage_complete MUST be: true
Your next_stage MUST be: null

DO NOT write anything else in telugu_response. Use EXACTLY the above sentence.
"""
            
        return base_prompt
    
    async def _parse_ai_response(self, ai_response: str, stage: ConversationStage, state: Dict) -> Dict[str, Any]:
        """
        Parse AI response and structure it for application use
        """
        try:
            # Extract JSON from AI response
            start = ai_response.find('{')
            end = ai_response.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = ai_response[start:end]
                parsed_response = json.loads(json_str)
            else:
                # Fallback parsing
                parsed_response = {
                    "telugu_response": ai_response[:200],
                    "english_summary": "AI response parsing failed",
                    "stage_complete": False,
                    "confidence_score": 0.5
                }
            
            # Add metadata
            parsed_response.update({
                'current_stage': stage.value,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'conversation_id': state.get('conversation_id'),
                'processing_successful': True
            })
            
            return parsed_response
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            return self._get_fallback_response(stage, state.get('user_name', ''), 
                                            state.get('user_gender', GenderDetection.UNKNOWN))
    
    def _get_fallback_response(self, stage: ConversationStage, user_name: str, 
                              user_gender: GenderDetection) -> Dict[str, Any]:
        """
        Provide fallback responses when AI fails
        """
        # Gender-appropriate addressing
        honorific = self._get_honorific(user_name, user_gender)
        
        fallback_responses = {
            ConversationStage.GREETING: f"నమస్కారం! నేను జన స్పందన AI. మీ పేరు ఏమిటి?",
            ConversationStage.NAME_COLLECTION: f"స్వాగతం {user_name} {honorific}! మీరు ఏ గ్రామం నుండి వచ్చారు?",
            ConversationStage.SECTOR_IDENTIFICATION: f"మీకు ఏ విషయంలో సమస్య ఉంది {honorific}?",
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
    
    def _get_honorific(self, name: str, gender: GenderDetection) -> str:
        """
        Get appropriate Telugu honorific based on name and gender
        """
        if not name:
            return "గారూ"
            
        if gender == GenderDetection.MALE:
            return "అన్న"  # Brother (respectful for males)
        elif gender == GenderDetection.FEMALE:
            return "అక్క"  # Sister (respectful for females)
        else:
            return "గారూ"  # Universal respectful term
    
    async def start_new_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Initialize a new conversation session
        """
        initial_state = {
            'conversation_id': conversation_id,
            'stage': ConversationStage.GREETING.value,
            'user_name': '',
            'user_gender': GenderDetection.UNKNOWN.value,
            'village_name': '',
            'identified_sector': '',
            'collected_issues': [],
            'conversation_log': [],
            'start_time': datetime.now(timezone.utc).isoformat(),
            'question_count': 0,
            'detailed_question_count': 0  # Track detailed inquiry questions separately
        }
        
        self.active_conversations[conversation_id] = initial_state
        
        # Generate greeting
        greeting_response = await self.generate_response(initial_state, "")
        
        return {
            'conversation_id': conversation_id,
            'initial_response': greeting_response,
            'session_started': True
        }
    
    async def process_user_input(self, conversation_id: str, user_input: str, 
                                audio_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        FIXED VERSION: Process user input and advance conversation properly
        """
        if conversation_id not in self.active_conversations:
            return {'error': 'Conversation not found', 'success': False}
        
        state = self.active_conversations[conversation_id]
        
        # Update state based on stage and user input
        await self._update_conversation_state(state, user_input)
        
        # Generate AI response
        ai_response = await self.generate_response(state, user_input)
        
        # Log conversation step
        conversation_step = {
            'conversation_id': conversation_id,
            'user_input': user_input,
            'ai_response': ai_response['telugu_response'],
            'stage': state['stage'],
            'question_count': state['question_count'],
            'audio_metadata': audio_metadata,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        state['conversation_log'].append(conversation_step)
        state['question_count'] += 1

        return {
            'success': True,
            'ai_response': ai_response,
            'conversation_state': state,
            'step_logged': True
        }
    
    async def _update_conversation_state(self, state: Dict, user_input: str):
        """
        FIXED VERSION: Update conversation state AND progress stage BEFORE AI response
        """
        current_stage = ConversationStage(state['stage'])
        
        # Store previous stage for tracking
        state['previous_stage'] = current_stage.value
        
        if current_stage == ConversationStage.GREETING and user_input:
            # Extract name from input - handle common Telugu patterns
            user_input_clean = user_input.strip()
            
            # Handle "నా పేరు X" or "నేను X" patterns
            if 'నా పేరు' in user_input_clean:
                name = user_input_clean.split('నా పేరు')[-1].strip()
            elif 'నేను' in user_input_clean:
                name = user_input_clean.split('నేను')[-1].strip()
            elif 'పేరు' in user_input_clean:
                name = user_input_clean.split('పేరు')[-1].strip()
            else:
                name = user_input_clean
            
            state['user_name'] = name
            state['user_gender'] = self.detect_gender_from_name(name).value
            
            # PROGRESS STAGE IMMEDIATELY 
            state['stage'] = ConversationStage.NAME_COLLECTION.value
            
            logger.info(f"Name extracted: {name}, Gender: {state['user_gender']}")
            logger.info(f"Stage progressed: {current_stage.value} → {state['stage']}")
            
        elif current_stage == ConversationStage.NAME_COLLECTION and user_input:
            # Extract village name - handle common Telugu patterns
            user_input_clean = user_input.strip()
            
            # Handle "మా గ్రామం X" or "నేను X నుండి" patterns
            if 'మా గ్రామం' in user_input_clean:
                village = user_input_clean.split('మా గ్రామం')[-1].strip()
            elif 'గ్రామం' in user_input_clean:
                village = user_input_clean.split('గ్రామం')[0].strip()
            elif 'నుండి' in user_input_clean:
                village = user_input_clean.split('నుండి')[0].strip()
            else:
                village = user_input_clean
            
            state['village_name'] = village
            
            # PROGRESS STAGE IMMEDIATELY
            state['stage'] = ConversationStage.SECTOR_IDENTIFICATION.value
            
            logger.info(f"Village extracted: {village}")
            logger.info(f"Stage progressed: {current_stage.value} → {state['stage']}")
            
        elif current_stage == ConversationStage.SECTOR_IDENTIFICATION and user_input:
            # Attempt to identify sector from user input
            identified_sector = await self._identify_sector(user_input)
            if identified_sector:
                state['identified_sector'] = identified_sector
                
                # PROGRESS STAGE IMMEDIATELY
                state['stage'] = ConversationStage.DETAILED_INQUIRY.value
                
                logger.info(f"Sector identified: {identified_sector}")
                logger.info(f"Stage progressed: {current_stage.value} → {state['stage']}")
            
        elif current_stage == ConversationStage.DETAILED_INQUIRY and user_input:
            # Track how many detailed answers we've received (not including the sector choice)
            detailed_question_count = state.get('detailed_question_count', 0)
            
            # Increment count for this detailed answer
            state['detailed_question_count'] = detailed_question_count + 1
            
            # More detailed logging
            logger.info(f"Detailed answer #{state['detailed_question_count']} received")
            
            # Progress to confirmation after receiving 2 detailed answers  
            # (This means user will see: Question 1 → Answer 1 → Question 2 → Answer 2 → Confirmation)
            if state['detailed_question_count'] >= 2:
                state['stage'] = ConversationStage.CONFIRMATION.value
                logger.info(f"Stage progressed: {current_stage.value} → {state['stage']} (after {state['detailed_question_count']} detailed answers)")
            else:
                logger.info(f"Continuing detailed inquiry: received {state['detailed_question_count']}/2 detailed answers, asking follow-up question")
                
        elif current_stage == ConversationStage.CONFIRMATION and user_input:
            # Progress to conclusion
            state['stage'] = ConversationStage.CONCLUSION.value
            logger.info(f"Stage progressed: {current_stage.value} → {state['stage']}")
        
        # Add user input to issues if it contains problem description (substantial input)
        if len(user_input.split()) > 3:  # More than 3 words indicates substantial input
            state['collected_issues'].append({
                'content': user_input,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'stage': state['stage']  # Use the NEW stage
            })
    
    async def _identify_sector(self, user_input: str) -> Optional[str]:
        """
        Identify service sector from user input using keyword matching
        FIXED: Added English and transliterated Telugu keywords
        """
        input_lower = user_input.lower()
        
        sector_keywords = {
            'వైద్య సేవలు': [
                # Telugu
                'వైద్య', 'సేవలు', 'ఆస్పత్రి', 'వైద్యుడు', 'మందులు', 'ఆరోగ్యం', 'చికిత్స', 'దవాఖానా', 'డాక్టర్',
                # English  
                'medical', 'health', 'doctor', 'hospital', 'medicine', 'healthcare', 'treatment', 'clinic',
                # Transliterated
                'vaidya', 'sevalu', 'aspatra', 'aspitri', 'vaidyudu', 'mandulu', 'arogyam', 'chikitsa', 'davakhana'
            ],
            'మౌలిక వసతులు': [
                # Telugu
                'మౌలిక', 'వసతులు', 'రోడ్డు', 'నీరు', 'విద్యుత్', 'రహదారి', 'కరెంట', 'తాగునీరు', 'రోడ్', 'వైఫై',
                # English
                'infrastructure', 'road', 'water', 'electricity', 'power', 'internet', 'wifi', 'transport',
                # Transliterated  
                'maulika', 'vasatulu', 'roddu', 'neeru', 'vidyut', 'rahadari', 'current', 'taguneeru'
            ],
            'విద్యా సేవలు': [
                # Telugu
                'విద్య', 'సేవలు', 'స్కూలు', 'ఉపాధ్యాయుడు', 'విద్య', 'పిల్లలు', 'పాఠశాల', 'టీచర్', 'చదువు',
                # English
                'education', 'school', 'teacher', 'student', 'learning', 'study', 'class', 'children',
                # Transliterated
                'vidya', 'sevalu', 'skulu', 'upadhyayudu', 'pillalu', 'patashala', 'teacher', 'chaduvu'
            ],
            'సంక్షేమ పథకాలు': [
                # Telugu  
                'సంక్షేమ', 'పథకాలు', 'పెన్షన్', 'రేషన్', 'అన్నం', 'సంక్షేమం', 'సహాయం', 'స్కీమ్', 'ప్రభుత్వ',
                # English
                'welfare', 'pension', 'ration', 'scheme', 'government', 'assistance', 'support', 'benefit',
                # Transliterated
                'sankshema', 'pathakalu', 'pension', 'ration', 'annam', 'sahayam', 'scheme', 'prabhutva'
            ]
        }
        
        # Check each sector's keywords
        for sector, keywords in sector_keywords.items():
            for keyword in keywords:
                if keyword in input_lower:
                    return sector
        
        # Additional flexible matching for common variations
        if any(word in input_lower for word in ['medical', 'doctor', 'health', 'vaidya', 'aspatra']):
            return 'వైద్య సేవలు'
        elif any(word in input_lower for word in ['road', 'water', 'electricity', 'infrastructure', 'maulika']):
            return 'మౌలిక వసతులు'  
        elif any(word in input_lower for word in ['education', 'school', 'teacher', 'vidya', 'skulu']):
            return 'విద్యా సేవలు'
        elif any(word in input_lower for word in ['pension', 'ration', 'welfare', 'scheme', 'sankshema']):
            return 'సంక్షేమ పథకాలు'
        
        return None
    
    async def end_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        End conversation and save to database
        """
        if conversation_id not in self.active_conversations:
            return {'error': 'Conversation not found', 'success': False}
        
        state = self.active_conversations[conversation_id]
        
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
            'completion_status': 'completed'
        }
        
        # Save to database
# Save to database using data processor
        try:
            from data_processor import save_conversation_to_database
            
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

        except Exception as e:
            logger.error(f"Failed to save conversation: {str(e)}")
            return {
                'success': False,
                'error': 'Failed to save conversation',
                'summary': conversation_summary
            }

# Global conversation engine instance
jan_spandana = JanSpandanaAI()

# Utility functions for easy import
async def start_conversation(conversation_id: str) -> Dict[str, Any]:
    """Start a new conversation session"""
    return await jan_spandana.start_new_conversation(conversation_id)

async def process_message(conversation_id: str, user_input: str, 
                         audio_metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Process user message and get AI response"""
    return await jan_spandana.process_user_input(conversation_id, user_input, audio_metadata)

async def end_conversation_session(conversation_id: str) -> Dict[str, Any]:
    """End conversation and save data"""
    return await jan_spandana.end_conversation(conversation_id)

if __name__ == "__main__":
    # Test conversation engine
    async def test_conversation():
        print("Testing JanSpandana.AI Conversation Engine...")
        
        # Start new conversation
        conv_id = "test_conversation_001"
        start_result = await start_conversation(conv_id)
        
        if start_result['session_started']:
            print("✅ Conversation started successfully")
            print(f"Initial response: {start_result['initial_response']['telugu_response']}")
        else:
            print("❌ Failed to start conversation")
            return
        
        # Test message processing
        test_messages = [
            "నా పేరు రాము",
            "మా గ్రామం రామారావుపేట",
            "వైద్య సేవలలో సమస్య ఉంది",
            "మందులు రావడం లేదు"
        ]
        
        for message in test_messages:
            print(f"\nUser: {message}")
            response = await process_message(conv_id, message)
            if response['success']:
                print(f"AI: {response['ai_response']['telugu_response']}")
                print(f"Stage: {response['ai_response']['current_stage']}")
                print(f"Stage Complete: {response['ai_response'].get('stage_complete', False)}")
            else:
                print(f"Error: {response}")
        
        # End conversation
        end_result = await end_conversation_session(conv_id)
        if end_result['success']:
            print("\n✅ Conversation ended and saved successfully")
        else:
            print(f"\n❌ Failed to end conversation: {end_result}")
    
    asyncio.run(test_conversation())