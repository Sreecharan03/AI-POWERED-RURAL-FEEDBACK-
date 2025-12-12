"""
JanSpandana.AI - Google Cloud Speech Services Integration
Handles Telugu voice recognition (STT) and synthesis (TTS)
Optimized for rural Andhra Pradesh dialects and network conditions
"""

import os
import asyncio
import io
import logging
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import json
import time

# Google Cloud imports
from google.cloud import speech
from google.cloud import texttospeech
from google.oauth2 import service_account
import google.api_core.exceptions as google_exceptions

# Audio processing
from pydub import AudioSegment
from pydub.effects import normalize
import soundfile as sf
import numpy as np

# Environment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure pydub to use explicit FFmpeg/ffprobe paths if provided
FFMPEG_BINARY = os.getenv("FFMPEG_BINARY")
FFPROBE_BINARY = os.getenv("FFPROBE_BINARY")
if FFMPEG_BINARY:
    AudioSegment.converter = FFMPEG_BINARY
if FFPROBE_BINARY:
    AudioSegment.ffprobe = FFPROBE_BINARY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeluguSpeechProcessor:
    """
    Main class for handling Telugu speech processing
    Includes both Speech-to-Text and Text-to-Speech capabilities
    """
    
    def __init__(self):
        self.credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        self.project_id = os.getenv('GCP_PROJECT_ID')
        
        # Initialize clients
        self.speech_client = None
        self.tts_client = None
        self.credentials = None
        
        self._initialize_clients()
        
        # Telugu language configurations
        self.telugu_language_code = 'te-IN'
        self.fallback_language_code = 'hi-IN'  # Hindi as fallback
        self.english_language_code = 'en-IN'
        
        # Audio processing settings
        self.sample_rate = 16000  # Optimal for speech recognition
        self.audio_channel_count = 1  # Mono audio
        self.max_audio_duration = 300  # 5 minutes max
        
        # Regional dialect configurations
        self.ap_dialects = {
            'coastal': ['visakhapatnam', 'vijayawada', 'guntur'],
            'rayalaseema': ['tirupati', 'kadapa', 'anantapur'],
            'telangana_border': ['hyderabad_adjacent', 'warangal_border']
        }
    
    def _initialize_clients(self):
        """Initialize Google Cloud Speech and TTS clients"""
        try:
            if self.credentials_path and os.path.exists(self.credentials_path):
                self.credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                
                # Initialize Speech-to-Text client
                self.speech_client = speech.SpeechClient(credentials=self.credentials)
                
                # Initialize Text-to-Speech client
                self.tts_client = texttospeech.TextToSpeechClient(credentials=self.credentials)
                
                logger.info("Google Cloud Speech clients initialized successfully")
            else:
                logger.error(f"Credentials file not found: {self.credentials_path}")
                raise FileNotFoundError("Google Cloud credentials not found")
                
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud clients: {str(e)}")
            raise
    
    async def preprocess_audio(self, audio_data: bytes, source_format: str = 'webm') -> Tuple[bytes, Dict]:
        """
        Preprocess audio for optimal speech recognition
        Handles various input formats and rural audio quality issues
        """
        try:
            # Load audio using pydub
            if source_format.lower() in ['webm', 'ogg']:
                audio = AudioSegment.from_file(io.BytesIO(audio_data), format='webm')
            elif source_format.lower() == 'mp3':
                audio = AudioSegment.from_file(io.BytesIO(audio_data), format='mp3')
            elif source_format.lower() == 'wav':
                audio = AudioSegment.from_wav(io.BytesIO(audio_data))
            else:
                # Try to auto-detect format
                audio = AudioSegment.from_file(io.BytesIO(audio_data))
            
            # Audio quality improvements for rural recordings
            # 1. Convert to mono
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # 2. Set optimal sample rate
            audio = audio.set_frame_rate(self.sample_rate)
            
            # 3. Normalize volume (helps with varying microphone levels)
            audio = normalize(audio)
            
            # 4. Reduce background noise (basic)
            audio = audio.high_pass_filter(300).low_pass_filter(3000)
            
            # 5. Convert to WAV format for Google Speech
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format='wav')
            processed_audio = wav_buffer.getvalue()
            
            # Audio metadata
            metadata = {
                'duration_seconds': len(audio) / 1000.0,
                'sample_rate': audio.frame_rate,
                'channels': audio.channels,
                'file_size_bytes': len(processed_audio),
                'original_format': source_format
            }
            
            logger.info(f"Audio preprocessed: {metadata}")
            return processed_audio, metadata
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            raise
    
    async def speech_to_text(self, audio_data: bytes, source_format: str = 'webm', 
                           region_hint: str = 'coastal') -> Dict[str, Any]:
        """
        Convert Telugu speech to text with regional dialect support
        """
        try:
            # Preprocess audio
            processed_audio, audio_metadata = await self.preprocess_audio(audio_data, source_format)
            
            # Configure recognition
            audio = speech.RecognitionAudio(content=processed_audio)
            
            # Enhanced configuration for Telugu
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code=self.telugu_language_code,
                alternative_language_codes=[self.fallback_language_code, self.english_language_code],
                
                # Enhanced recognition features
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
                enable_word_time_offsets=True,
                
                # Multiple candidate results
                max_alternatives=3,
                
                # Audio adaptation for rural conditions
                adaptation={
                    'phrase_sets': [
                        speech.PhraseSet(phrases=['గ్రామం', 'సర్పంచ్', 'కలెక్టర్', 'పంచాయతీ', 'పెన్షన్'])
                    ]
                } if hasattr(speech, 'PhraseSet') else None,
                
                # Model selection
                model='latest_long',  # Better for longer audio
                use_enhanced=True  # Enhanced model for better accuracy
            )
            
            # Perform speech recognition
            start_time = time.time()
            response = self.speech_client.recognize(config=config, audio=audio)
            processing_time = time.time() - start_time
            
            # Process results
            results = []
            best_transcript = ""
            best_confidence = 0.0
            
            for result in response.results:
                for alternative in result.alternatives:
                    transcript = alternative.transcript
                    confidence = alternative.confidence
                    
                    # Word-level details
                    words_info = []
                    if hasattr(alternative, 'words'):
                        for word in alternative.words:
                            words_info.append({
                                'word': word.word,
                                'start_time': word.start_time.total_seconds() if word.start_time else 0,
                                'end_time': word.end_time.total_seconds() if word.end_time else 0,
                                'confidence': getattr(word, 'confidence', 0.0)
                            })
                    
                    result_data = {
                        'transcript': transcript,
                        'confidence': confidence,
                        'words': words_info,
                        'language_detected': self.telugu_language_code
                    }
                    results.append(result_data)
                    
                    # Track best result
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_transcript = transcript
            
            # Compile final result
            recognition_result = {
                'success': True,
                'best_transcript': best_transcript,
                'best_confidence': best_confidence,
                'all_alternatives': results,
                'audio_metadata': audio_metadata,
                'processing_time_seconds': processing_time,
                'language_code': self.telugu_language_code,
                'region_hint': region_hint,
                'total_alternatives': len(results)
            }
            
            logger.info(f"Speech-to-text completed. Confidence: {best_confidence:.2f}")
            return recognition_result
            
        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Google API error in STT: {str(e)}")
            return {
                'success': False,
                'error': f"Google API error: {str(e)}",
                'error_type': 'api_error'
            }
        except Exception as e:
            logger.error(f"Speech-to-text error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'processing_error'
            }
    
    async def text_to_speech(self, text: str, voice_gender: str = 'female', 
                           speaking_rate: float = 0.9) -> Dict[str, Any]:
        """
        Convert Telugu text to natural speech
        Optimized for rural acceptance and clarity
        """
        try:
            # Input validation
            if not text or not text.strip():
                raise ValueError("Text input is required")
            
            # Text preprocessing for better TTS
            processed_text = self._preprocess_text_for_tts(text)
            
            # Voice selection based on gender and regional preference
            if voice_gender.lower() == 'female':
                voice_name = 'te-IN-Standard-A'  # Female Telugu voice
            else:
                voice_name = 'te-IN-Standard-B'  # Male Telugu voice (if available)
            
            # TTS Configuration
            synthesis_input = texttospeech.SynthesisInput(text=processed_text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=self.telugu_language_code,
                name=voice_name,
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE if voice_gender.lower() == 'female' 
                           else texttospeech.SsmlVoiceGender.MALE
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speaking_rate,  # Slightly slower for rural understanding
                pitch=0.0,
                volume_gain_db=2.0,  # Slightly louder for mobile speakers
                sample_rate_hertz=24000,  # Good quality for mobile playback
                effects_profile_id=['telephony-class-application']  # Optimized for phone speakers
            )
            
            # Perform text-to-speech
            start_time = time.time()
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            processing_time = time.time() - start_time
            
            # Audio post-processing for mobile optimization
            audio_content = response.audio_content
            
            # Create result
            tts_result = {
                'success': True,
                'audio_content': audio_content,
                'audio_format': 'mp3',
                'text_processed': processed_text,
                'voice_used': voice_name,
                'speaking_rate': speaking_rate,
                'processing_time_seconds': processing_time,
                'audio_size_bytes': len(audio_content),
                'language_code': self.telugu_language_code
            }
            
            logger.info(f"Text-to-speech completed. Audio size: {len(audio_content)} bytes")
            return tts_result
            
        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Google API error in TTS: {str(e)}")
            return {
                'success': False,
                'error': f"Google API error: {str(e)}",
                'error_type': 'api_error'
            }
        except Exception as e:
            logger.error(f"Text-to-speech error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'processing_error'
            }
    
    def _preprocess_text_for_tts(self, text: str) -> str:
        """
        Preprocess Telugu text for better TTS output
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Add pauses for better pronunciation
        text = text.replace('।', ', ')  # Replace devanagari periods with commas for natural pauses
        text = text.replace('.', ', ')  # Replace periods with commas
        
        # Handle common Telugu pronunciation issues
        # Add more preprocessing rules as needed
        
        return text
    
    async def batch_process_audio_files(self, audio_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple audio files concurrently
        Useful for bulk grievance processing
        """
        tasks = []
        for audio_info in audio_files:
            task = self.speech_to_text(
                audio_info['audio_data'],
                audio_info.get('format', 'webm'),
                audio_info.get('region_hint', 'coastal')
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'file_index': i,
                    'success': False,
                    'error': str(result)
                })
            else:
                result['file_index'] = i
                processed_results.append(result)
        
        return processed_results
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check the health of speech services
        """
        health = {
            'speech_client': False,
            'tts_client': False,
            'credentials': False
        }
        
        try:
            # Check credentials
            if self.credentials and self.credentials_path and os.path.exists(self.credentials_path):
                health['credentials'] = True
            
            # Test speech client with minimal audio
            if self.speech_client:
                # Create a minimal test audio (silence)
                test_audio = AudioSegment.silent(duration=100)  # 100ms silence
                wav_buffer = io.BytesIO()
                test_audio.set_frame_rate(16000).set_channels(1).export(wav_buffer, format='wav')
                
                audio = speech.RecognitionAudio(content=wav_buffer.getvalue())
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code=self.telugu_language_code
                )
                
                # This should not fail (though might return empty results)
                response = self.speech_client.recognize(config=config, audio=audio)
                health['speech_client'] = True
            
            # Test TTS client
            if self.tts_client:
                synthesis_input = texttospeech.SynthesisInput(text="టెస్ట్")
                voice = texttospeech.VoiceSelectionParams(
                    language_code=self.telugu_language_code,
                    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )
                
                response = self.tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
                
                if response.audio_content:
                    health['tts_client'] = True
        
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
        
        return health

# Global speech processor instance
speech_processor = TeluguSpeechProcessor()

# Utility functions for easy import
async def process_voice_to_text(audio_data: bytes, format: str = 'webm') -> Dict[str, Any]:
    """Simplified function for voice to text conversion"""
    return await speech_processor.speech_to_text(audio_data, format)

async def process_text_to_voice(text: str, gender: str = 'female') -> Dict[str, Any]:
    """Simplified function for text to voice conversion"""
    return await speech_processor.text_to_speech(text, gender)

async def check_speech_services_health() -> Dict[str, bool]:
    """Check if speech services are working"""
    return await speech_processor.health_check()

if __name__ == "__main__":
    # Test the speech services
    async def test_speech_services():
        print("Testing JanSpandana.AI Speech Services...")
        
        # Health check
        health = await check_speech_services_health()
        print("Speech Services Health:")
        for service, status in health.items():
            print(f"  {service}: {'✅' if status else '❌'}")
        
        # Test TTS
        if health['tts_client']:
            print("\nTesting Text-to-Speech...")
            tts_result = await process_text_to_voice("నమస్కారం, జన్ స్పందన AI కి స్వాగతం")
            if tts_result['success']:
                print(f"✅ TTS successful: {tts_result['audio_size_bytes']} bytes audio generated")
            else:
                print(f"❌ TTS failed: {tts_result['error']}")
    
    asyncio.run(test_speech_services())
