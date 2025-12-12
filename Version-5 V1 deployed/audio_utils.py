"""
JanSpandana.AI - Audio Processing Utilities
Enhanced audio handling for rural network conditions and multiple formats
Optimized for voice quality improvement and efficient processing
"""

import os
import io
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import tempfile
import hashlib
from datetime import datetime, timezone

# Audio processing libraries
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from pydub.silence import split_on_silence, detect_nonsilent
import soundfile as sf
import numpy as np

# Web and async
import aiofiles
import httpx

# Environment
from dotenv import load_dotenv
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

class AudioProcessor:
    """
    Advanced audio processing for JanSpandana.AI
    Handles format conversion, quality enhancement, and rural network optimization
    """
    
    def __init__(self):
        # Audio processing settings
        self.target_sample_rate = 16000  # Optimal for speech recognition
        self.target_channels = 1  # Mono
        self.target_bitrate = "64k"  # Balanced quality/size for rural networks
        
        # Supported formats
        self.supported_input_formats = [
            'webm', 'ogg', 'mp3', 'wav', 'm4a', 'aac', 'flac', '3gp'
        ]
        self.output_format = 'mp3'
        
        # Quality thresholds
        self.min_duration_ms = 500  # Minimum 0.5 seconds
        self.max_duration_ms = 300000  # Maximum 5 minutes
        self.min_volume_threshold = -60  # dBFS
        
        # Rural network optimization
        self.compression_quality = {
            'low_bandwidth': {'bitrate': '32k', 'quality': 3},
            'medium_bandwidth': {'bitrate': '64k', 'quality': 5},
            'high_bandwidth': {'bitrate': '128k', 'quality': 7}
        }
        
        # Audio cache for processing optimization
        self.audio_cache = {}
        self.cache_max_size = 50  # Maximum cached audio files
    
    async def process_audio_upload(self, audio_data: bytes, 
                                 original_format: str = 'webm',
                                 network_quality: str = 'medium_bandwidth',
                                 enhance_quality: bool = True) -> Dict[str, Any]:
        """
        Main audio processing pipeline for uploaded audio
        """
        try:
            logger.info(f"Processing audio: {len(audio_data)} bytes, format: {original_format}")
            
            # Generate processing ID for tracking
            process_id = hashlib.md5(audio_data[:1024]).hexdigest()[:8]
            
            # Load audio from bytes
            audio_segment = await self._load_audio_from_bytes(audio_data, original_format)
            
            # Validate audio
            validation_result = await self._validate_audio(audio_segment)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'process_id': process_id
                }
            
            # Audio enhancement pipeline
            if enhance_quality:
                audio_segment = await self._enhance_audio_quality(audio_segment)
            
            # Optimize for network conditions
            audio_segment = await self._optimize_for_network(audio_segment, network_quality)
            
            # Convert to target format
            processed_audio_data = await self._export_audio(audio_segment, self.output_format)
            
            # Generate metadata
            metadata = await self._generate_audio_metadata(
                original_audio=audio_data,
                processed_audio=processed_audio_data,
                original_format=original_format,
                audio_segment=audio_segment
            )
            
            return {
                'success': True,
                'process_id': process_id,
                'processed_audio': processed_audio_data,
                'metadata': metadata,
                'format': self.output_format,
                'optimizations_applied': enhance_quality
            }
            
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            return {
                'success': False,
                'error': f"Audio processing error: {str(e)}",
                'process_id': process_id if 'process_id' in locals() else 'unknown'
            }
    
    async def _load_audio_from_bytes(self, audio_data: bytes, format: str) -> AudioSegment:
        """
        Load audio from bytes with format detection and error handling
        """
        try:
            # Create temporary file for complex format handling
            with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            
            try:
                # Try loading with specified format first
                if format.lower() in ['webm', 'ogg']:
                    audio = AudioSegment.from_file(tmp_file_path, format='webm')
                elif format.lower() == 'mp3':
                    audio = AudioSegment.from_mp3(tmp_file_path)
                elif format.lower() == 'wav':
                    audio = AudioSegment.from_wav(tmp_file_path)
                elif format.lower() == 'm4a':
                    audio = AudioSegment.from_file(tmp_file_path, format='m4a')
                else:
                    # Auto-detect format
                    audio = AudioSegment.from_file(tmp_file_path)
                
                return audio
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            logger.error(f"Failed to load audio format {format}: {str(e)}")
            # Fallback: try loading as raw audio
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_data))
                return audio
            except:
                raise Exception(f"Unable to load audio in any supported format: {str(e)}")
    
    async def _validate_audio(self, audio: AudioSegment) -> Dict[str, Any]:
        """
        Validate audio meets minimum requirements
        """
        # Check duration
        duration_ms = len(audio)
        if duration_ms < self.min_duration_ms:
            return {
                'valid': False,
                'error': f"Audio too short: {duration_ms}ms (minimum: {self.min_duration_ms}ms)"
            }
        
        if duration_ms > self.max_duration_ms:
            return {
                'valid': False,
                'error': f"Audio too long: {duration_ms}ms (maximum: {self.max_duration_ms}ms)"
            }
        
        # Check volume levels
        max_volume = audio.max_dBFS
        if max_volume < self.min_volume_threshold:
            return {
                'valid': False,
                'error': f"Audio volume too low: {max_volume}dBFS (minimum: {self.min_volume_threshold}dBFS)"
            }
        
        # Check for silence
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=1000, silence_thresh=-50)
        if not nonsilent_ranges:
            return {
                'valid': False,
                'error': "Audio contains only silence"
            }
        
        return {
            'valid': True,
            'duration_ms': duration_ms,
            'max_volume_dbfs': max_volume,
            'nonsilent_ranges': len(nonsilent_ranges)
        }
    
    async def _enhance_audio_quality(self, audio: AudioSegment) -> AudioSegment:
        """
        Apply audio quality enhancements for better speech recognition
        """
        try:
            # 1. Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # 2. Normalize sample rate
            if audio.frame_rate != self.target_sample_rate:
                audio = audio.set_frame_rate(self.target_sample_rate)
            
            # 3. Normalize volume
            audio = normalize(audio)
            
            # 4. Apply dynamic range compression (helps with varying volumes)
            audio = compress_dynamic_range(audio, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
            
            # 5. High-pass filter to remove low-frequency noise
            audio = audio.high_pass_filter(300)
            
            # 6. Low-pass filter to remove high-frequency noise
            audio = audio.low_pass_filter(3400)  # Human speech range
            
            # 7. Remove leading and trailing silence
            audio = self._trim_silence(audio)
            
            logger.info("Audio quality enhancement completed")
            return audio
            
        except Exception as e:
            logger.warning(f"Audio enhancement failed: {str(e)}, using original audio")
            return audio
    
    def _trim_silence(self, audio: AudioSegment, silence_thresh: int = -50) -> AudioSegment:
        """
        Remove silence from beginning and end of audio
        """
        try:
            # Find non-silent parts
            nonsilent_ranges = detect_nonsilent(
                audio, 
                min_silence_len=100,  # 100ms minimum silence
                silence_thresh=silence_thresh
            )
            
            if nonsilent_ranges:
                start_trim = nonsilent_ranges[0][0]
                end_trim = nonsilent_ranges[-1][1]
                audio = audio[start_trim:end_trim]
            
            return audio
            
        except Exception as e:
            logger.warning(f"Silence trimming failed: {str(e)}")
            return audio
    
    async def _optimize_for_network(self, audio: AudioSegment, network_quality: str) -> AudioSegment:
        """
        Optimize audio for different network conditions
        """
        quality_settings = self.compression_quality.get(
            network_quality, 
            self.compression_quality['medium_bandwidth']
        )
        
        try:
            # Apply compression based on network quality
            if network_quality == 'low_bandwidth':
                # More aggressive compression for poor networks
                audio = audio.set_frame_rate(8000)  # Lower sample rate
                # Additional compression would be applied during export
                
            elif network_quality == 'high_bandwidth':
                # Keep higher quality for good networks
                audio = audio.set_frame_rate(22050)
            
            return audio
            
        except Exception as e:
            logger.warning(f"Network optimization failed: {str(e)}")
            return audio
    
    async def _export_audio(self, audio: AudioSegment, format: str, 
                          bitrate: str = "64k") -> bytes:
        """
        Export audio to bytes in specified format
        """
        try:
            buffer = io.BytesIO()
            
            if format == 'mp3':
                audio.export(buffer, format='mp3', bitrate=bitrate)
            elif format == 'wav':
                audio.export(buffer, format='wav')
            elif format == 'ogg':
                audio.export(buffer, format='ogg')
            else:
                # Default to MP3
                audio.export(buffer, format='mp3', bitrate=bitrate)
            
            buffer.seek(0)
            return buffer.read()
            
        except Exception as e:
            logger.error(f"Audio export failed: {str(e)}")
            raise
    
    async def _generate_audio_metadata(self, original_audio: bytes, 
                                     processed_audio: bytes,
                                     original_format: str,
                                     audio_segment: AudioSegment) -> Dict[str, Any]:
        """
        Generate comprehensive metadata about audio processing
        """
        return {
            'original_size_bytes': len(original_audio),
            'processed_size_bytes': len(processed_audio),
            'compression_ratio': len(original_audio) / len(processed_audio) if len(processed_audio) > 0 else 0,
            'original_format': original_format,
            'processed_format': self.output_format,
            'duration_seconds': len(audio_segment) / 1000.0,
            'sample_rate': audio_segment.frame_rate,
            'channels': audio_segment.channels,
            'max_volume_dbfs': audio_segment.max_dBFS,
            'rms_volume_dbfs': audio_segment.rms,
            'processing_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def batch_process_audio_files(self, audio_files: List[Dict[str, Any]],
                                      network_quality: str = 'medium_bandwidth') -> List[Dict[str, Any]]:
        """
        Process multiple audio files concurrently
        """
        async def process_single_file(file_data):
            return await self.process_audio_upload(
                file_data['audio_data'],
                file_data.get('format', 'webm'),
                network_quality
            )
        
        # Process files concurrently
        tasks = [process_single_file(file_data) for file_data in audio_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'error': str(result),
                    'file_index': i
                })
            else:
                result['file_index'] = i
                processed_results.append(result)
        
        return processed_results
    
    async def analyze_audio_quality(self, audio_data: bytes, 
                                  format: str = 'webm') -> Dict[str, Any]:
        """
        Analyze audio quality without processing
        """
        try:
            audio = await self._load_audio_from_bytes(audio_data, format)
            
            # Calculate various quality metrics
            duration_seconds = len(audio) / 1000.0
            max_volume = audio.max_dBFS
            rms_volume = audio.rms
            
            # Detect silence percentage
            nonsilent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)
            nonsilent_duration = sum(end - start for start, end in nonsilent_ranges)
            silence_percentage = max(0, (len(audio) - nonsilent_duration) / len(audio) * 100)
            
            # Estimate signal-to-noise ratio (simplified)
            signal_power = max_volume
            noise_floor = min(audio.get_array_of_samples()) if len(audio.get_array_of_samples()) > 0 else -60
            estimated_snr = signal_power - noise_floor if noise_floor < signal_power else 0
            
            return {
                'duration_seconds': duration_seconds,
                'max_volume_dbfs': max_volume,
                'rms_volume_dbfs': rms_volume,
                'sample_rate': audio.frame_rate,
                'channels': audio.channels,
                'silence_percentage': silence_percentage,
                'estimated_snr': estimated_snr,
                'quality_score': self._calculate_quality_score(
                    duration_seconds, max_volume, silence_percentage, estimated_snr
                ),
                'recommendations': self._get_quality_recommendations(
                    duration_seconds, max_volume, silence_percentage
                )
            }
            
        except Exception as e:
            logger.error(f"Audio quality analysis failed: {str(e)}")
            return {
                'error': str(e),
                'quality_score': 0.0
            }
    
    def _calculate_quality_score(self, duration: float, max_volume: float, 
                               silence_percent: float, snr: float) -> float:
        """
        Calculate overall audio quality score (0.0 - 1.0)
        """
        score = 1.0
        
        # Duration penalty
        if duration < 1.0:
            score *= 0.5
        elif duration > 120.0:  # Very long audio
            score *= 0.8
        
        # Volume penalty
        if max_volume < -30:  # Very quiet
            score *= 0.6
        elif max_volume > -3:  # Too loud/clipping
            score *= 0.7
        
        # Silence penalty
        if silence_percent > 50:
            score *= 0.4
        elif silence_percent > 25:
            score *= 0.7
        
        # SNR bonus/penalty
        if snr > 20:
            score *= 1.1
        elif snr < 5:
            score *= 0.5
        
        return min(1.0, max(0.0, score))
    
    def _get_quality_recommendations(self, duration: float, max_volume: float, 
                                   silence_percent: float) -> List[str]:
        """
        Get recommendations for improving audio quality
        """
        recommendations = []
        
        if duration < 0.5:
            recommendations.append("రికార్డింగ్ చాలా చిన్నది. మరింత మాట్లాడండి")
        
        if max_volume < -30:
            recommendations.append("మైక్రోఫోన్‌కు దగ్గరగా మాట్లాడండి")
        
        if max_volume > -3:
            recommendations.append("చాలా బిగ్గరగా ఉంది. మరింత మెత్తగా మాట్లాడండి")
        
        if silence_percent > 30:
            recommendations.append("చాలా మౌనం ఉంది. నిరంతరంగా మాట్లాడండి")
        
        if not recommendations:
            recommendations.append("ఆడియో నాణ్యత బాగుంది")
        
        return recommendations

# Global audio processor instance
audio_processor = AudioProcessor()

# Utility functions for easy import
async def process_voice_audio(audio_data: bytes, format: str = 'webm', 
                            network_quality: str = 'medium_bandwidth') -> Dict[str, Any]:
    """Process voice audio with quality enhancement"""
    return await audio_processor.process_audio_upload(audio_data, format, network_quality)

async def analyze_voice_quality(audio_data: bytes, format: str = 'webm') -> Dict[str, Any]:
    """Analyze voice audio quality"""
    return await audio_processor.analyze_audio_quality(audio_data, format)

async def batch_process_voices(audio_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process multiple voice files"""
    return await audio_processor.batch_process_audio_files(audio_files)

if __name__ == "__main__":
    # Test audio processing
    async def test_audio_processing():
        print("Testing JanSpandana.AI Audio Processing...")
        
        # Test with a simple sine wave (mock audio)
        try:
            from pydub.generators import Sine
            
            # Generate test audio (1 second, 440Hz)
            test_audio = Sine(440).to_audio_segment(duration=1000)
            test_audio_bytes = test_audio.export(format='wav').read()
            
            # Process the audio
            result = await process_voice_audio(test_audio_bytes, 'wav')
            
            if result['success']:
                print("✅ Audio processing successful")
                print(f"Original size: {result['metadata']['original_size_bytes']} bytes")
                print(f"Processed size: {result['metadata']['processed_size_bytes']} bytes")
                print(f"Compression ratio: {result['metadata']['compression_ratio']:.2f}")
            else:
                print(f"❌ Audio processing failed: {result['error']}")
            
            # Test quality analysis
            quality_result = await analyze_voice_quality(test_audio_bytes, 'wav')
            print(f"Quality score: {quality_result.get('quality_score', 0):.2f}")
            
        except ImportError:
            print("⚠️ Pydub generators not available, skipping audio test")
        except Exception as e:
            print(f"❌ Audio test failed: {str(e)}")
    
    asyncio.run(test_audio_processing())
