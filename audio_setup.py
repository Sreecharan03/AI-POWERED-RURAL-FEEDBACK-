"""
JanSpandana.AI - Enhanced Audio Storage & FFmpeg Setup
Handles audio file storage with cloud integration and local fallback
Includes FFmpeg installation and configuration for Windows
"""

import os
import subprocess
import sys
import shutil
import zipfile
import requests
from pathlib import Path
import tempfile
import logging
from typing import Dict, Any, Optional
import asyncio
import json
from datetime import datetime, timezone

# Cloud storage (optional)
try:
    import boto3
    from google.cloud import storage as gcs
    CLOUD_AVAILABLE = True
except ImportError:
    CLOUD_AVAILABLE = False

# Local imports
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class FFmpegSetup:
    """
    FFmpeg installation and configuration for Windows
    """
    
    def __init__(self):
        self.ffmpeg_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        self.ffmpeg_dir = Path("ffmpeg")
        self.ffmpeg_exe = self.ffmpeg_dir / "bin" / "ffmpeg.exe"
        
    def is_ffmpeg_installed(self) -> bool:
        """Check if ffmpeg is available"""
        try:
            # Check system PATH first
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            # Check local installation
            return self.ffmpeg_exe.exists()
    
    def download_ffmpeg(self) -> bool:
        """Download and install ffmpeg for Windows"""
        try:
            if self.is_ffmpeg_installed():
                print("✅ FFmpeg already installed")
                return True
            
            print("📥 Downloading FFmpeg...")
            
            # Create directory
            self.ffmpeg_dir.mkdir(exist_ok=True)
            
            # Download zip file
            response = requests.get(self.ffmpeg_url, stream=True)
            zip_path = self.ffmpeg_dir / "ffmpeg.zip"
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print("📦 Extracting FFmpeg...")
            
            # Extract zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.ffmpeg_dir)
            
            # Find extracted folder and move contents
            extracted_dirs = [d for d in self.ffmpeg_dir.iterdir() if d.is_dir() and d.name.startswith('ffmpeg')]
            if extracted_dirs:
                src_dir = extracted_dirs[0]
                # Move bin folder to correct location
                if (src_dir / "bin").exists():
                    if (self.ffmpeg_dir / "bin").exists():
                        shutil.rmtree(self.ffmpeg_dir / "bin")
                    shutil.move(str(src_dir / "bin"), str(self.ffmpeg_dir / "bin"))
                    shutil.rmtree(src_dir)
            
            # Clean up zip file
            zip_path.unlink()
            
            # Add to PATH for current session
            bin_path = str(self.ffmpeg_dir / "bin")
            if bin_path not in os.environ['PATH']:
                os.environ['PATH'] += os.pathsep + bin_path
            
            if self.ffmpeg_exe.exists():
                print("✅ FFmpeg installed successfully!")
                return True
            else:
                print("❌ FFmpeg installation failed")
                return False
                
        except Exception as e:
            print(f"❌ FFmpeg download failed: {e}")
            return False
    
    def get_ffmpeg_path(self) -> Optional[str]:
        """Get path to ffmpeg executable"""
        if shutil.which('ffmpeg'):
            return 'ffmpeg'  # Available in PATH
        elif self.ffmpeg_exe.exists():
            return str(self.ffmpeg_exe)
        else:
            return None

class AudioStorageManager:
    """
    Enhanced audio storage with local and cloud options
    """
    
    def __init__(self):
        self.base_dir = Path("static/audio")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage configuration
        self.storage_config = {
            'local': True,  # Always enabled
            'cloud': CLOUD_AVAILABLE and os.getenv('CLOUD_STORAGE_ENABLED', 'false').lower() == 'true',
            'max_local_files': int(os.getenv('MAX_LOCAL_AUDIO_FILES', '1000')),
            'cleanup_interval_hours': int(os.getenv('AUDIO_CLEANUP_INTERVAL', '24'))
        }
        
        # Cloud storage clients (if available)
        self.s3_client = None
        self.gcs_client = None
        
        if self.storage_config['cloud']:
            self._initialize_cloud_storage()
    
    def _initialize_cloud_storage(self):
        """Initialize cloud storage clients"""
        try:
            # AWS S3
            if os.getenv('AWS_ACCESS_KEY_ID'):
                self.s3_client = boto3.client('s3')
                logger.info("AWS S3 storage initialized")
            
            # Google Cloud Storage
            if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                self.gcs_client = gcs.Client()
                logger.info("Google Cloud Storage initialized")
                
        except Exception as e:
            logger.warning(f"Cloud storage initialization failed: {e}")
    
    async def save_audio_file(self, audio_data: bytes, filename: str, 
                            metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Save audio file with local and optional cloud storage
        """
        result = {
            'success': False,
            'local_path': None,
            'cloud_url': None,
            'metadata': metadata or {}
        }
        
        try:
            # Save locally
            local_path = self.base_dir / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(local_path, 'wb') as f:
                await f.write(audio_data)
            
            result['local_path'] = str(local_path)
            result['local_url'] = f"/static/audio/{filename}"
            result['success'] = True
            
            logger.info(f"Audio saved locally: {filename}")
            
            # Save to cloud (if enabled)
            if self.storage_config['cloud']:
                cloud_url = await self._upload_to_cloud(audio_data, filename)
                if cloud_url:
                    result['cloud_url'] = cloud_url
            
            # Save metadata
            await self._save_metadata(filename, result['metadata'])
            
            # Cleanup old files if needed
            await self._cleanup_old_files()
            
            return result
            
        except Exception as e:
            logger.error(f"Audio save failed: {e}")
            result['error'] = str(e)
            return result
    
    async def _upload_to_cloud(self, audio_data: bytes, filename: str) -> Optional[str]:
        """Upload audio to cloud storage"""
        try:
            # Try S3 first
            if self.s3_client:
                bucket = os.getenv('AWS_S3_BUCKET', 'janspandana-audio')
                key = f"audio/{datetime.now().strftime('%Y/%m/%d')}/{filename}"
                
                self.s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=audio_data,
                    ContentType='audio/mp3'
                )
                
                return f"https://{bucket}.s3.amazonaws.com/{key}"
            
            # Try Google Cloud Storage
            elif self.gcs_client:
                bucket_name = os.getenv('GCS_BUCKET', 'janspandana-audio')
                bucket = self.gcs_client.bucket(bucket_name)
                blob_name = f"audio/{datetime.now().strftime('%Y/%m/%d')}/{filename}"
                blob = bucket.blob(blob_name)
                
                blob.upload_from_string(audio_data, content_type='audio/mp3')
                
                return f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
            
            return None
            
        except Exception as e:
            logger.warning(f"Cloud upload failed: {e}")
            return None
    
    async def _save_metadata(self, filename: str, metadata: Dict[str, Any]):
        """Save audio metadata to JSON file"""
        try:
            metadata_file = self.base_dir / f"{filename}.meta.json"
            metadata['saved_at'] = datetime.now(timezone.utc).isoformat()
            metadata['filename'] = filename
            
            async with aiofiles.open(metadata_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, ensure_ascii=False, indent=2))
                
        except Exception as e:
            logger.warning(f"Metadata save failed: {e}")
    
    async def _cleanup_old_files(self):
        """Cleanup old audio files if limit exceeded"""
        try:
            audio_files = list(self.base_dir.glob("*.mp3"))
            
            if len(audio_files) > self.storage_config['max_local_files']:
                # Sort by creation time
                audio_files.sort(key=lambda f: f.stat().st_ctime)
                
                # Remove oldest files
                files_to_remove = audio_files[:-self.storage_config['max_local_files']]
                
                for file_path in files_to_remove:
                    file_path.unlink(missing_ok=True)
                    # Also remove metadata file
                    meta_file = file_path.with_suffix('.meta.json')
                    meta_file.unlink(missing_ok=True)
                
                logger.info(f"Cleaned up {len(files_to_remove)} old audio files")
                
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            audio_files = list(self.base_dir.glob("*.mp3"))
            total_size = sum(f.stat().st_size for f in audio_files)
            
            return {
                'total_files': len(audio_files),
                'total_size_mb': total_size / (1024 * 1024),
                'storage_config': self.storage_config,
                'cloud_enabled': self.storage_config['cloud'],
                'local_path': str(self.base_dir)
            }
            
        except Exception as e:
            logger.error(f"Storage stats failed: {e}")
            return {'error': str(e)}

class AudioSystemSetup:
    """
    Complete audio system setup for JanSpandana.AI
    """
    
    def __init__(self):
        self.ffmpeg_setup = FFmpegSetup()
        self.storage_manager = AudioStorageManager()
    
    def setup_complete_system(self) -> Dict[str, Any]:
        """Setup complete audio system"""
        results = {
            'ffmpeg_installed': False,
            'storage_initialized': False,
            'ready_for_production': False
        }
        
        try:
            # Setup FFmpeg
            print("🔧 Setting up JanSpandana.AI Audio System...")
            results['ffmpeg_installed'] = self.ffmpeg_setup.download_ffmpeg()
            
            # Test FFmpeg
            if results['ffmpeg_installed']:
                ffmpeg_path = self.ffmpeg_setup.get_ffmpeg_path()
                print(f"✅ FFmpeg available at: {ffmpeg_path}")
            
            # Initialize storage
            storage_stats = self.storage_manager.get_storage_stats()
            results['storage_initialized'] = 'error' not in storage_stats
            
            if results['storage_initialized']:
                print(f"✅ Audio storage initialized: {storage_stats['local_path']}")
                print(f"   Cloud storage: {'Enabled' if storage_stats['cloud_enabled'] else 'Disabled'}")
            
            # Overall system status
            results['ready_for_production'] = (
                results['ffmpeg_installed'] and 
                results['storage_initialized']
            )
            
            if results['ready_for_production']:
                print("🎉 JanSpandana.AI Audio System ready for production!")
            else:
                print("⚠️ Audio system setup incomplete")
            
            return results
            
        except Exception as e:
            print(f"❌ Audio system setup failed: {e}")
            results['error'] = str(e)
            return results

# Global instances
ffmpeg_setup = FFmpegSetup()
storage_manager = AudioStorageManager()
audio_system = AudioSystemSetup()

# Utility functions
def setup_audio_system():
    """Setup complete audio system"""
    return audio_system.setup_complete_system()

async def save_audio_with_metadata(audio_data: bytes, filename: str, 
                                 conversation_id: str, audio_type: str) -> Dict[str, Any]:
    """Save audio with conversation metadata"""
    metadata = {
        'conversation_id': conversation_id,
        'audio_type': audio_type,
        'size_bytes': len(audio_data),
        'format': 'mp3'
    }
    
    return await storage_manager.save_audio_file(audio_data, filename, metadata)

def get_audio_storage_stats():
    """Get current audio storage statistics"""
    return storage_manager.get_storage_stats()

if __name__ == "__main__":
    # Run complete audio system setup
    print("JanSpandana.AI Audio System Setup")
    print("=" * 40)
    
    setup_results = setup_audio_system()
    
    print("\nSetup Results:")
    for key, value in setup_results.items():
        status = "✅" if value else "❌"
        print(f"  {status} {key}: {value}")
    
    if setup_results.get('ready_for_production'):
        print(f"\n🚀 Run this to test audio processing:")
        print(f"python audio_utils.py")
        
        print(f"\n📂 Audio files will be saved to:")
        storage_stats = get_audio_storage_stats()
        print(f"   {storage_stats.get('local_path', 'Unknown')}")
        
        print(f"\n🎤 Access voice interface at:")
        print(f"   file:///{os.path.abspath('voice_interface.html')}")
    else:
        print(f"\n⚠️ Please resolve setup issues before proceeding")