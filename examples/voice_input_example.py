#!/usr/bin/env python3
"""
Example script demonstrating voice input configuration and usage.

This script shows how to configure and use the voice input system.
It does not actually run the system, but demonstrates the API.
"""

import os

# Example: Setting environment variables for voice input
def setup_voice_environment():
    """Configure environment variables for voice input."""
    # Enable voice input
    os.environ["VOICE_ENABLED"] = "true"
    
    # Configure GPU (use GPU 0 for voice, with interface)
    os.environ["VOICE_GPU"] = "0"
    
    # Smart Turn configuration
    os.environ["SMART_TURN_ENABLED"] = "true"
    os.environ["SMART_TURN_ONNX_PATH"] = "models/smart_turn/smart-turn-v3.1-gpu.onnx"
    
    # For single GPU mode (optional)
    # os.environ["WORLD_SIZE"] = "1"


# Example: Manual voice module initialization (for testing)
def test_voice_components():
    """Example of manually initializing voice components."""
    try:
        from core.voice import (
            SmartTurnCuda,
            SmartTurnCudaConfig,
            ParakeetASR,
            ParakeetConfig,
            VoiceSession,
            VoiceSessionConfig,
        )
        
        # Smart Turn configuration
        smart_turn_cfg = SmartTurnCudaConfig(
            onnx_path="models/smart_turn/smart-turn-v3.1-gpu.onnx",
            device_id=0,
            threshold=0.5,
            sample_rate=16000,
        )
        
        # Parakeet ASR configuration
        parakeet_cfg = ParakeetConfig(
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            device="cuda:0",
            use_amp=True,
            sample_rate=16000,
        )
        
        # Voice session configuration
        session_cfg = VoiceSessionConfig(
            sample_rate=16000,
            buffer_max_seconds=30.0,
            turn_check_seconds=2.0,
            enable_smart_turn=True,
        )
        
        print("Voice component configurations created successfully")
        print(f"Smart Turn: {smart_turn_cfg}")
        print(f"Parakeet: {parakeet_cfg}")
        print(f"Session: {session_cfg}")
        
        # Note: Actual initialization requires GPU and models
        # smart_turn = SmartTurnCuda(smart_turn_cfg)
        # parakeet_asr = ParakeetASR(parakeet_cfg)
        # voice_session = VoiceSession(session_cfg, smart_turn, parakeet_asr)
        
    except ImportError as e:
        print(f"Voice modules not available: {e}")


# Example: WebSocket message handling
def example_websocket_messages():
    """Example WebSocket messages for voice input."""
    import json
    
    # Start voice recording
    start_msg = {"type": "voice_start"}
    print("Client sends:", json.dumps(start_msg))
    
    # Client would then send binary PCM16 audio chunks
    # (Not shown here as it's binary data)
    print("Client sends: <binary PCM16 audio chunks>")
    
    # Server response: voice_started
    started_response = {
        "type": "voice_started",
        "message": "Voice recording started",
        "timestamp": 1234567890.123
    }
    print("Server responds:", json.dumps(started_response, indent=2))
    
    # Server sends transcript when detected
    transcript_msg = {
        "type": "voice_transcript",
        "text": "Hello, how are you?",
        "timestamp": 1234567890.456
    }
    print("Server sends:", json.dumps(transcript_msg, indent=2))
    
    # Stop voice recording
    stop_msg = {"type": "voice_stop"}
    print("Client sends:", json.dumps(stop_msg))
    
    # Server response: voice_stopped
    stopped_response = {
        "type": "voice_stopped",
        "message": "Voice recording stopped",
        "timestamp": 1234567890.789
    }
    print("Server responds:", json.dumps(stopped_response, indent=2))


if __name__ == "__main__":
    print("=" * 60)
    print("Voice Input Configuration Example")
    print("=" * 60)
    print()
    
    print("1. Environment Setup:")
    print("-" * 60)
    setup_voice_environment()
    print("Environment variables configured")
    print()
    
    print("2. Component Configuration:")
    print("-" * 60)
    test_voice_components()
    print()
    
    print("3. WebSocket Message Examples:")
    print("-" * 60)
    example_websocket_messages()
    print()
    
    print("=" * 60)
    print("See VOICE_INPUT.md for complete documentation")
    print("=" * 60)
