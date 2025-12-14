# Models Directory

This directory stores local AI model files for voice input processing.

## Smart Turn

Place the Smart Turn ONNX model in `smart_turn/` directory:
- `smart_turn/smart-turn-v3.1-gpu.onnx`

You can download the Smart Turn model from the official source and place it here for local inference.

## Parakeet ASR

Parakeet models are downloaded automatically by NeMo on first use and cached in the Hugging Face cache directory.

To pre-download the Parakeet model:
```bash
python -c "import nemo.collections.asr as nemo_asr; nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')"
```

## Model Storage

Large model files (*.onnx, *.pt, *.pth) are excluded from git to keep the repository size small. Please download models separately and place them in the appropriate directories as instructed above.
