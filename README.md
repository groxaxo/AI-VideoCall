```
   ___  _____   _   _ _     _            ___      _ _ 
  / _ \|_   _| | | | (_)   | |          / __\__ _| | |
 / /_\ \ | |   | | | | | __| | ___  ___/ /  / _` | | |
 |  _  | | |   | | | | |/ _` |/ _ \/ __/ /__| (_| | | |
 | | | |_| |_  \ \_/ / | (_| |  __/ \__\____/\__,_|_|_|
 \_| |_/\___/   \___/|_|\__,_|\___|\___/
```

# AI VideoCall

**AI VideoCall** is an enhanced WebSocket-based video calling system with advanced features. Originally based on 
**RealVideo** by [Zhipu AI](https://z.ai/blog/realvideo), this fork adds local CUDA-based voice input, improved 
real-time processing, and a more powerful conversational AI experience.

The system leverages **GLM-4.5-AirX** and **GLM-TTS** models to generate audio responses and utilizes autoregressive 
diffusion to generate corresponding video frames in real-time. It now includes optional local voice input with 
Smart Turn end-of-turn detection and Parakeet ASR for speech recognition.

## Example Video


<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/4353a47f-32db-4f07-af68-c7cf4eb9b7ec" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/13a674d7-9d2b-4979-be00-3ba37664252d" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/e8e02325-5e63-4bfe-8ffc-c319cea5fe21" width="100%" controls autoplay loop></video>
      </td>
  </tr>
</table>

## Features

### Core Features
- **Text Input**: Fast and responsive text message input interface.
- **AI Voice Response**: Integrates GLM-4.5-AirX and GLM-TTS models to generate natural voice responses.
- **Lip Sync Video**: Generates real-time conversational video based on any input image and audio.
- **Real-time Communication**: WebSocket-based real-time bidirectional communication with low latency.
- **Custom Avatars**: Upload any image to use as the video avatar.
- **Voice Cloning**: Upload audio samples (3+ seconds) for custom voice cloning.

### Advanced Features (New in This Fork)
- **Local Voice Input** 🎤: Optional CUDA-based speech recognition
  - **Smart Turn v3.x**: GPU-accelerated end-of-turn detection
  - **Parakeet ASR**: High-quality local speech recognition with NeMo
  - **Energy-based VAD**: Voice activity detection with pause detection
  - **No External APIs**: All processing done locally on your GPU
- **Modular Architecture**: Clean code structure with separated concerns
- **Single or Multi-GPU Support**: Runs on 1 or 2 GPUs with automatic workload distribution
- **Production Ready**: Comprehensive error handling, logging, and documentation

## Download

| Model                        | Download Links                                                                                                                                                       |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    RealVideo          | [🤗 Hugging Face](https://huggingface.co/zai-org/RealVideo)<br>[🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/RealVideo)                           |

## Quick Start

### 1. Requirements

- Python 3.10 - 3.12
- pip3
- Modern browser (supporting WebSocket and Web Audio API)

### 2. Install Dependencies

```bash
pip3 install -r requirements.txt
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir-use-symlinks False --local-dir wan_models/Wan2.2-S2V-14B
```

### 3. Configure API Key

Before using, please set the ZAI API key:

```bash
export ZAI_API_KEY="your_actual_api_key_here"
```

and change `config/config.py` line:

```python
PATH_TO_YOUR_MODEL = "zai-org/RealVideo/model.pt"  # Replace with your model path
```

### 4. Start the Service

Specify the number of GPUs you wish to use and run the startup script, at least 2 GPUs (per 80GB, such as H100, H200).

For example:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/run_app.sh
```

One GPU will be used for the VAE service, while the remaining GPUs will be automatically allocated for parallel
computation of the DiT service.

#### Optional: Audio-Only Mode

To disable video generation and use audio-only mode:

```bash
export VIDEO_ENABLED=false
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/run_app.sh
```

This will skip video frame generation and only process audio, reducing GPU memory usage and processing time.

The table below shows reference times (in ms) for DiT to generate one block. If the time is within **500ms**, smooth
real-time generation can be achieved. Numbers in parentheses indicate the time taken with compilation enabled.

| DiT sp size / Denoising steps | 2                         | 4                     |
|-------------------------------|---------------------------|-----------------------|
| 1                             | 563.84 ms (**442.61 ms**) | 943.13 ms (723.06 ms) |
| 2                             | **384.86 ms**             | 655.92 ms (527.11 ms) |
| 4                             | **306.39 ms**             | 513.72 ms (**480.68 ms**) |

### 5. Access the Application

- **Main Page**: http://localhost:8003

## Usage Instructions

1. **Set Avatar and Voice**: Use the file upload button to upload an image to set the avatar, or upload a speech audio
   file longer than 3 seconds for voice cloning.
2. **Connect WebSocket**: Click the "Connect" button to establish the WebSocket connection.
3. **Text Input**: Enter a message in the text box and press Enter or click "Send" to send the message.
4. **Voice Input** (Optional): Enable with `VOICE_ENABLED=true` environment variable. See [VOICE_INPUT.md](VOICE_INPUT.md) for details.
5. **Video Generation** (Optional): Enable with `VIDEO_ENABLED=true` (default). Set to `false` for audio-only mode.
6. **Real-time Response**: The real-time generated video response will be displayed on the left (if video is enabled).

### Voice Input (Advanced Feature)

This system supports optional local CUDA-based voice input using:
- **Smart Turn v3.x**: End-of-turn detection on GPU
- **Parakeet ASR**: Speech recognition with NeMo

**Quick Setup:**
```bash
# Enable voice input
export VOICE_ENABLED=true
export VOICE_GPU=0

# Download Smart Turn ONNX model (optional, see VOICE_INPUT.md)
# Place in models/smart_turn/smart-turn-v3.1-gpu.onnx
```

For complete voice input documentation, configuration options, and WebSocket protocol details, see [VOICE_INPUT.md](VOICE_INPUT.md).

## Technical Highlights

- **Model Integration**: Allows for convenient and quick voice cloning, taking text input to generate audio output.
- **Modular Design**: Clear code structure, easy to maintain and extend.
- **Real-time Performance**: Optimized audio processing and real-time video generation algorithms.

## Acknowledgements

### Original Project
This project is based on **RealVideo** by [**Zhipu AI**](https://z.ai/) (智谱AI).

**RealVideo** was created by the team at Zhipu AI and represents groundbreaking work in real-time conversational AI 
with lip-sync video generation. We are grateful for their innovation and for making this technology available 
to the community.

- **Original Blog Post**: [https://z.ai/blog/realvideo](https://z.ai/blog/realvideo)
- **Hugging Face**: [zai-org/RealVideo](https://huggingface.co/zai-org/RealVideo)
- **ModelScope**: [ZhipuAI/RealVideo](https://modelscope.cn/models/ZhipuAI/RealVideo)
- **License**: Apache License 2.0 (Copyright 2025 Zhipu AI)

### Enhancements in This Fork
This fork adds:
- Local CUDA-based voice input with Smart Turn and Parakeet ASR
- Modular voice processing pipeline
- Enhanced error handling and logging
- Comprehensive documentation
- Single GPU mode support
- Production-ready code structure

### Dependencies
This project utilizes the following open-source libraries and frameworks:

- [Self Forcing](https://github.com/guandeh17/Self-Forcing) - Autoregressive diffusion for video generation
- [NeMo Toolkit](https://github.com/NVIDIA/NeMo) - NVIDIA's toolkit for conversational AI
- [ONNXRuntime](https://onnxruntime.ai/) - Cross-platform inference acceleration
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework for APIs
- [PyTorch](https://pytorch.org/) - Deep learning framework

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. When contributing:
- Follow the existing code style and structure
- Add tests for new features when possible
- Update documentation to reflect your changes
- Ensure backward compatibility

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

**Copyright 2025 Zhipu AI** (Original RealVideo)

## Support

For issues related to:
- **Original RealVideo**: Visit [z.ai](https://z.ai) or the official repositories
- **This Fork**: Open an issue on this repository's GitHub page

## Citation

If you use this project in your research or application, please cite the original RealVideo work by Zhipu AI.

---

**Made with ❤️ by the community** | **Originally created by Zhipu AI**
