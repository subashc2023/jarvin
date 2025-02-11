import logging
import os
import sys
import queue
import threading
import time
from typing import List, Tuple
import numpy as np
import sounddevice as sd
import webrtcvad
import torch
from scipy.io import wavfile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from litellm import completion
import subprocess
import soundfile as sf
from kokoro import KPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if os.environ.get("PYTHONUTF8") != "1":
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)
    
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
SILENCE_THRESHOLD = 10
VAD_MODE = 3
WHISPER_MODEL_ID = "openai/whisper-large-v3"
LLM_MODEL = "groq/llama3-8b-8192"
TTS_SAMPLE_RATE = 24000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STT_DIR = os.path.join(BASE_DIR, "stt")
TTS_DIR = os.path.join(BASE_DIR, "tts")
os.makedirs(STT_DIR, exist_ok=True)
os.makedirs(TTS_DIR, exist_ok=True)

def clear_folder(folder_path: str) -> None:
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error("Failed to delete %s. Reason: %s", file_path, e)

# Clear the STT and TTS folders when starting up
clear_folder(STT_DIR)
clear_folder(TTS_DIR)

def convert_to_pcm16(audio_block: np.ndarray) -> bytes:
    pcm = (audio_block * 32767).astype(np.int16)
    return pcm.tobytes()

def is_speech(vad: webrtcvad.Vad, pcm_bytes: bytes) -> bool:
    return vad.is_speech(pcm_bytes, sample_rate=SAMPLE_RATE)

class VADRecorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE, vad_mode: int = VAD_MODE) -> None:
        self.sample_rate = sample_rate
        self.current_buffer: List[np.ndarray] = []
        self.transcriptions: List[Tuple[int, str]] = []
        self.segment_index: int = 0
        self.recording: bool = True
        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.vad = webrtcvad.Vad(vad_mode)
        self.frame_duration_ms = FRAME_DURATION_MS
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.silence_threshold = SILENCE_THRESHOLD

    def callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if not self.recording:
            return
        self.audio_queue.put(indata.copy())

    def process_audio(self, asr_pipeline: pipeline) -> None:
        silence_count = 0
        while self.recording or not self.audio_queue.empty():
            try:
                block = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self.current_buffer.append(block)
            audio_block = block[:, 0]
            num_frames = len(audio_block) // self.frame_size

            for i in range(num_frames):
                frame = audio_block[i * self.frame_size:(i + 1) * self.frame_size]
                pcm_bytes = convert_to_pcm16(frame)
                if is_speech(self.vad, pcm_bytes):
                    silence_count = 0
                else:
                    silence_count += 1

            if silence_count >= self.silence_threshold and self.current_buffer:
                segment_audio = np.concatenate(self.current_buffer, axis=0)
                filename = os.path.join(STT_DIR, f"segment_{self.segment_index}.wav")
                wavfile.write(filename, self.sample_rate, segment_audio)
                logger.info("Saved segment %d to %s", self.segment_index, filename)

                transcription = transcribe_segment(filename, asr_pipeline)
                self.transcriptions.append((self.segment_index, transcription))
                self.segment_index += 1

                self.current_buffer = []
                silence_count = 0

    def stop(self) -> None:
        self.recording = False

def initialize_whisper() -> pipeline:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info("Initializing Whisper model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "en", "task": "transcribe"}
    )
    logger.info("Whisper model initialized.")
    return asr_pipeline

def transcribe_segment(audio_path: str, asr_pipeline: pipeline) -> str:
    result = asr_pipeline(audio_path)
    transcription = result.get("text", "")
    logger.info("Transcription for %s: %s", audio_path, transcription)
    return transcription

def send_to_deepseek(conversation_history: list) -> str:
    logger.info("Sending conversation history to LLM via litellm")
    response = completion(
        model=LLM_MODEL,
        messages=conversation_history,
        stream=True
    )
    full_response = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end='', flush=True)
            full_response += content
    return full_response

def generate_tts_audio_thread(text: str, tts_done_event: threading.Event) -> None:
    logger.info("Starting TTS generation")
    tts_pipeline = KPipeline(lang_code='a')
    for i, (gs, ps, audio) in enumerate(
        tts_pipeline(text, voice='af_heart', speed=1, split_pattern=r'\n+')
    ):
        output_path = os.path.join(TTS_DIR, f'output_{i}.wav')
        logger.info("Saving TTS segment %d: %s", i, gs)
        sf.write(output_path, audio, TTS_SAMPLE_RATE)
    tts_done_event.set()
    logger.info("TTS generation complete.")

def play_tts_audio_stream(tts_done_event: threading.Event) -> None:
    logger.info("Starting TTS playback thread")
    i = 0
    while True:
        output_path = os.path.join(TTS_DIR, f'output_{i}.wav')
        if os.path.exists(output_path):
            logger.info("Playing %s", output_path)
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-af", "atempo=1.33", output_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            i += 1
        else:
            if tts_done_event.is_set():
                break
            else:
                time.sleep(0.1)
    logger.info("TTS playback complete.")

def main() -> None:
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    os.environ['GROQ_API_KEY'] = groq_api_key

    asr_pipeline = initialize_whisper()

    conversation_history = [
        {"role": "system", "content": "You are a helpful assistant. Be concise and informative."}
    ]

    try:
        while True:
            logger.info("Starting a new recording session.")
            print("Recording session started.")
            print("Press Enter to finish recording, or type 'exit' and press Enter to quit.")
            recorder = VADRecorder(sample_rate=SAMPLE_RATE)
            overall_start_time = time.time()

            # Start the audio recording and processing in a separate thread.
            with sd.InputStream(callback=recorder.callback, channels=1, samplerate=SAMPLE_RATE):
                processing_thread = threading.Thread(target=recorder.process_audio, args=(asr_pipeline,))
                processing_thread.start()
                
                # Wait for user input to end the recording session.
                user_input = input("Recording... (press Enter to finish, or type 'exit' to quit): ")
                if user_input.strip().lower() == "exit":
                    logger.info("Exit command received.")
                    recorder.stop()
                    processing_thread.join(timeout=2)
                    break
                else:
                    recorder.stop()
                    processing_thread.join(timeout=2)

            # Process any leftover audio.
            if recorder.current_buffer:
                segment_audio = np.concatenate(recorder.current_buffer, axis=0)
                filename = os.path.join(STT_DIR, f"segment_{recorder.segment_index}.wav")
                wavfile.write(filename, SAMPLE_RATE, segment_audio)
                transcription = transcribe_segment(filename, asr_pipeline)
                recorder.transcriptions.append((recorder.segment_index, transcription))
                recorder.segment_index += 1

            recorder.transcriptions.sort(key=lambda x: x[0])
            full_transcript = " ".join(trans for idx, trans in recorder.transcriptions)
            total_elapsed = time.time() - overall_start_time
            logger.info("Final Aggregated Transcription: %s", full_transcript)
            logger.info("Total recording/transcription time: %.2f seconds", total_elapsed)

            conversation_history.append({"role": "user", "content": full_transcript})

            groq_response = send_to_deepseek(conversation_history)
            logger.info("\nLLM Response: %s", groq_response)
            conversation_history.append({"role": "assistant", "content": groq_response})

            tts_done_event = threading.Event()
            tts_thread = threading.Thread(target=generate_tts_audio_thread, args=(groq_response, tts_done_event))
            playback_thread = threading.Thread(target=play_tts_audio_stream, args=(tts_done_event,))
            tts_thread.start()
            playback_thread.start()
            tts_thread.join()
            playback_thread.join()

            clear_folder(STT_DIR)
            clear_folder(TTS_DIR)

            logger.info("Cycle complete. Preparing for the next recording session...\n")
    except KeyboardInterrupt:
        logger.info("Exiting the continuous loop.")
    except Exception as e:
        logger.error("An error occurred: %s", e)

if __name__ == "__main__":
    main()
