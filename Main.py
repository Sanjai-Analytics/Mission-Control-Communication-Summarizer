import torch
import librosa
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Executing on: {device}")

# ==========================================
# 1. Initialize OpenAI Whisper (Transcription)
# ==========================================
processor = AutoProcessor.from_pretrained("openai/whisper-base")
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base").to(device)

def transcribe_audio(audio_path):
    """Converts Mission Control audio logs into text using Whisper."""
    print("Loading audio file...")
    # Whisper expects a sampling rate of exactly 16,000 Hz
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
    
    print("Transcribing audio...")
    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").to(device)
    
    with torch.no_grad():
        predicted_ids = whisper_model.generate(inputs["input_features"])
        
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# ==========================================
# 2. Initialize Google Flan-T5 (Summarization)
# ==========================================
# We use flan-t5-base, which you previously utilized for board summaries
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

def summarize_mission_log(transcription):
    """Generates a highly technical summary of the transcribed mission log."""
    print("Generating mission summary...")
    
    # Flan-T5 performs best when given explicit instructions in the prompt
    prompt = f"Summarize the following aerospace mission control communication, highlighting only critical technical actions and system statuses: {transcription}"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    
    with torch.no_grad():
        summary_ids = t5_model.generate(
            inputs["input_ids"], 
            max_length=150, 
            min_length=30, 
            num_beams=4, # Beam search improves the logical flow of the summary
            early_stopping=True
        )
        
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# ==========================================
# 3. Execute the Full Pipeline
# ==========================================
if __name__ == "__main__":
    audio_file = "apollo_11_descent.wav"     
    try:
        raw_transcript = transcribe_audio(audio_file)
        mission_summary = summarize_mission_log(raw_transcript)
        
        print("\n--- RAW TRANSCRIPT ---")
        print(raw_transcript)
        
        print("\n--- ACTIONABLE MISSION SUMMARY ---")
        print(mission_summary)
        
    except Exception as e:
        print(f"Pipeline Error: {e}")

import urllib.request

# Pulling a highly stable Hugging Face test audio file to bypass the NASA 404 error
url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
file_name = "apollo_11_full_descent.wav" 

print("Downloading test audio file...")
urllib.request.urlretrieve(url, file_name)
print(f"Successfully downloaded test file: {file_name}")
print("You can now re-run the Whisper pipeline cell!")

from transformers import pipeline

# 1. Initialize the robust ASR Pipeline with chunking enabled
whisper_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device=device,
    chunk_length_s=30, 
    batch_size=8,      
)

def transcribe_long_audio(audio_path):
    """Safely transcribes mission control logs of any length."""
    print(f"Processing massive audio file: {audio_path}...")
    
    # The pipeline automatically handles the librosa loading, sampling rate conversion, 
    # chunking, PyTorch tensor routing, and final text stitching.
    result = whisper_pipeline(audio_path)
    
    return result["text"]

# Execution
raw_transcript = transcribe_long_audio("apollo_11_full_descent.wav")
mission_summary = summarize_mission_log(raw_transcript)
