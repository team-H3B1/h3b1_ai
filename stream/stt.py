import whisper

def stt_mp4(audio_file):
    return whisper.speech_to_text(audio_file)