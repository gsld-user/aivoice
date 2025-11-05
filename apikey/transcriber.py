
import os
import tempfile
from io import BytesIO
from pydub import AudioSegment
from faster_whisper import WhisperModel
from flask import render_template, send_from_directory
from flask_socketio import emit


device = "cpu"
model = WhisperModel("base", device=device, compute_type="float32")
print(f"🚀 faster-whisper 使用設備：{device}")

last_sentences = []

def register_routes(app, socketio):

    @app.route('/text')
    def text1():
        return render_template('text1.html')

    @app.route('/download/<filename>')
    def download_file_t(filename):
        return send_from_directory(directory="transcriptions", path=filename, as_attachment=True)
    
    @socketio.on("start_recording")
    def handle_start_recording():
        print("🎬 開始 Whisper 錄音")
        # 可以加一些預備處理邏輯（目前可保留空）

    @socketio.on("stop_recording")
    def handle_stop_recording():
        print("🛑 停止 Whisper 錄音")
        # 可以加一些收尾處理邏輯（目前可保留空）

    @socketio.on('audio')
    def handle_audio(data):
        try:
            audio_bytes = BytesIO(data["audio"])
            audio = AudioSegment.from_file(audio_bytes)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_filename = tmp.name
                audio.export(wav_filename, format="wav")
            print(f"🎧 收到音訊：{wav_filename}")

            segments, _ = model.transcribe(
                wav_filename,
                language="zh",
                beam_size=5,
                vad_filter=True,
                
            )

            new_sentences = []
            for seg in segments:
                sentence = seg.text.strip()
                new_sentences.append(sentence)
                last_sentences.append(sentence)

            final_text = new_sentences[-1] if new_sentences else "(無新句子)"
            emit("transcription", {"text": final_text})

            if not os.path.exists("transcriptions"):
                os.makedirs("transcriptions")
            with open(os.path.join("transcriptions", "transcription.txt"), "w", encoding="utf-8") as f:
                f.write(final_text)

            emit("transcription", {
                "text": final_text,
                "download_url": "/download/transcription.txt"
            })

            os.remove(wav_filename)

        except Exception as e:
            print("❌ 發生錯誤：", str(e))
            emit("transcription", {"text": f"(錯誤：{str(e)})"})

    

    @socketio.on("clear_transcription")
    def clear_transcription():
        global last_sentences
        last_sentences = []
        emit("transcription", "")


