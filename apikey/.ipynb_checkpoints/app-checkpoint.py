from flask import Flask, render_template, redirect, request, session, flash, url_for, send_file,send_from_directory, jsonify, abort
import os
import secrets
import time
import tempfile
import whisper
import whisperx
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'

USER_FILE = "users.txt"  # 使用者資料存儲檔案

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
TRANSCRIBED_FILES_DIR = "transcribed_files"
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a'}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIBED_FILES_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# 設定 WhisperX
device = "cpu"  
batch_size = 16
compute_type = "int8"

# 載入 WhisperX 模型
model = whisperx.load_model("base", device, compute_type=compute_type)
diarize_model = whisperx.DiarizationPipeline(device=device, use_auth_token="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")

# 儲存使用者到記事本
def save_user_to_file(username, password, api_key=None):
    with open(USER_FILE, "a", encoding="utf-8") as file:
        if api_key:
            file.write(f"{username},{password},{api_key}\n")
        else:
            file.write(f"{username},{password}\n")

# 從記事本讀取使用者資料
def load_users():
    users = {}
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    username, password = parts
                    users[username] = {"password": password, "api_key": None}
                elif len(parts) == 3:
                    username, password, api_key = parts
                    users[username] = {"password": password, "api_key": api_key}
    return users


# 註冊頁面
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        users = load_users()

        if username in users:
            flash("使用者名稱已存在！", "danger")
        else:
            save_user_to_file(username, password)
            flash("註冊成功！請登入", "success")
            return redirect("/login")

    return render_template("register.html")

# 登入頁面
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        users = load_users()

        if username not in users:
            flash("您未註冊帳號！", "danger")  # 若帳號不存在
        elif users[username]["password"] == password:
            session["username"] = username
            return redirect("/apikey")  # 登入成功後跳轉到首頁
        else:
            flash("帳號或密碼錯誤！", "danger")

    return render_template("login.html")

# 首頁（登入後顯示）
@app.route("/")
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))  # 如果未登入，跳回登入頁面

    return render_template('apikey.html')

@app.route("/apikey")
def apikey():
    if 'username' not in session:
        flash("請先登入！", "danger")  # 提示未登入
        return redirect(url_for('login'))  # 如果未登入，跳回登入頁面

    return render_template("apikey.html")  # 顯示 API Key 頁面

def read_api_key(username):
    if not os.path.exists(USER_FILE):
        return "尚未產生 API Key"
    
    with open(USER_FILE, "r") as file:
        for line in file:
            data = line.strip().split(",")  # 根據「,」拆分
            if len(data) == 3:  # 確保有 username, password, apikey
                user, password, key = data
                if user == username:
                    return key  # 回傳 API Key

    return "尚未產生 API Key"

@app.route('/generate_apikey', methods=['POST'])
def generate_apikey():
    if 'username' not in session:
        return "請先登入", 401

    username = session['username']
    new_key = secrets.token_hex(8)  # 生成 16 字節的 API Key

    # 讀取所有使用者 API Key
    users = []
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as file:
            for line in file:
                data = line.strip().split(",")
                if len(data) == 3:
                    users.append(data)

    # 更新 API Key
    for user in users:
        if user[0] == username:
            user[2] = new_key  # 更新 API Key

    # 寫回 `users.txt`
    with open(USER_FILE, "w") as file:
        for user in users:
            file.write(",".join(user) + "\n")

    return new_key

# 取得目前的 API Key
@app.route('/get_apikey', methods=['GET'])
def get_apikey():
    if 'username' not in session:
        return "請先登入", 401

    username = session['username']
    return read_api_key(username)

# 檢查文件是否為允許的格式
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 使用 Whisper 轉錄音檔
def transcribe_audio(audio_path, language, model_name):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, language=language)
    return result

@app.route('/voicetotext', methods=['GET', 'POST'])
def index():
    transcript = None  # 存放轉錄結果
    output_filename = None  # 儲存轉錄檔案名稱
    error_message = None  # 用於錯誤提示
    transcription_time = None  # 轉錄所需時間

    if request.method == 'POST':
        if 'audio_file' not in request.files or request.files['audio_file'].filename == "":
            error_message = "請選擇音訊檔案"
        else:
            # 取得表單數據
            audio_file = request.files['audio_file']

            ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a'}
            def allowed_file(filename):
                return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

            # 檢查音訊格式
        if not allowed_file(audio_file.filename):
                error_message = "不支援的音訊格式，請上傳 mp3、m4a 或 wav 檔案"
        else:

            language = request.form.get('language')  # 預設語言為英文
            model_name = request.form.get('model')  # 預設 Whisper base 模型
            output_format = request.form.get('output_format')  # 預設輸出格式為 txt

            # 儲存音訊檔案
            audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
            audio_file.save(audio_path)

            # 轉錄音訊並計算時間
            start_time = time.time()  # 開始計時
            transcript, segments = process_audio(audio_path, language, model_name)
            end_time = time.time()  # 結束計時

            transcription_time = end_time - start_time  # 計算所需時間

            if transcript:
                # 生成對應格式的輸出檔案
                output_filename = create_output_file(audio_file.filename, transcript, segments, output_format)
            else:
                error_message = "轉錄失敗，請檢查音檔"

    return render_template('index.html', transcript=transcript, error=error_message, output_file=output_filename, transcription_time=transcription_time)

def process_audio(audio_path, language, model_name):
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path, language=language)
        return result["text"], result["segments"]
    except Exception as e:
        print(f"轉錄時發生錯誤: {e}")
        return None, None

def create_output_file(filename, transcript, segments, output_format):
    """根據用戶選擇的格式生成對應的檔案"""
    output_filename = f"{os.path.splitext(filename)[0]}.{output_format}"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    if output_format == "txt":
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript)

    elif output_format == "srt":
        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, start=1):
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                f.write(f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n\n")

    elif output_format == "tsv":
        with open(output_path, "w", encoding="utf-8") as f:
            for segment in segments:
                f.write(f"{segment['start']}\t{segment['end']}\t{segment['text']}\n")

    return output_filename  # 只回傳檔名，不回傳完整路徑

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

@app.route('/download/<filename>')
def download_file(filename):
    """提供下載轉錄後的檔案"""
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    return send_file(file_path, as_attachment=True)

@app.route("/meeting", methods=["GET", "POST"])
def meeting():
    table_html = None
    output_file = None
    transcription_time = None  # 新增變數來儲存轉錄時間
    error_message = None  # 儲存錯誤訊息


    if request.method == "POST":
        if "audio_file" not in request.files or request.files["audio_file"].filename == "":
            error_message = "請選擇音訊檔案"
        else:
            file = request.files["audio_file"]

            # 允許的音訊格式
            ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a"}
            def allowed_file(filename):
                return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

            # 檢查音訊格式
            if not allowed_file(file.filename):
                error_message = "不支援的音訊格式，請上傳 mp3、m4a 或 wav 檔案"
            else:
                # 儲存檔案
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                
            # 記錄轉錄開始時間
            start_time = time.time()

            # 進行轉錄
            audio = whisperx.load_audio(file_path)
            result = model.transcribe(audio, batch_size=batch_size)
            transcription_df = pd.DataFrame(result["segments"])

            # 說話者辨識
            diarize_segments = diarize_model(file_path)
            speaker_df = pd.DataFrame(diarize_segments)

            # 初始化 speaker 欄位
            transcription_df["speaker"] = "Unknown"

            time_tolerance = 1.0
            for i, row in transcription_df.iterrows():
                best_match = None
                best_overlap = 0

                for _, speaker_row in speaker_df.iterrows():
                    speaker_start = speaker_row["start"] - time_tolerance
                    speaker_end = speaker_row["end"] + time_tolerance
                    overlap = min(row["end"], speaker_end) - max(row["start"], speaker_start)

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = speaker_row["speaker"]

                if best_match:
                    transcription_df.at[i, "speaker"] = best_match

                transcription_df = transcription_df[['speaker', 'start', 'end' ,'text']]

            # 下載檔案資訊:說話者與文字
            text_output = "\n".join([f"{row['speaker']}: {row['text']}" for _, row in transcription_df.iterrows()])

            # 存成 txt 檔
            output_file = f"{file.filename}.txt"  # 儲存檔案的相對路徑
            output_path = os.path.join(UPLOAD_FOLDER, output_file)  # 取得正確路徑
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text_output)

            # 轉成 HTML 表格顯示在前端
            table_html = transcription_df.to_html(classes="table table-bordered", index=False)

            # 記錄轉錄結束時間
            end_time = time.time()
            transcription_time = round(end_time - start_time, 2)  # 計算並四捨五入到小數點後兩位

    return render_template("meeting.html", transcription=table_html, output_file=output_file, transcription_time=transcription_time)


@app.route('/download_meeting/<filename>', methods=['GET'])
def download_meeting(filename):
    # 確保這裡的路徑是正確的
    file_path = os.path.join(UPLOAD_FOLDER, filename)  # 使用正確的 uploads 路徑
    file_path = os.path.normpath(file_path)  # 標準化路徑

    print(f"下載檔案的路徑: {file_path}")
    
    if os.path.exists(file_path):
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
    else:
        flash("檔案不存在！", "danger")
        return redirect(url_for('meeting'))



# 儲存轉錄結果
def save_transcription(result, filename, output_format):
    output_path = os.path.join("transcriptions", f"{filename}.{output_format}")
    transcription_df = pd.DataFrame(result["segments"])
    
    if output_format == "csv":
        transcription_df.to_csv(output_path, index=False)
    elif output_format == "txt":
        with open(output_path, "w") as f:
            for segment in result["segments"]:
                f.write(f"{segment['start']} - {segment['end']} : {segment['text']}\n")

    # 加入輸出檔案路徑的 print
    print(f"儲存轉錄檔案的路徑: {output_path}")

    return output_path




@app.route("/transcribe", methods=["GET", "POST"])
def transcribe():
    if request.method == "GET":
        return render_template("transcribe.html")

    if 'audio_file' not in request.files:
        return jsonify({"error": "未找到音檔"}), 400

    file = request.files['audio_file']

    if file.filename == '':
        return jsonify({"error": "請選擇一個音檔"}), 400

    filename = "recorded_audio.wav" if file.filename == "blob" else secure_filename(file.filename)

    if allowed_file(filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 檢查是否為靜音音檔（簡單的判斷方法，可以依照需要優化）
        if is_silent(filepath):
            return jsonify({"error": "錄製的音檔為靜音，請重新錄音"}), 400

        language = request.form.get('language', 'zh')
        model_name = request.form.get('model', 'base')

        transcription = transcribe_audio(filepath, language, model_name)

        output_format = request.form.get('output_format', 'txt')
        output_file = save_transcription(transcription, filename.rsplit('.', 1)[0], output_format)

        return send_file(output_file, as_attachment=True)

    return jsonify({"error": "無效的音檔格式"}), 400

# 檢查是否為靜音音檔
def is_silent(filepath):
    import wave
    import numpy as np

    with wave.open(filepath, 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)

    # 簡單的靜音檢查：若音訊的最大振幅過低，視為靜音
    return np.max(np.abs(audio_data)) < 500

@app.route("/text", methods=["GET", "POST"])
def text():
    if request.method == "GET":
        return render_template("text.html")

    if 'audio_file' not in request.files:
        return jsonify({"error": "未找到音檔"}), 400

    file = request.files['audio_file']

    if file.filename == '':
        return jsonify({"error": "請選擇一個音檔"}), 400

    filename = "recorded_audio.wav" if file.filename == "blob" else secure_filename(file.filename)

    if allowed_file(filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 檢查是否為靜音音檔（簡單的判斷方法，可以依照需要優化）
        if is_silent(filepath):
            return jsonify({"error": "錄製的音檔為靜音，請重新錄音"}), 400

        language = request.form.get('language', 'zh')
        model_name = request.form.get('model', 'base')

        transcription = transcribe_audio(filepath, language, model_name)

        output_format = request.form.get('output_format', 'txt')
        output_file = save_transcription(transcription, filename.rsplit('.', 1)[0], output_format)

        return send_file(output_file, as_attachment=True)

    return jsonify({"error": "無效的音檔格式"}), 400

# 檢查是否為靜音音檔
def is_silent(filepath):
    import wave
    import numpy as np

    with wave.open(filepath, 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)

    # 簡單的靜音檢查：若音訊的最大振幅過低，視為靜音
    return np.max(np.abs(audio_data)) < 500


@app.route('/latest_file', methods=['GET'])
def get_latest_file():
    # 目標資料夾
    folder = 'outputs'
    
    # 取得資料夾中的所有檔案
    files = os.listdir(folder)
    # 排除不是檔案的項目
    files = [f for f in files if os.path.isfile(os.path.join(folder, f))]
    
    if not files:
        return jsonify({"error": "No files found"}), 404
    
    # 根據檔案的修改時間來找出最新的檔案
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(folder, f)))
    
    return jsonify({"filename": latest_file})

@app.route('/transcribe', methods=['POST'])
def api_transcribe():
    if 'username' not in session:
        return jsonify({"error": "未登入！"}), 401  # 使用者未登入

    # 檢查是否有上傳音檔
    file = request.files.get('audio_file')
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "無效的音檔格式或未上傳音檔"}), 400

    # 儲存音檔
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 取得語言與模型選擇
    language = request.form.get('language', 'en')
    model_name = request.form.get('model', 'base')

    # 進行音檔轉錄
    transcription = transcribe_audio(filepath, language, model_name)

    # 取得所選的檔案格式
    output_format = request.form.get('output_format', 'txt')
    output_file = save_transcription(transcription, filename.rsplit('.', 1)[0], output_format)

    return send_file(output_file, as_attachment=True)  # 直接返回轉錄後的檔案

# 上傳音檔並轉錄，無需表單，直接在 /upload 完成
@app.route("/upload", methods=["POST"])
def upload():
    # 檢查音檔是否存在
    file = request.files.get('audio_file')  # 從 POST 請求中提取音檔
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 取得語言與模型選項，從 POST 請求參數中提取
        language = request.form.get('language', 'en')  # 默認為英文
        model_name = request.form.get('model', 'base')  # 默認使用 base 模型

        # 進行轉錄
        transcription = transcribe_audio(filepath, language, model_name)

        # 獲取所選的檔案格式
        output_format = request.form.get('output_format', 'txt')  # 默認格式為 txt
        output_file = save_transcription(transcription, filename.rsplit('.', 1)[0], output_format)

        # 返回轉錄檔案的下載鏈接
        return send_file(output_file, as_attachment=True)
    
    else:
        return jsonify({"error": "請上傳有效的音檔格式！"}), 400  # 檢查音檔格式

@app.route("/change-password", methods=["GET", "POST"])
def change_password():
    if 'username' not in session:
        return redirect('/login')  # 如果用戶沒有登入，跳轉到登入頁面
    
    username = session['username']
    users = load_users()

    if request.method == "POST":
        old_password = request.form["old_password"]
        new_password = request.form["new_password"]

        # 檢查舊密碼是否正確
        if username in users and users[username]['password'] == old_password:
            # 更新密碼
            users[username]['password'] = new_password
            # 更新檔案中的資料
            with open(USER_FILE, "w", encoding="utf-8") as file:
                for user, info in users.items():
                    if 'api_key' in info:
                        file.write(f"{user},{info['password']},{info['api_key']}\n")
                    else:
                        file.write(f"{user},{info['password']}\n")
            flash("密碼更改成功！", "success")
            return redirect("/home")  # 密碼更改成功後跳轉
            
        else:
            flash("舊密碼錯誤！", "danger")  # 如果舊密碼錯誤，顯示錯誤訊息

    return render_template("change_password.html")


# 登出
@app.route("/logout")
def logout():
    session.pop("username", None)
    flash("已登出", "info")
    return redirect("/login")

if __name__ == "__main__":
    if not os.path.exists(USER_FILE):
        open(USER_FILE, "w").close()  # 若無檔案則建立
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    app.run(debug=True, host='0.0.0.0', port=8080)  # 修改為 8080 端口