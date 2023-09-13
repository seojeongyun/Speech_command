from IPython.display import Javascript, display
from base64 import b64decode
from io import BytesIO
from pydub import AudioSegment

def save_recorded_audio(filename, sample_rate=16000, target_num_samples=16000):
    RECORD = (
        b"const sleep  = time => new Promise(resolve => setTimeout(resolve, time))\n"
        b"const b2text = blob => new Promise(resolve => {\n"
        b"  const reader = new FileReader()\n"
        b"  reader.onloadend = e => resolve(e.srcElement.result)\n"
        b"  reader.readAsDataURL(blob)\n"
        b"})\n"
        b"var record = time => new Promise(async resolve => {\n"
        b"  stream = await navigator.mediaDevices.getUserMedia({ audio: true })\n"
        b"  recorder = new MediaRecorder(stream)\n"
        b"  chunks = []\n"
        b"  recorder.ondataavailable = e => chunks.push(e.data)\n"
        b"  recorder.start()\n"
        b"  await sleep(time)\n"
        b"  recorder.onstop = async ()=>{\n"
        b"    blob = new Blob(chunks)\n"
        b"    text = await b2text(blob)\n"
        b"    resolve(text)\n"
        b"  }\n"
        b"  recorder.stop()\n"
        b"})"
    )
    RECORD = RECORD.decode("ascii")

    print(f"Recording started for {target_num_samples} samples at {sample_rate} Hz sample rate.")
    display(Javascript(RECORD))
    s = colab_output.eval_js("record(%d)" % (target_num_samples / sample_rate * 1000))
    print("Recording ended.")
    b = b64decode(s.split(",")[1])

    fileformat = "wav"
    filename = f"{filename}.{fileformat}"

    # PyDub를 사용하여 샘플링 속도와 샘플 수를 조정하고 저장
    audio = AudioSegment.from_file(BytesIO(b))
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(1)  # 모노 오디오로 설정

    # 실제 샘플 수 확인
    actual_num_samples = len(audio.get_array_of_samples())

    # 샘플 수를 목표 샘플 수로 조절
    if actual_num_samples < target_num_samples:
        padding = AudioSegment.silent(duration=(target_num_samples - actual_num_samples) * 1000 // sample_rate)
        audio = audio + padding
    elif actual_num_samples > target_num_samples:
        audio = audio[:target_num_samples]

    audio.export(filename, format=fileformat)
    print(f"Audio saved as {filename} with {sample_rate} Hz sample rate and {target_num_samples} samples")

# 16000 Hz 샘플링 속도와 16000 개의 샘플로 녹음하여 저장하는 예시
save_recorded_audio("my_recorded_audio", sample_rate=16000, target_num_samples=16000)
