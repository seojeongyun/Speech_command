import torchaudio
import torchaudio.transforms as T

if __name__ == '__main__':
    file_name = "dog"

    # 입력 파일 경로와 출력 파일 경로를 설정합니다.
    input_file = '/home/jysuh/Downloads/{}.wav'.format(file_name)
    output_file = '/home/jysuh/Downloads/output.wav'

    data, sample_rate = torchaudio.load(input_file)
    input_freq = data.shape[1]

    # WAV 파일을 16kHz로 변환합니다.
    transform = T.Resample(orig_freq=input_freq, new_freq=16000)
    waveform, sample_rate = torchaudio.load(input_file)
    resampled_waveform = transform(waveform)

    # 변환된 오디오를 저장합니다.
    torchaudio.save(output_file, resampled_waveform, sample_rate=16000)
