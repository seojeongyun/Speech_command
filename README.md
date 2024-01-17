1. pytorch audio

https://pytorch.org/audio/stable/index.html
https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html



2. 음성인식에 필요한 기초개념

https://lynnshin.tistory.com/42
https://deeesp.github.io/speech/ASR/

 

3. 한국어음성인식 관련 스터디 자료 모음

https://github.com/sooftware/Speech-Recognition-Tutorial



4. 실습할 pytorch code 

명령어인식 https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html



5. 오픈AI의 Whisper 오픈소스

https://devocean.sk.com/blog/techBoardDetail.do?ID=164545



[구글코랩실습] https://colab.research.google.com/drive/1g8TYPv9sy4usQsHSB7tm8OiB6Q6gO0VE?usp=sharing&fbclid=IwAR3quoGCWcQA_Het5P8ho9hqKeFclUe2UK8BrWp2vKAYcPfPn6-YiNMi_68





* 파이토치 한국사용자 모임

https://discuss.pytorch.kr/


# 음성인식 관련 데이터는 sample rate 맞춰주는 게 중요함. 주로 16000Hz를 다뤘는데 16000Hz로 작업하면 네트워크 연산량이 너무 많아져서 8000Hz로 Donwsample하고 진행함.
# M5 network를 사용했는데 성능이 너무 안 좋아서 추후에 network를 교체해볼 예정.
# record 함수가 자바 기반 코드라 연구실 서버로 돌리니까 안 돌아갔음. 우분투에 자바 설치해보려다 실패.
