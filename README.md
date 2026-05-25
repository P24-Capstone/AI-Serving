1단계: 주피터 노트북 환경 세팅 (패키지 설치)

pip install -U openai-whisper torch
pip install openai fastapi uvicorn python-multipart
pip install python-dotenv

qwen 설치
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils==0.0.8
pip install --upgrade torch torchvision torchaudio einops

통신용 https 개설
apt-get update
apt-get install -y nodejs npm


키 입력
cd AI-Serving
echo "OPENAI_API_KEY=sk-본인의_오픈AI_키" > .env
echo "MY_GCUBE_SECRET=본인이_설정한_비밀번호" >> .env
확인인
cat .env

2단계: 전체 흐름(파이프라인) 

입력: 사용자가 음성 파일(mp3, wav 등)을 서버로 보냄.

STT: gcube 서버에 있는 GPU를 활용해 Whisper가 음성을 텍스트로 변환.

요약: 변환된 텍스트를 OpenAI API(gpt-4o-mini)로 보내서 요약 요청.

출력: 최종적으로 {"original_text": "...", "summary": "..."} 형태의 JSON 반환.

3단계: 서버 실행 및 테스트

주피터 노트북에서 터미널 2개 준비

1. 주피터 노트북 상단 메뉴의 New -> Terminal을 클릭하여 터미널을 연 뒤, app.py 파일이 있는 경로에서 아래 명령어를 입력합니다.

uvicorn app:app --host 0.0.0.0 --port 8000 --reload

화면에 Application startup complete. 라는 문구가 뜨면 서버가 성공적으로 열린 것입니다.

2. 새로 연 터미널에 아래 명령어를 입력합니다.
npx localtunnel --port 8000
명령어가 성공적으로 실행되면 터미널 화면에 아래와 비슷한 문구가 뜹니다.
your url is: https://adjective-animal-123.loca.lt
