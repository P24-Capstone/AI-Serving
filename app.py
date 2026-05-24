import os
import tempfile
import whisper
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import APIKeyHeader # 추가된 모듈
from openai import OpenAI
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="STT & Summary API Server")

# ==========================================
# 🛡️ 문지기(보안) 로직 세팅
# 클라이언트(AWS 백엔드)가 헤더에 'X-API-Key'를 담아 보내도록 설정
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# .env 파일에서 우리가 설정한 비밀번호 불러오기
MY_SECRET_KEY = os.getenv("MY_GCUBE_SECRET")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """요청이 들어올 때마다 비밀번호가 맞는지 검사하는 함수"""
    if api_key != MY_SECRET_KEY:
        # 비밀번호가 틀리거나 없으면 403 에러로 쫓아냄
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다. (Invalid API Key)")
    return api_key
# ==========================================

# Whisper 및 OpenAI 초기화 로직 (이전과 동일)
model = whisper.load_model("base") 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# 엔드포인트에 보안 검사(Depends) 적용
@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...), 
    api_key: str = Depends(verify_api_key) # 🔥 이 부분이 추가됨! 
):
    """
    이제 이 엔드포인트는 올바른 X-API-Key를 가진 사람만 들어올 수 있습니다.
    """
    if not file.filename.endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    # ... (이하 STT 및 요약 진행 로직은 이전과 완전히 동일) ...
    # 1단계: 업로드된 파일을 임시 파일로 저장 (Whisper가 읽을 수 있도록)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # 2단계: Whisper로 STT(음성 인식) 진행
        result = model.transcribe(tmp_path)
        transcribed_text = result['text']

        # 3단계: OpenAI GPT-4o-mini를 이용해 텍스트 요약
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 전문 요약가야. 제공된 텍스트를 핵심 내용만 깔끔하게 요약해줘."},
                {"role": "user", "content": transcribed_text}
            ],
            temperature=0.3
        )
        summary_text = response.choices[0].message.content

        # 4단계: JSON 형태로 결과 반환 (FastAPI는 딕셔너리를 자동으로 JSON으로 변환합니다)
        return {
            "status": "success",
            "original_text": transcribed_text,
            "summary": summary_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에러가 발생했습니다: {str(e)}")
    
    finally:
        # 작업이 끝나면 임시 오디오 파일 삭제 (서버 용량 확보)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

