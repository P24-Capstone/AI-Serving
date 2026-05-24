import os
import json
import tempfile
import requests
import whisper
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import APIKeyHeader
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Serving API Server")

# ==========================================
# 🛡️ 인증 로직
# ==========================================
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
MY_SECRET_KEY = os.getenv("MY_GCUBE_SECRET")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """요청마다 X-API-Key 헤더가 올바른지 검사"""
    if api_key != MY_SECRET_KEY:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다. (Invalid API Key)")
    return api_key

# ==========================================
# Whisper / OpenAI 초기화
# ==========================================
model = whisper.load_model("base")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


# ==========================================
# 📋 요청/응답 스키마
# ==========================================

class MeetingRecordRequest(BaseModel):
    rec_file_key: str           # 오디오 파일의 URL (S3 presigned URL 또는 로컬 URL)

class MeetingRecordResponse(BaseModel):
    title: str
    full_script: str
    summary: str


class VerifyRequest(BaseModel):
    mission_content: str        # 미션 내용
    verify_prompt: Optional[str] = None   # 추가 인증 조건 (선택)
    verify_content: str         # 제출 내용 (텍스트)
    image_url: Optional[str] = None       # 첨부 이미지 URL (선택)

class VerifyResponse(BaseModel):
    rejected: bool              # true = 미달, false = 통과
    reason: str                 # 판단 이유


# ==========================================
# 🎙️ 회의록: 오디오 URL → STT → 제목/요약
# POST /meeting-record
# ==========================================
@app.post("/meeting-record", response_model=MeetingRecordResponse)
async def meeting_record(
    request: MeetingRecordRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    오디오 파일 URL을 받아 STT(Whisper) → 제목·요약(GPT) 후 결과를 반환합니다.
    백엔드 AiProcessingService가 호출합니다.
    """
    # 1단계: 오디오 파일 다운로드
    try:
        audio_resp = requests.get(request.rec_file_key, timeout=60)
        audio_resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"오디오 파일 다운로드 실패: {str(e)}")

    # URL 쿼리 파라미터 제거 후 확장자 추출
    ext = os.path.splitext(request.rec_file_key.split("?")[0])[-1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(audio_resp.content)
        tmp_path = tmp.name

    try:
        # 2단계: Whisper STT
        result = model.transcribe(tmp_path)
        full_script = result["text"]

        # 3단계: GPT — 제목 + 요약 생성 (JSON 모드)
        gpt_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 회의록 전문 작성가야. "
                        "제공된 회의 텍스트를 분석해서 반드시 아래 JSON 형식으로만 응답해:\n"
                        '{"title": "회의 제목 (20자 이내)", "summary": "핵심 내용 요약"}'
                    ),
                },
                {"role": "user", "content": full_script},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        parsed = json.loads(gpt_resp.choices[0].message.content)
        title   = parsed.get("title", "회의록")
        summary = parsed.get("summary", "")

        return MeetingRecordResponse(title=title, full_script=full_script, summary=summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==========================================
# ✅ 미션 인증: 제출 내용 → AI 판단
# POST /verify
# ==========================================
@app.post("/verify", response_model=VerifyResponse)
async def verify_mission(
    request: VerifyRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    미션 제출 내용이 미션 조건을 충족하는지 GPT로 판단합니다.
    백엔드 MissionService가 호출합니다.
    """
    user_prompt = (
        f"미션 내용: {request.mission_content}\n"
        + (f"추가 인증 조건: {request.verify_prompt}\n" if request.verify_prompt else "")
        + f"제출 내용: {request.verify_content}\n"
        + (f"첨부 이미지: {request.image_url}\n" if request.image_url else "")
        + "\n위 제출이 미션 조건을 충족하는지 판단하고, 반드시 아래 JSON 형식으로만 응답해:\n"
        + '{"rejected": false, "reason": "판단 이유를 한국어로 작성"}\n'
        + "rejected가 true이면 조건 미달입니다."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "너는 공정한 미션 심사관이야. 제출 내용이 미션 기준을 충족하는지 객관적으로 판단해.",
            },
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    parsed = json.loads(response.choices[0].message.content)
    return VerifyResponse(
        rejected=bool(parsed.get("rejected", False)),
        reason=str(parsed.get("reason", "")),
    )


# ==========================================
# 🎙️ 기존 /process-audio (파일 직접 업로드 방식, 호환성 유지)
# ==========================================
@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    if not file.filename.endswith((".mp3", ".wav", ".m4a")):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path)
        transcribed_text = result["text"]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 전문 요약가야. 제공된 텍스트를 핵심 내용만 깔끔하게 요약해줘."},
                {"role": "user", "content": transcribed_text},
            ],
            temperature=0.3,
        )
        summary_text = response.choices[0].message.content

        return {
            "status": "success",
            "original_text": transcribed_text,
            "summary": summary_text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에러가 발생했습니다: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
