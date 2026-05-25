import os
import json
import tempfile
import requests
import asyncio
import torch
import whisper
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import APIKeyHeader
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import uvicorn
from dotenv import load_dotenv

# Qwen-VL 관련 라이브러리 추가
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 환경 변수 로드 (.env)
load_dotenv()

app = FastAPI(title="AI Serving API Server")

# ==========================================
# 🛡️ 보안 인증 로직 (API Key)
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
# 🤖 AI 모델 초기화 (서버 가동 시 1회만 실행)
# ==========================================
print("1. Whisper(STT) 모델 로딩 중...")
model_whisper = whisper.load_model("base")

print("2. OpenAI 클라이언트 연결 중...")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

print("3. Qwen2.5-VL-3B 모델 로딩 중... (시간이 다소 소요됩니다)")
model_qwen = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", 
    torch_dtype=torch.float16, 
    device_map="auto"
)
processor_qwen = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
print("✅ 모든 AI 모델 로딩 완료!")

# GPU 메모리 부족(OOM) 방지를 위한 대기열 자물쇠 (동시 처리 1건으로 제한)
gpu_lock = asyncio.Semaphore(1)


# ==========================================
# 📋 요청/응답 스키마 (데이터 구조 정의)
# ==========================================
class MeetingRecordRequest(BaseModel):
    rec_file_key: str  # 오디오 파일 URL

class MeetingRecordResponse(BaseModel):
    title: str
    full_script: str
    summary: str

class VerifyRequest(BaseModel):
    mission_content: str                  # 미션 내용
    verify_prompt: Optional[str] = None   # 추가 인증 조건 (선택)
    verify_content: str                   # 제출 내용 (텍스트)
    image_url: Optional[str] = None       # 첨부 이미지 URL (선택)

class VerifyResponse(BaseModel):
    rejected: bool                        # true = 미달, false = 통과
    reason: str                           # 판단 이유


# ==========================================
# 🎙️ [기능 1] 회의록: 오디오 URL → STT → 제목/요약
# ==========================================
@app.post("/meeting-record", response_model=MeetingRecordResponse)
async def meeting_record(
    request: MeetingRecordRequest, 
    api_key: str = Depends(verify_api_key)
):
    try:
        audio_resp = requests.get(request.rec_file_key, timeout=60)
        audio_resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"오디오 파일 다운로드 실패: {str(e)}")

    ext = os.path.splitext(request.rec_file_key.split("?")[0])[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(audio_resp.content)
        tmp_path = tmp.name

    try:
        # Whisper STT
        result = model_whisper.transcribe(tmp_path)
        full_script = result["text"]

        # GPT-4o-mini 요약
        gpt_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": '너는 회의록 전문 작성가야. 아래 JSON 형식으로만 응답해:\n{"title": "회의 제목 (20자 이내)", "summary": "핵심 내용 요약"}'},
                {"role": "user", "content": full_script},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        parsed = json.loads(gpt_resp.choices[0].message.content)
        return MeetingRecordResponse(
            title=parsed.get("title", "회의록"), 
            full_script=full_script, 
            summary=parsed.get("summary", "")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==========================================
# 🖼️ [기능 2] 미션 인증: Qwen2.5-VL-3B 로컬 처리
# ==========================================
@app.post("/verify", response_model=VerifyResponse)
async def verify_mission(
    request: VerifyRequest, 
    api_key: str = Depends(verify_api_key)
):
    # AI 프롬프트 작성
    system_prompt = (
        f"미션 내용: {request.mission_content}\n"
        + (f"추가 조건: {request.verify_prompt}\n" if request.verify_prompt else "")
        + f"제출 내용: {request.verify_content}\n\n"
        + "위 내용과 이미지를 분석하여 미션 성공 여부를 판단해. 반드시 아래 JSON 형식으로만 대답해:\n"
        + '{"rejected": false, "reason": "성공/실패 판단 이유를 구체적으로 작성"}\n'
        + "조건을 하나라도 어겼거나 이미지가 불분명하면 rejected를 true로 설정해."
    )

    content_list = []
    if request.image_url:
        content_list.append({"type": "image", "image": request.image_url})
    content_list.append({"type": "text", "text": system_prompt})

    messages = [{"role": "user", "content": content_list}]

    try:
        # 동시 요청 시 1건씩 순차 처리 (OOM 방지)
        async with gpu_lock:
            text = processor_qwen.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor_qwen(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            generated_ids = model_qwen.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor_qwen.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        # 마크다운 찌꺼기 제거 후 JSON 파싱
        cleaned_output = output_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned_output)
        
        return VerifyResponse(
            rejected=bool(parsed.get("rejected", False)),
            reason=str(parsed.get("reason", ""))
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI가 JSON 형식이 아닌 응답을 반환했습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"미션 분석 중 에러가 발생했습니다: {str(e)}")


# ==========================================
# 🎙️ [기능 3] 직접 파일 업로드 기반 STT (테스트용)
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
        result = model_whisper.transcribe(tmp_path)
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