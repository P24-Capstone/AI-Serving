import os
import json
import tempfile
import requests
import asyncio
import torch
import whisper
import re
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import APIKeyHeader
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import uvicorn
from dotenv import load_dotenv
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

load_dotenv()
app = FastAPI(title="AI Serving API Server")

# 보안 인증
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
MY_SECRET_KEY = os.getenv("MY_GCUBE_SECRET")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != MY_SECRET_KEY:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다.")
    return api_key

# 모델 로딩
print("1. 모델 로딩 중...")
model_whisper = whisper.load_model("base")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.float16, device_map="auto"
)
processor_qwen = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
gpu_lock = asyncio.Semaphore(1)

# 스키마 정의
class MeetingRecordRequest(BaseModel):
    rec_file_key: str

class MeetingRecordResponse(BaseModel):
    title: str
    full_script: str
    summary: str
    
class VerifyRequest(BaseModel):
    mission_content: str
    verify_prompt: Optional[str] = None
    verify_content: str
    image_url: Optional[str] = None

class VerifyResponse(BaseModel):
    rejected: bool
    reason: str

def _extract_json(raw: str) -> dict:
   import re

def _extract_json(raw: str) -> dict:
    """
    모델 출력에서 JSON 객체를 추출한다.
    만약 모델이 JSON 포맷(중괄호)을 무시하고 쌩 텍스트로 답해도 이를 파싱해낸다.
    """
    text = raw.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end   = text.rfind("}")
    
    # [플랜 A] 정상적으로 중괄호를 찾은 경우
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass # 파싱 실패 시 아래 플랜 B로 넘어감
            
    # [플랜 B] 모델이 중괄호 없이 "false, '이유'" 형태로 뱉었을 때의 야매 파싱
    raw_lower = raw.lower()
    
    # 1. rejected 여부 판단 (true가 앞에 있으면 True, 아니면 False)
    rejected = False
    if "true" in raw_lower[:15]:
        rejected = True
        
    # 2. 이유(reason) 추출: 쌍따옴표 안의 문장을 찾음
    reason_match = re.search(r'"([^"]*)"', raw)
    if reason_match:
        reason = reason_match.group(1)
    else:
        # 쌍따옴표도 없다면 true, false, 쉼표 등을 지우고 남은 텍스트를 통째로 사용
        reason = raw.replace("true", "").replace("false", "").replace("True", "").replace("False", "").replace(",", "").strip()
        
    return {"rejected": rejected, "reason": reason}

@app.post("/meeting-record", response_model=MeetingRecordResponse)
async def meeting_record(request: MeetingRecordRequest, api_key: str = Depends(verify_api_key)):
    tmp_path = None
    try:
        response = requests.get(request.rec_file_key, timeout=60, stream=True)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name
        
        # Whisper STT
        result = model_whisper.transcribe(tmp_path)
        full_script = result["text"]
        
        # GPT 요약
        gpt_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": '회의록 전문가입니다. JSON으로 응답해: {"title": "제목(20자)", "summary": "요약"}'},
                {"role": "user", "content": full_script},
            ],
            response_format={"type": "json_object"}
        )
        parsed = json.loads(gpt_resp.choices[0].message.content)
        return MeetingRecordResponse(
            title=parsed.get("title", "회의록"), 
            full_script=full_script, 
            summary=parsed.get("summary", "")
        )
    finally:
        if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)


@app.post("/verify", response_model=VerifyResponse)
async def verify_mission(request: VerifyRequest, api_key: str = Depends(verify_api_key)):
    try:
        async with gpu_lock:
            content_list = []

            # [이미지 처리]
            if request.image_url:
                resp = requests.get(request.image_url, stream=True, timeout=15)
                resp.raise_for_status()
                image = Image.open(resp.raw).convert("RGB")
                content_list.append({"type": "image", "image": image, "max_pixels": 313600})

            # [프롬프트] — verify_prompt(검증 기준) 포함, 반환 형식 명시
            criteria = request.verify_prompt if request.verify_prompt else "미션 내용과 일치하는지 판단하세요."
            user_text = (
                f"[미션 설명] {request.mission_content}\n"
                f"[검증 기준] {criteria}\n"
                f"[제출자 설명] {request.verify_content or '(설명 없음)'}\n\n"
                "위 기준을 바탕으로 첨부 이미지와 제출 내용이 미션 인증 조건을 충족하는지 판단하세요.\n"
                "다음 JSON 형식으로만 결과값을 출력하세요.\n"
                "다른 설명이나 문장은 절대 포함하지 마세요.\n"
                "{\n"
                '  "rejected": true/false,\n'
                '  "reason": "판정 이유를 한국어로 짧게 작성"\n'
                "}"
            )
            content_list.append({"type": "text", "text": user_text})

            # [메시지] — system 역할로 JSON 전용 출력 강제
            messages = [
                {
                    "role": "system",
                    "content": (
                        "당신은 미션 인증 판정 AI입니다. "
                        "사용자의 요청에 대해 반드시 {\"rejected\": boolean, \"reason\": string} "
                        "형태의 JSON만 출력해야 합니다. 다른 텍스트는 출력하지 마세요."
                    ),
                },
                {"role": "user", "content": content_list},
            ]

            # [모델 추론]
            text = processor_qwen.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor_qwen(
                text=[text], images=image_inputs, padding=True, return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                generated_ids = model_qwen.generate(**inputs, max_new_tokens=256)

            raw_output = processor_qwen.batch_decode(
                [generated_ids[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True
            )[0]

            torch.cuda.empty_cache()

            # [JSON 파싱] — 추출 실패 시 서버 오류 대신 "대기(P 유지)" 응답 반환
            try:
                parsed = _extract_json(raw_output)
                rejected = bool(parsed.get("rejected", False))
                reason   = str(parsed.get("reason", ""))
            except (ValueError, json.JSONDecodeError) as parse_err:
                # 파싱 실패: 백엔드가 상태를 P(대기)로 유지하도록 예외를 그대로 올림
                raise HTTPException(
                    status_code=502,
                    detail=f"AI 모델이 유효한 JSON을 반환하지 않았습니다: {parse_err}",
                )

            return VerifyResponse(rejected=rejected, reason=reason)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에러 발생: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)