# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # CUDA 가용성 확인
# def check_cuda():
#     if torch.cuda.is_available():
#         print("CUDA is available. Using GPU.")
#         print(f"CUDA version: {torch.version.cuda}")
#     else:
#         print("CUDA is not available. Using CPU.")

# def load_model():
#     model_name = "Gwangwoon/muse2"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
#     model = model.to(device)
#     return tokenizer, model, device

# def generate_response(tokenizer, model, device, prompt, max_length=100):
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
    
#     try:
#         with torch.no_grad():
#             outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1, do_sample=True)
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     except Exception as e:
#         return f"오류가 발생했습니다: {str(e)}"
    
#     return response# model_utils.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFace

def load_model(model_name: str, device: torch.device):
    """
    모델과 토크나이저를 로드하는 함수.
    :param model_name: 모델 이름
    :param device: 사용할 디바이스 (CUDA / CPU)
    :return: tokenizer, model, device
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device)
    return tokenizer, model

def generate_response(tokenizer, model, device, query, chat_history, system_prompt, max_length=100):
    """
    주어진 입력에 대한 응답을 생성하는 함수.
    :param tokenizer: 토크나이저
    :param model: 모델
    :param device: 디바이스 (CUDA / CPU)
    :param query: 사용자 입력
    :param chat_history: 대화 히스토리
    :param system_prompt: 시스템 프롬프트
    :param max_length: 생성할 응답의 최대 길이
    :return: 생성된 응답
    """
    # 전체 메시지 구성 (시스템 프롬프트와 대화 히스토리 포함)
    messages = [{"role": "system", "content": system_prompt}]
    messages += chat_history  # 과거 대화 유지
    messages.append({
        "role": "user",
        "content": f"질문: {query}\n답변:"
    })

    # 텍스트 생성
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    # 응답 생성
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        top_p=0.9,
        temperature=0.3
    )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 'assistant' 이후 텍스트만 추출
    if "assistant" in output_text:
        answer = output_text.split("assistant")[-1].strip()
    else:
        answer = output_text.strip()

    return answer
