# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import time

# @st.cache_resource
# def load_model():
#     model_name = "Gwangwoon/muse"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
#     return tokenizer, model, device

# def generate_response(tokenizer, model, device, prompt, max_length=100):
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
#     try:
#         with torch.no_grad():
#             outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1, do_sample=True)
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     except Exception as e:
#         return f"오류가 발생했습니다: {str(e)}"
    
#     return response

# def main():
#     st.title("Muse Q&A App")
#     st.write("Hugging Face 모델을 이용한 질문 답변 서비스")

#     tokenizer, model, device = load_model()

#     user_input = st.text_area("질문을 입력하세요:")
#     if st.button("답변 받기"):
#         if user_input:
#             with st.spinner('답변을 생성 중입니다...'):
#                 response = generate_response(tokenizer, model, device, user_input)
#                 st.write("**답변:**", response)
#         else:
#             st.warning("질문을 입력해주세요.")

# if __name__ == "__main__":
#     main()
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# CUDA 가용성 확인
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("CUDA is not available. Using CPU.")

@st.cache_resource
def load_model():
    model_name = "Gwangwoon/muse"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device)
    return tokenizer, model, device

def generate_response(tokenizer, model, device, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"
    
    return response

def main():
    st.title("Muse Q&A App")
    st.write("Hugging Face 모델을 이용한 질문 답변 서비스")

    tokenizer, model, device = load_model()

    user_input = st.text_area("질문을 입력하세요:")
    if st.button("답변 받기"):
        if user_input:
            with st.spinner('답변을 생성 중입니다...'):
                response = generate_response(tokenizer, model, device, user_input)
                st.write("**답변:**", response)
        else:
            st.warning("질문을 입력해주세요.")

if __name__ == "__main__":
    main()
