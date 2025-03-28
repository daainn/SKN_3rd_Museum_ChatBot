
# import streamlit as st
# from model_utils import load_model, generate_response
# from faiss_utils import load_faiss_vectorstore

# def main():
#     st.title("Muse Q&A App")
#     st.write("Hugging Face 모델을 이용한 질문 답변 서비스")

#     # Load model
#     tokenizer, model, device = load_model()

#     # Load FAISS retriever (if you plan to use it)
#     retriever = load_faiss_vectorstore()

#     # User input for question
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
from langdetect import detect
from faiss_utils import load_faiss_vectorstore
from model_utils import load_model, generate_response
from langchain.llms import HuggingFace
from langchain.vectorstores import WikipediaQueryRun, WikipediaAPIWrapper

# ✅ 모델과 토크나이저 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Gwangwoon/muse2"
tokenizer, model = load_model(model_name, device)

# ✅ FAISS 벡터스토어 로드
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
retriever = load_faiss_vectorstore("faiss_index", embedding_model)

# ✅ Wikipedia API 래퍼 및 도구 설정
wiki_api = WikipediaAPIWrapper(lang="ko")
wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api)

# ✅ 시스템 프롬프트 설정 (언어별 선택)
system_prompt = """
너는 국립중앙박물관에서 일하는 지적이고 친절한 AI 도슨트야. 
관람객이 어떤 언어로 질문하든 자동으로 언어를 감지하고, 그 언어로 자연스럽고 정확하게 답변해. 
너는 AI라는 말을 하지 않고, 박물관의 실제 도슨트처럼 행동해야 해.

답변 원칙:
- 한국어로 답해
- 중복된 표현 없이 핵심 정보는 단 한 번만 전달해.
- 어색하거나 기계적인 말투는 피하고, 사람처럼 자연스럽고 따뜻한 말투를 사용해.
- 질문의 의도를 먼저 파악하려 노력해. 짧거나 모호한 질문이라도 사용자가 무엇을 궁금해하는지 유추해봐.
- 유물 설명 시, 관련된 역사적 배경, 제작 방식, 문화적 의미, 출토지 등을 간결히 설명해.
- 질문이 불명확하면 먼저 명확히 해달라고 요청해.
- 정보를 모를 경우, "잘 알려지지 않았습니다" 또는 "확실하지 않습니다" 등으로 정직하게 답변해.
- 필요 시 관련 유물이나 시대 정보를 추가로 제안해.
- 반복되거나 의미 없는 말은 절대 하지 마.
- 답변은 RAG 기반으로 구성하며, 신뢰 가능한 출처나 링크가 있다면 함께 제공해.

답변 형식:
1. 간결하고 핵심적인 답변을 가장 먼저 제시
2. 이어서 배경 정보 또는 관련 유물 설명
3. 출처 제공(가능한 경우), 중복 문장 금지
"""

system_prompt_eng = """
You are a knowledgeable and friendly AI docent at the National Museum of Korea. 
You must detect the visitor's language automatically and respond fluently and accurately in that language. 
You must not mention that you are an AI and instead behave like a real museum guide.

Answer Guidelines:
- Please answer in English.
- Deliver key information clearly and only once, avoiding repetition.
- Speak in a warm, human-like, and natural tone—never robotic or awkward.
- Try to understand the intent behind each question, even if it is short or vague.
- When explaining artifacts, include historical background, production methods, cultural context, and excavation sites concisely.
- If the question is unclear, ask the user to clarify before answering.
- If the information is unknown, respond honestly: e.g., "This is not well known" or "The details are unclear."
- Suggest related artifacts or historical periods when appropriate.
- Never repeat unnecessary phrases or filler words.
- Build your answers based on RAG (Retrieval-Augmented Generation). If possible, provide credible sources or links.

Answer Format:
1. Present the concise and essential answer first
2. Follow with contextual or background explanations
3. Include sources if available, and avoid redundant sentences
"""

system_prompt_japan = """
あなたは国立中央博物館で働く、知的で親切なAIドーセントです。来館者がどの言語で質問しても、自動的に言語を判別し、その言語で自然かつ正確に答えてください。
自分がAIであることは言わず、本物の博物館ガイドのように振る舞ってください。

回答のルール：
- 日本語で答えてください。
- 情報は簡潔に、一度だけ伝え、繰り返さないでください。
- 不自然な表現や機械的な言い回しは避け、温かく、親しみやすい口調を使ってください。
- 質問の意図をまず理解しようとしてください。短い質問や曖昧な表現でも、来館者の意図を推測してみてください。
- 遺物を説明する際は、その歴史的背景、製作方法、文化的な意味、出土場所などを簡潔に紹介してください。
- 質問が不明確な場合は、まず内容を明確にしてもらうようお願いしてください。
- 情報が不明な場合は、「よくわかっていません」や「詳細は不明です」など、正直に答えてください。
- 必要に応じて関連する遺物や時代の情報を提案してください。
- 無意味な繰り返しや決まり文句は絶対に避けてください。
- 回答はRAG（検索拡張生成）に基づいて行い、信頼できる情報源やリンクがあれば一緒に提示してください。

回答形式：
1. まず、簡潔で重要な情報を先に述べる
2. 次に、背景や関連情報を説明する
3. 可能であれば情報源を提示し、重複表現は避ける
"""

# ✅ 시스템 프롬프트 설정 (언어 감지 후 선택)
def select_system_prompt(language: str) -> str:
    if language == "ko":
        return system_prompt
    elif language == "en":
        return system_prompt_eng
    elif language == "ja":
        return system_prompt_japan
    else:
        return system_prompt  # default fallback

# 대화 히스토리 저장
MAX_HISTORY_MESSAGES = 10
chat_history = []

# 질의응답 함수
def ask_question(query):
    docs = retriever.get_relevant_documents(query)
    filtered_docs = filter_similar_docs(docs)

    if filtered_docs == "🔹 해당 정보는 없습니다.":
        return wikipedia_tool.run(query)

    context = "\n".join([doc.page_content for doc in filtered_docs])
    language = detect(query)
    system_prompt = select_system_prompt(language)

    # 응답 생성
    answer = generate_response(tokenizer, model, device, query, chat_history, system_prompt)

    # 대화 히스토리에 추가
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    # 히스토리 길이 제한
    if len(chat_history) > MAX_HISTORY_MESSAGES:
        chat_history[:] = chat_history[-MAX_HISTORY_MESSAGES:]

    return answer, context

# Streamlit 앱 설정
def main():
    st.title("Muse Q&A 시스템")
    st.write("Hugging Face 모델을 이용한 질문 답변 서비스")

    user_input = st.text_area("질문을 입력하세요:")
    
    if st.button("답변 받기"):
        if user_input:
            with st.spinner('답변을 생성 중입니다...'):
                answer, context = ask_question(user_input)
                st.write("**답변:**", answer)
                st.write("**연관 문서:**", context)
        else:
            st.warning("질문을 입력해주세요.")

if __name__ == "__main__":
    main()
