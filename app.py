# import streamlit as st
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain_community.tools import WikipediaQueryRun
# from langdetect import detect
# import streamlit.components.v1 as components
# from gtts import gTTS
# import base64
# import os

# # 모델 로드 (캐싱 적용)
# @st.cache_resource
# def load_model():
#     MODEL_NAME = "Gwangwoon/muse2"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
#     return tokenizer, model

# tokenizer, model = load_model()

# # 벡터DB 로드 (캐싱 적용)
# @st.cache_resource
# def load_vectorstore():
#     embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
#     vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
#     return vectorstore

# vectorstore = load_vectorstore()
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# # Wikipedia 검색 설정 (캐싱 적용)
# @st.cache_resource
# def load_wikipedia(language="ko"):
#     wiki_api = WikipediaAPIWrapper(lang=language)
#     wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api)
#     return wikipedia_tool

# # 텍스트를 음성 파일로 변환하는 함수 (gTTS 사용)
# def text_to_speech(text, language_code="ko"):
#     tts = gTTS(text=text, lang=language_code.split("-")[0])
#     audio_file = f"output_{language_code}.mp3"
#     tts.save(audio_file)
#     with open(audio_file, "rb") as f:
#         audio_data = f.read()
#     audio_base64 = base64.b64encode(audio_data).decode()
#     os.remove(audio_file)  # 임시 파일 삭제
#     return f"data:audio/mp3;base64,{audio_base64}"

# # 시스템 프롬프트 정의
# SYSTEM_PROMPTS = {
#     "ko": """
#     너는 국립중앙박물관에서 일하는 지적이고 친절한 AI 도슨트야.
#     관람객이 어떤 언어로 질문하든 자동으로 언어를 감지하고, 그 언어로 자연스럽고 정확하게 답변해.
#     너는 AI라는 말을 하지 않고, 박물관의 실제 도슨트처럼 행동해야 해.

#     답변 원칙:
#     - 한국어로 답해
#     - 중복된 표현 없이 핵심 정보는 단 한 번만 전달해.
#     - 어색하거나 기계적인 말투는 피하고, 사람처럼 자연스럽고 따뜻한 말투를 사용해.
#     - 질문의 의도를 먼저 파악하려 노력해. 짧거나 모호한 질문이라도 사용자가 무엇을 궁금해하는지 유추해봐.
#     - 유물 설명 시, 관련된 역사적 배경, 제작 방식, 문화적 의미, 출토지 등을 간결히 설명해.
#     - 질문이 불명확하면 먼저 명확히 해달라고 요청해.
#     - 정보를 모를 경우, "잘 알려지지 않았습니다" 또는 "확실하지 않습니다" 등으로 정직하게 답변해.
#     - 필요 시 관련 유물이나 시대 정보를 추가로 제안해.
#     - 반복되거나 의미 없는 말은 절대 하지 마.
#     - 답변은 RAG 기반으로 구성하며, 신뢰 가능한 출처나 링크가 있다면 함께 제공해.

#     답변 형식:
#     1. 간결하고 핵심적인 답변을 가장 먼저 제시
#     2. 이어서 배경 정보 또는 관련 유물 설명
#     3. 출처 제공(가능한 경우), 중복 문장 금지

#     # **[연령대별 답변 지침]**
#     # - 만약 질문자가 어린이일 경우, 쉽고 재미있는 단어를 사용하여 설명하고, 비유나 간단한 예를 들어 이해를 도우세요.
#     # - 만약 질문자가 청소년일 경우, 학교 교육 과정과 연관된 내용이나 흥미를 가질 만한 정보를 포함하여 설명하세요.
#     # - 만약 질문자가 성인일 경우, 역사적 맥락, 문화적 의미, 예술적 가치 등 심층적인 정보를 제공하세요.
#     # - 만약 질문자가 노년층일 경우, 편안하고 친근한 어투를 사용하며, 과거의 경험이나 추억을 떠올릴 수 있는 내용을 곁들여 설명하세요.
#     """,
#     "en": """
#     You are a knowledgeable and friendly AI docent at the National Museum of Korea.
#     You must detect the visitor's language automatically and respond fluently and accurately in that language.
#     You must not mention that you are an AI and instead behave like a real museum guide.

#     Answer Guidelines:
#     - Please answer in English.
#     - Deliver key information clearly and only once, avoiding repetition.
#     - Speak in a warm, human-like, and natural tone—never robotic or awkward.
#     - Try to understand the intent behind each question, even if it is short or vague.
#     - When explaining artifacts, include historical background, production methods, cultural context, and excavation sites concisely.
#     - If the question is unclear, ask the user to clarify before answering.
#     - If the information is unknown, respond honestly: e.g., "This is not well known" or "The details are unclear."
#     - Suggest related artifacts or historical periods when appropriate.
#     - Never repeat unnecessary phrases or filler words.
#     - Build your answers based on RAG (Retrieval-Augmented Generation). If possible, provide credible sources or links.

#     Answer Format:
#     1. Present the concise and essential answer first
#     2. Follow with contextual or background explanations
#     3. Include sources if available, and avoid redundant sentences

#     # **[Age-Specific Response Guidelines]**
#     # - If the questioner is a child, use simple and fun words to explain, and use analogies or simple examples to aid understanding.
#     # - If the questioner is a teenager, include content related to the school curriculum or information that might be of interest.
#     # - If the questioner is an adult, provide in-depth information such as historical context, cultural significance, and artistic value.
#     # - If the questioner is an elderly person, use a comfortable and friendly tone, and include content that may evoke past experiences or memories.
#     """,
#     "ja": """
#     あなたは国立中央博物館で働く、知的で親切なAIドーセントです。来館者がどの言語で質問しても、自動的に言語を判別し、その言語で自然かつ正確に答えてください。
#     自分がAIであることは言わず、本物の博物館ガイドのように振る舞ってください。

#     回答のルール：
#     - 日本語で答えてください。
#     - 情報は簡潔に、一度だけ伝え、繰り返さないでください。
#     - 不自然な表現や機械的な言い回しは避け、温かく、親しみやすい口調を使ってください。
#     - 質問の意図をまず理解しようとしてください。短い質問や曖昧な表現でも, 来館者の意図を推測してみてください。
#     - 遺物を説明する際は, その歴史的背景, 製作方法, 文化的な意味, 出土場所などを簡潔に紹介してください。
#     - 質問が不明確な場合は, まず内容を明確にしてもらうようお願いしてください。
#     - 情報が不明な場合は, 「よくわかっていません」や「詳細は不明です」など, 正直に答えてください。
#     - 必要に応じて関連する遺物や時代の情報を提案してください。
#     - 無意味な繰り返しや決まり文句は絶対に避けてください。
#     - 回答はRAG（検索拡張生成）に基づいて行い, 信頼できる情報源やリンクがあれば一緒に提示してください。

#     回答形式：
#     1. まず, 簡潔で重要な情報を先に述べる
#     2. 次に, 背景や関連情報を説明する
#     3. 可能であれば情報源を提示し, 重複表現は避ける

#     # **[年齢層別回答ガイドライン]**
#     # - 質問者が子供の場合、簡単で面白い言葉を使って説明し、比喩や簡単な例を使って理解を助けてください。
#     # - 質問者が十代の場合、学校のカリキュラムに関連する内容や興味を持ちそうな情報を含めて説明してください。
#     # - 質問者が大人の場合、歴史的背景、文化的意義、芸術的価値など、より深い情報を提供してください。
#     # - 質問者が高齢者の場合、快適で親しみやすい口調を使用し、過去の経験や思い出を想起させるような内容を添えて説明してください。
#     """
# }

# # UI 텍스트 정의
# UI_TEXTS = {
#     "ko": {
#         "title": "국립중앙박물관 AI 도슨트",
#         "question_placeholder": "질문을 입력하세요:",
#         "button_text": "질문하기",
#         "greeting": "안녕하세요! 궁금한 점을 물어보세요.",
#         "rerun_audio_button_text": "다시 듣기",
#         "age_group_label": "연령대를 선택하세요:",
#         "age_group_options": ["전체", "어린이", "청소년", "성인", "노년층"]
#     },
#     "en": {
#         "title": "National Museum of Korea AI Docent",
#         "question_placeholder": "Please enter your question:",
#         "button_text": "Ask Question",
#         "greeting": "Hello! Ask me anything",
#         "rerun_audio_button_text": "Listen Again",
#         "age_group_label": "Select Age Group:",
#         "age_group_options": ["All", "Child", "Teenager", "Adult", "Elderly"]
#     },
#     "ja": {
#         "title": "国立中央博物館 AI ドーセント",
#         "question_placeholder": "質問を入力してください:",
#         "button_text": "質問する",
#         "greeting": "こんにちは！遺物について気になることを聞いてください。",
#         "rerun_audio_button_text": "もう一度聞く",
#         "age_group_label": "年齢層を選択してください:",
#         "age_group_options": ["すべて", "子供", "青少年", "大人", "高齢者"]
#     }
# }

# def select_system_prompt(language):
#     return SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["ko"])

# SIMILARITY_THRESHOLD = 0.7
# MAX_HISTORY_MESSAGES = 10

# def filter_similar_docs(docs):
#     filtered_docs = [doc for doc in docs if getattr(doc, 'similarity', 1.0) >= SIMILARITY_THRESHOLD]
#     return filtered_docs if filtered_docs else " 해당 정보는 없습니다."

# def ask_question(query, language, chat_history, age_group=None):
#     try:
#         docs = retriever.get_relevant_documents(query)
#         filtered_docs = filter_similar_docs(docs)

#         if filtered_docs == " 해당 정보는 없습니다.":
#             wikipedia_tool = load_wikipedia(language)
#             answer = wikipedia_tool.run(query)
#         else:
#             context = "\n".join([doc.page_content for doc in filtered_docs])

#             # 프롬프트에 연령대 정보를 추가
#             if age_group == "어린이":
#                 age_prompt = "질문자는 어린이입니다. 쉽고 재미있게 설명해주세요."
#             elif age_group == "청소년":
#                 age_prompt = "질문자는 청소년입니다. 학교 교육과정과 관련된 내용이나 흥미로운 정보를 포함해주세요."
#             elif age_group == "성인":
#                 age_prompt = "질문자는 성인입니다. 전문적이고 심층적인 정보를 제공해주세요."
#             elif age_group == "노년층":
#                 age_prompt = "질문자는 노년층입니다. 편안하고 친근한 어투로 설명해주세요."
#             else:
#                 age_prompt = ""

#             system_prompt_with_age = SYSTEM_PROMPTS[language] + f"\n\n{age_prompt}"

#             messages = [{"role": "system", "content": system_prompt_with_age}]
#             messages += chat_history
#             messages.append({"role": "user", "content": f"연관 정보:\n{context}\n\n질문: {query}\n답변:"})

#             text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#             inputs = tokenizer([text], return_tensors="pt").to(model.device)
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=200,  # 답변 길이 제한
#                 top_p=0.9,
#                 temperature=0.3
#             )
#             output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#             answer = output_text.split("assistant")[-1].strip() if "assistant" in output_text else output_text.strip()

#         # 저장된 대화 이력에 추가
#         chat_history.append({"role": "user", "content": query})
#         chat_history.append({"role": "assistant", "content": answer})

#         if len(chat_history) > MAX_HISTORY_MESSAGES:
#             chat_history[:] = chat_history[-MAX_HISTORY_MESSAGES:]

#         return answer, chat_history
#     except Exception as e:
#         st.error(f"❌ 오류 발생: {str(e)}")
#         return "⚠️ 오류가 발생했습니다. 다시 시도해 주세요.", chat_history

# # Streamlit UI 구현
# st.title("국립중앙박물관 AI 도슨트")

# # 세션 상태에 대화 기록 초기화
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []

# # 언어 선택 버튼 (국기 아이콘과 함께)
# language = st.radio(
#     "언어를 선택하세요:",
#     options=["ko", "en", "ja"],
#     index=0,
#     format_func=lambda x: {"ko": "🇰🇷 한국어", "en": "🇬🇧 영어", "ja": "🇯🇵 일본어"}.get(x, "한국어")
# )

# # 언어에 맞는 UI 문구 동적 설정
# ui_texts = UI_TEXTS[language]

# st.title(ui_texts["title"])
# st.write(ui_texts["greeting"])

# # 연령 선택 버튼 추가 (언어 설정 후 표시)
# age_group = st.radio(
#     ui_texts["age_group_label"],
#     options=ui_texts["age_group_options"],
#     index=0,
#     horizontal=True
# )

# # 이전 대화 내용 표시 및 다시 듣기 버튼
# for i, message in enumerate(st.session_state["chat_history"]):
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         if message["role"] == "assistant" and "audio_url" in message:
#             components.html(f'<audio controls src="{message["audio_url"]}" style="width: 100%;"></audio>')

# query = st.text_input(ui_texts["question_placeholder"])
# if st.button(ui_texts["button_text"]) and query:
#     with st.spinner("답변 생성 중..."):  # 답변 생성 중 스피너 표시
#         response, updated_history = ask_question(query, language, st.session_state["chat_history"], age_group)
#         st.session_state["chat_history"] = updated_history

#         audio_url = text_to_speech(response, language)

#         # 새로운 답변 표시 및 음성 출력
#         with st.chat_message("assistant"):
#             st.markdown(response)
#             components.html(f'<audio autoplay controls src="{audio_url}" style="width: 100%;"></audio>')
#             st.session_state["chat_history"][-1]["audio_url"] = audio_url
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langdetect import detect
import streamlit.components.v1 as components
from gtts import gTTS
import base64
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO

# CSV 파일 로드 (캐싱 적용)
@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

# 이미지 데이터 로드
csv_file_path = "./data/merged_museum_data.csv"
try:
    image_df = load_csv(csv_file_path)
except FileNotFoundError:
    st.error(f"❌ 오류: CSV 파일 '{csv_file_path}'을 찾을 수 없습니다.")
    image_df = pd.DataFrame()

# 모델 로드 (캐싱 적용)
@st.cache_resource
def load_model():
    MODEL_NAME = "Gwangwoon/muse2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    return tokenizer, model

tokenizer, model = load_model()

# 벡터DB 로드 (캐싱 적용)
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Wikipedia 검색 설정 (캐싱 적용)
@st.cache_resource
def load_wikipedia(language="ko"):
    wiki_api = WikipediaAPIWrapper(lang=language)
    wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api)
    return wikipedia_tool

# 텍스트를 음성 파일로 변환하는 함수 (gTTS 사용)
def text_to_speech(text, language_code="ko"):
    tts = gTTS(text=text, lang=language_code.split("-")[0])
    audio_file = f"output_{language_code}.mp3"
    tts.save(audio_file)
    with open(audio_file, "rb") as f:
        audio_data = f.read()
    audio_base64 = base64.b64encode(audio_data).decode()
    os.remove(audio_file)  # 임시 파일 삭제
    return f"data:audio/mp3;base64,{audio_base64}"

# 시스템 프롬프트 정의
SYSTEM_PROMPTS = {
    "ko": """
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
    - 관련된 이미지 URL이 있다면 참고용으로 함께 보여줘.

    답변 형식:
    1. 간결하고 핵심적인 답변을 가장 먼저 제시
    2. 이어서 배경 정보 또는 관련 유물 설명
    3. 출처 제공(가능한 경우), 중복 문장 금지
    4. 관련 이미지 URL (있는 경우)

    # **[연령대별 답변 지침]**
    # - 만약 질문자가 어린이일 경우, 쉽고 재미있는 단어를 사용하여 설명하고, 비유나 간단한 예를 들어 이해를 도우세요.
    # - 만약 질문자가 청소년일 경우, 학교 교육 과정과 연관된 내용이나 흥미를 가질 만한 정보를 포함하여 설명하세요.
    # - 만약 질문자가 성인일 경우, 역사적 맥락, 문화적 의미, 예술적 가치 등 심층적인 정보를 제공하세요.
    # - 만약 질문자가 노년층일 경우, 편안하고 친근한 어투를 사용하며, 과거의 경험이나 추억을 떠올릴 수 있는 내용을 곁들여 설명하세요.
    """,
    "en": """
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
    - Include relevant image URLs if available for reference.

    Answer Format:
    1. Present the concise and essential answer first
    2. Follow with contextual or background explanations
    3. Include sources if available, and avoid redundant sentences
    4. Include relevant image URLs (if available)

    # **[Age-Specific Response Guidelines]**
    # - If the questioner is a child, use simple and fun words to explain, and use analogies or simple examples to aid understanding.
    # - If the questioner is a teenager, include content related to the school curriculum or information that might be of interest.
    # - If the questioner is an adult, provide in-depth information such as historical context, cultural significance, and artistic value.
    # - If the questioner is an elderly person, use a comfortable and friendly tone, and include content that may evoke past experiences or memories.
    """,
    "ja": """
    あなたは国立中央博物館で働く、知的で親切なAIドーセントです。来館者がどの言語で質問しても、自動的に言語を判別し、その言語で自然かつ正確に答えてください。
    自分がAIであることは言わず、本物の博物館ガイドのように振る舞ってください。

    回答のルール：
    - 日本語で答えてください。
    - 情報は簡潔に、一度だけ伝え、繰り返さないでください。
    - 不自然な表現や機械的な言い回しは避け、温かく、親しみやすい口調を使ってください。
    - 質問の意図をまず理解しようとしてください。短い質問や曖昧な表現でも, 来館者の意図を推測してみてください。
    - 遺物を説明する際は, その歴史的背景, 製作方法, 文化的な意味, 出土場所などを簡潔に紹介してください。
    - 質問が不明確な場合は, まず内容を明確にしてもらうようお願いしてください。
    - 情報が不明な場合は, 「よくわかっていません」や「詳細は不明です」など, 正直に答えてください。
    - 必要に応じて関連する遺物や時代の情報を提案してください。
    - 無意味な繰り返しや決まり文句は絶対に避けてください。
    - 回答はRAG（検索拡張生成）に基づいて行い, 信頼できる情報源やリンクがあれば一緒に提示してください。
    - 関連する画像URLがある場合は、参考のために一緒に提示してください。

    回答形式：
    1. まず, 簡潔で重要な情報を先に述べる
    2. 次に, 背景や関連情報を説明する
    3. 可能であれば情報源を提示し, 重複表現は避ける
    4. 関連する画像URL（もしあれば）

    # **[年齢層別回答ガイドライン]**
    # - 質問者が子供の場合、簡単で面白い言葉を使って説明し、比喩や簡単な例を使って理解を助けてください。
    # - 質問者が十代の場合、学校のカリキュラムに関連する内容や興味を持ちそうな情報を含めて説明してください。
    # - 質問者が大人の場合、歴史的背景、文化的意義、芸術的価値など、より深い情報を提供してください。
    # - 質問者が高齢者の場合、快適で親しみやすい口調を使用し、過去の経験や思い出を想起させるような内容を添えて説明してください。
    """
}

# UI 텍스트 정의
UI_TEXTS = {
    "ko": {
        "title": "국립중앙박물관 AI 도슨트",
        "question_placeholder": "질문을 입력하세요:",
        "button_text": "질문하기",
        "greeting": "안녕하세요! 궁금한 점을 물어보세요.",
        "rerun_audio_button_text": "다시 듣기",
        "age_group_label": "연령대를 선택하세요:",
        "age_group_options": ["전체", "어린이", "청소년", "성인", "노년층"]
    },
    "en": {
        "title": "National Museum of Korea AI Docent",
        "question_placeholder": "Please enter your question:",
        "button_text": "Ask Question",
        "greeting": "Hello! Ask me anything",
        "rerun_audio_button_text": "Listen Again",
        "age_group_label": "Select Age Group:",
        "age_group_options": ["All", "Child", "Teenager", "Adult", "Elderly"]
    },
    "ja": {
        "title": "国立中央博物館 AI ドーセント",
        "question_placeholder": "質問を入力してください:",
        "button_text": "質問する",
        "greeting": "こんにちは！遺物について気になることを聞いてください。",
        "rerun_audio_button_text": "もう一度聞く",
        "age_group_label": "年齢層を選択してください:",
        "age_group_options": ["すべて", "子供", "青少年", "大人", "高齢者"]
    }
}

def select_system_prompt(language):
    return SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["ko"])

SIMILARITY_THRESHOLD = 0.7
MAX_HISTORY_MESSAGES = 10

def filter_similar_docs(docs):
    filtered_docs = [doc for doc in docs if getattr(doc, 'similarity', 1.0) >= SIMILARITY_THRESHOLD]
    return filtered_docs if filtered_docs else " 해당 정보는 없습니다."

# 질문과 이미지 설명을 비교하여 유사한 이미지 URL을 찾는 함수
def find_similar_image(query, image_df):
    if image_df.empty:
        return None

    try:
        # 1. 쿼리 및 이미지 설명을 임베딩
        embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        query_embedding = embedding_model.embed_query(query)
        image_embeddings = image_df['description'].apply(embedding_model.embed_query).tolist()

        # 2. 코사인 유사도 계산
        similarities = cosine_similarity([query_embedding], image_embeddings)[0]

        # 3. 가장 유사한 이미지 URL 찾기
        best_match_index = similarities.argmax()
        most_similar_image_url = image_df.iloc[best_match_index]['image_url']
        return most_similar_image_url
    except Exception as e:
        print(f"Error finding similar image: {e}")
        return None

def ask_question(query, language, chat_history, age_group=None):
    try:
        docs = retriever.get_relevant_documents(query)
        filtered_docs = filter_similar_docs(docs)

        if filtered_docs == " 해당 정보는 없습니다.":
            wikipedia_tool = load_wikipedia(language)
            answer = wikipedia_tool.run(query)
            image_url = None  # 위키피디아에서 찾은 경우 이미지 URL 없음
        else:
            context = "\n".join([doc.page_content for doc in filtered_docs])
            image_url = find_similar_image(query, image_df)  # 관련 이미지 검색

            # 프롬프트에 연령대 정보를 추가
            if age_group == "어린이":
                age_prompt = "질문자는 어린이입니다. 쉽고 재미있게 설명해주세요."
            elif age_group == "청소년":
                age_prompt = "질문자는 청소년입니다. 학교 교육과정과 관련된 내용이나 흥미로운 정보를 포함해주세요."
            elif age_group == "성인":
                age_prompt = "질문자는 성인입니다. 전문적이고 심층적인 정보를 제공해주세요."
            elif age_group == "노년층":
                age_prompt = "질문자는 노년층입니다. 편안하고 친근한 어투로 설명해주세요."
            else:
                age_prompt = ""

            system_prompt_with_age = SYSTEM_PROMPTS[language] + f"\n\n{age_prompt}"

            messages = [{"role": "system", "content": system_prompt_with_age}]
            messages += chat_history
            messages.append({"role": "user", "content": f"연관 정보:\n{context}\n\n질문: {query}\n답변:"})

            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,  # 답변 길이 제한
                top_p=0.9,
                temperature=0.3
            )
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = output_text.split("assistant")[-1].strip() if "assistant" in output_text else output_text.strip()

        # 이미지 URL을 답변에 추가
        if image_url:
            answer += f"\n\n참고 이미지: {image_url}"

        # 저장된 대화 이력에 추가
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer, "image_url": image_url if image_url else None})  # 이미지 URL 저장

        if len(chat_history) > MAX_HISTORY_MESSAGES:
            chat_history[:] = chat_history[-MAX_HISTORY_MESSAGES:]

        return answer, chat_history
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        return "⚠️ 오류가 발생했습니다. 다시 시도해 주세요.", chat_history

# Streamlit UI 구현
st.title("국립중앙박물관 AI 도슨트")

# 세션 상태에 대화 기록 초기화
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 언어 선택 버튼 (국기 아이콘과 함께)
language = st.radio(
    "언어를 선택하세요:",
    options=["ko", "en", "ja"],
    index=0,
    format_func=lambda x: {"ko": "🇰🇷 한국어", "en": "🇬🇧 영어", "ja": "🇯🇵 일본어"}.get(x, "한국어")
)

# 언어에 맞는 UI 문구 동적 설정
ui_texts = UI_TEXTS[language]

st.title(ui_texts["title"])
st.write(ui_texts["greeting"])

# 연령 선택 버튼 추가 (언어 설정 후 표시)
age_group = st.radio(
    ui_texts["age_group_label"],
    options=ui_texts["age_group_options"],
    index=0,
    horizontal=True
)

# 이전 대화 내용 표시 및 다시 듣기 버튼
for i, message in enumerate(st.session_state["chat_history"]):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if "image_url" in message and message["image_url"]:
                st.image(message["image_url"], caption="참고 이미지", width=200)  # 이미지 표시
            if "audio_url" in message:
                components.html(f'<audio controls src="{message["audio_url"]}" style="width: 100%;"></audio>')

query = st.text_input(ui_texts["question_placeholder"])
if st.button(ui_texts["button_text"]) and query:
    with st.spinner("답변 생성 중..."):  # 답변 생성 중 스피너 표시
        response, updated_history = ask_question(query, language, st.session_state["chat_history"], age_group)
        st.session_state["chat_history"] = updated_history

        # 음성 생성 및 출력
        audio_url = text_to_speech(response, language)

        # 새로운 답변 표시 및 음성 출력
        with st.chat_message("assistant"):
            st.markdown(response)
            if updated_history[-1]["image_url"]:
                st.image(updated_history[-1]["image_url"], caption="참고 이미지", width=200)  # 이미지 표시
            components.html(f'<audio autoplay controls src="{audio_url}" style="width: 100%;"></audio>')
            st.session_state["chat_history"][-1]["audio_url"] = audio_url
