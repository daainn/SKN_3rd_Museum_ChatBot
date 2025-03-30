import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langdetect import detect

# 라이브러리 설치 (필요한 경우 주석 해제)
# !pip install -U langchain-community langchain-huggingface bitsandbytes

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
def load_wikipedia():
    wiki_api = WikipediaAPIWrapper(lang="ko")
    wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api)
    return wikipedia_tool

wikipedia_tool = load_wikipedia()

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

    답변 형식:
    1. 간결하고 핵심적인 답변을 가장 먼저 제시
    2. 이어서 배경 정보 또는 관련 유물 설명
    3. 출처 제공(가능한 경우), 중복 문장 금지

    예시:
    [질문] 이 유물은 어떤 시대에 만들어졌나요?
    [답변] 이 유물은 고려 시대(918~1392년)에 제작된 청자로, 왕실에서 의례용으로 사용되었습니다. 강진 지역에서 출토되었으며, 특유의 푸른빛과 정교한 문양이 특징입니다.
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

    Answer Format:
    1. Present the concise and essential answer first
    2. Follow with contextual or background explanations
    3. Include sources if available, and avoid redundant sentences

    Examples:
    [Question] When was this artifact made?
    [Answer] This artifact is a celadon piece from the Goryeo Dynasty (918–1392), traditionally used in royal rituals. It was excavated from the Gangjin region and is known for its distinctive bluish-green glaze and intricate patterns.
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

    回答形式：
    1. まず, 簡潔で重要な情報を先に述べる
    2. 次に, 背景や関連情報を説明する
    3. 可能であれば情報源を提示し, 重複表現は避ける

    例：
    ［質問］この遺物はいつの時代に作られたものですか？
    ［回答］この遺物は高麗時代（918～1392年）に制作された青磁で, 왕실의 의식에 사용되었던 것으로 추정됩니다. 전라남도 강진 지역에서 출토되었으며, 독특한 청록색 유약과 정교한 문양이 특징입니다.
    """
}

def select_system_prompt(language):
    return SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["ko"])

SIMILARITY_THRESHOLD = 0.7
MAX_HISTORY_MESSAGES = 10
chat_history = []

def filter_similar_docs(docs):
    filtered_docs = [doc for doc in docs if getattr(doc, 'similarity', 1.0) >= SIMILARITY_THRESHOLD]
    return filtered_docs if filtered_docs else " 해당 정보는 없습니다."

def ask_question(query):
    try:
        docs = retriever.get_relevant_documents(query)
        filtered_docs = filter_similar_docs(docs)

        if filtered_docs == " 해당 정보는 없습니다.":
            return wikipedia_tool.run(query)

        context = "\n".join([doc.page_content for doc in filtered_docs])

        messages = [{"role": "system", "content": select_system_prompt(detect(query))}]
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

        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})

        if len(chat_history) > MAX_HISTORY_MESSAGES:
            chat_history[:] = chat_history[-MAX_HISTORY_MESSAGES:]

        return answer
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        return "⚠️ 오류가 발생했습니다. 다시 시도해 주세요."

# Streamlit UI 구현
st.title(" 국립중앙박물관 AI 도슨트")
st.write("안녕하세요! 유물에 대한 궁금한 점을 물어보세요.")

query = st.text_input("질문을 입력하세요:")
if st.button("질문하기") and query:
    with st.spinner("답변 생성 중..."): # 답변 생성 중 스피너 표시
        response = ask_question(query)
        st.write(" 답변:", response)

