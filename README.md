# **SKN09-3rd-5Team** 

> SK네트웍스 Family AI 캠프 9기 3차 프로젝트
> 개발기간: 25.03.18 - 25.03.31

<br>

---

# 📚 Contents

1. 팀 소개
2. 프로젝트 개요
3. 기술 스택 및 사용 모델
4. 시스템 아키텍처
5. WBS
6. 요구사항 명세서
7. 수집한 데이터 및 전처리 요약
8. DB 연동 구현 코드
9. 테스트 계획 및 결과 보고서
10. 진행 과정 중 프로그램 개선 노력
11. 수행결과
12. 결론
13. 개선 사항
14. 한 줄 회고
<br>
<br>

---

# 1. Introduce Team

#### 💡팀명: MUSE (Multilingual Universal Storyteller Engine)
#### 💡프로젝트명: LLM기반 박물관 도슨트 질의응답 서비스
<br>

##### ⬇️팀원 소개 ⬇️

<table align="center" width="100%">
  <tr>
    <td align="center" width="20%">
      <a href="https://github.com/youngseo98"><b>@김영서</b></a>
    </td>
    <td align="center" width="20%">
      <a href="https://github.com/Leegwangwoon"><b>@이광운</b></a>
    </td>
    <td align="center" width="20%">
      <a href="https://github.com/daainn"><b>@이다인</b></a>
    </td>
    <td align="center" width="20%">
      <a href="https://github.com/ohback"><b>@임수연</b></a>
    </td>
    <td align="center" width="20%">
      <a href="https://github.com/SIQRIT"><b>@조이현</b></a>
    </td>
  </tr>
  <tr>
    <td align="center"><img src="./readme_image/영서.jpg" width="50px" alt="김영서" /></td>
    <td align="center"><img src="./readme_image/광운.jpg" width="50px" alt="이광운" /></td>
    <td align="center"><img src="./readme_image/다인.png" width="100px" alt="이다인" /></td>
    <td align="center"><img src="./readme_image/수연.jpg" width="120px" alt="임수연" /></td>
    <td align="center"><img src="./readme_image/이현.jpg" width="100px" alt="조이현" /></td>
  </tr>
</table>

<br>

---


# 2. Project Overview
### ✅ 프로젝트 소개
해당 프로젝트에서는 LLM(대형 언어 모델)을 기반으로 **박물관 도슨트 질의응답 챗봇 서비스**를 개발하였습니다. 사용자가 박물관의 작품 정보가 궁금할 때, 다양한 언어로 문화적 정보와 이야기를 제공할 수 있는 시스템을 구축하여, 더 많은 사람들이 세계 여러 문화를 쉽고 재미있게 접할 수 있도록 돕습니다. 

### ✅ 프로젝트 필요성

현재 국립중앙박물관 애플리케이션에는 ‘스마트전시관 챗봇’ 서비스가 제공되고 있다.이 챗봇은 박물관 내 건물과 작품의 위치 안내에는 특화되어 있으나, **작품에 대한 설명이 부족하다는 한계가 존재한다.** 따라서 관람객들이 박물관 작품을 보다 쉽고 깊이 있게 이해할 수 있도록, **작품 설명 중심의 도슨트 질의응답 챗봇의 도입이 필요한 상황**이다.

또한 기존 챗봇 서비스는 다국어 지원이 제공되지 않아 외국인 관람객의 접근성이 낮다는 점도 개선이 요구된다. 이에 따라 **다양한 언어를 지원하는 질의응답형 챗봇 개발이 필요**하다.

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="./readme_image/필요성1.jpg" width="200">
  <img src="./readme_image/필요성2.jpg" width="200">
</div>


### ✅ 프로젝트 목표
- 전시품 설명에 특화된 질의응답형 챗봇 서비스 개발
- 다국어 지원을 통한 관람객 접근성 및 경험 향상
- 음성 기반의 전시품 설명 기능 제공
- 자연어 처리 기반의 연령 맞춤형 콘텐츠 제공


<br>

---

# 3. Technology Stack & Models

## ✅ 기술 스택 및 사용한 모델


| **Language** | **Development** | **Embedding Model** | **Vector DB** | **LLM Model** | **Framework** | **Demo** | **Collaboration Tool** |
|--------------|------------------|----------------|----------------|----------------|----------------|-------------|--------------------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | ![VS Code](https://img.shields.io/badge/-VS%20Code-007ACC?logo=visualstudiocode&logoColor=white)<br>![Colab](https://img.shields.io/badge/-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)<br>![RunPod](https://img.shields.io/badge/-RunPod-5F43DC?logo=cloud&logoColor=white) | ![Hugging Face](https://img.shields.io/badge/-HuggingFace-FFD21F?logo=huggingface&logoColor=black) | ![FAISS](https://img.shields.io/badge/-FAISS-009999?logo=meta&logoColor=white) | ![Qwen](https://img.shields.io/badge/-Qwen-8A2BE2?logo=alibaba&logoColor=white)<br>![Gemma V3](https://img.shields.io/badge/-Gemma%20V3-4285F4?logo=google&logoColor=white) | ![LangChain](https://img.shields.io/badge/-LangChain-F9AB00?logo=LangChain&logoColor=white) | ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white) | ![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)<br>![Git](https://img.shields.io/badge/-Git-F05032?logo=git&logoColor=white)<br>![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white) |

<br><br>

# 4. 시스템 아키텍처
<img src="" width="700" height="600">

<br><br>

# 5. WBS
<img src="" width="700" height="600">


<br><br>

# 6. 요구사항 명세서



<br><br>

# 7. 수집한 데이터 및 전처리 요약
- **데이터 수집**: 다양한 박물관의 전시품 정보, 관련된 문화적 배경 데이터, 역사적 사실 등을 수집
- **전처리**: 텍스트 데이터 정제, 중복 제거, 언어별 번역 확인
- **기타**: 각 전시물에 대한 이미지 및 메타데이터도 함께 수집하여, 다양한 정보를 제공할 수 있도록 하였습니다.

-> 데이터 파인튜닝시는 한자 및 특수문자를 제거 (이유는 특수문자가 포함된 데이터로 파인튜닝 시킨 후 질의를 해봤을 때 특수문자가 같이 출력돼서 질의가 제대로 되지 않거나 특수문자가 반복적으로 출력되는 경우가 생겼기 때문)
-> rag를 붙힐 때는 한자 및 특수문자를 제거하지 않았잖아 이유가 한자와 특수문자를 제거했을 때 문맥 이해가 어려울 수 있기 때문에 붙혔음

<br><br>

# 8. DB 연동 구현 코드 (링크만)


<br><br>

# 9. 테스트 계획 및 결과 보고서


<br><br>

# 10. 진행 과정 중 프로그램 개선 노력
- 허깅 페이스 모델 링크도 넣기

<br><br>

# 11. 수행결과(테스트/시연 페이지)


<br><br>

# 12. 결론

<br><br>

# 13. 업데이트

<br><br>

# 14. 한 줄 회고
- 김영서: 
- 이광운:
- 이다인:
- 임수연:
- 조이현: