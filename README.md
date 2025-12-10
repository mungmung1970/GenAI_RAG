# 1 과제 개요

- LLM이 외부 문서의 정보를 참고하여 답변할 수 있도록 RAG를 구현해 보는 미션입
- LangChain을 이용해 RAG 시스템을 구현한 뒤, 사용된 문서와 관련된 질문을 하고 적절한 답변이 나오는지 확인

- 사용 데이터셋
  - 국세청에서 발간한 2024년 연말정산 신고 안내 문서를 활용
  - 연말정산 절차, 각종 공제 항목, 유의사항은 물론이고 2024년 기준으로 개정된 세법에 대한 정보도 있어서 RAG 시스템을 구현하고 검증하는 데 적합한 문서임

- 가이드라인
|구분|내용|
|-|-|
|문서 로드 및 청킹(Chunking)|- 먼저 사용할 문서를 불러오고, 검색 효율을 높이기 위해 문서를 적절한 길이로 나누는 청킹 작업을 수행<br>-다양한 청킹 옵션으로 실험을 반복하면서 최적의 옵션 탐색(필요한 경우 단순히 문자 수로 문서를 나누지 않고, 문서의 구조나 의미 고려 필요)|
|임베딩 생성 및 벡터 데이터베이스에 저장|-각 청크마다 임베딩을 생성하고, 이를 검색할 수 있도록 벡터 데이터베이스에 저장<br>-Hugging Face에서 어떤 모델을 사용하는지에 따라 검색 성능이 크게 달라질 수 있음<br>-특히 한국어 문서를 사용할 경우에는 모델 선정에 언어까지 고려해야 해요.벡터 데이터베이스의 종류도 다양<br>-구현의 편의성이나 검색 성능을 종합적으로 고려해 선택|
|언어 모델 및 토크나이저 설정|-Hugging Face에서 어떤 모델을 사용하면 좋을지 결정<br>-특히 한국어로 질문하고 응답받고 싶을 경우에는 언어도 고려하여 모델을 선정<br>-필요한 경우 양자화를 통해 메모리 부하를 줄이고 응답 속도 향상<br>-Temperature, penalty 등 텍스트 생성과 관련된 다양한 옵션을 적절한 값으로 설정|
|RAG 구현|-사용자의 질문이 들어왔을 때, 연관된 문서 청크를 찾아 맥락으로 활용해 답변을 생성하는 RAG 시스템을 구현|
|다양한 질문 입력 및 성능 평가|-RAG 시스템에 여러 질문을 던져 보면서 적절한 답변이 나오는지 평가<br>-질문은 문서 내용에 기반하여 답변의 정확성을 검증할 수 있는 질문이어야 좋음<br>-예를 들어 연말 정산 문서로 RAG 시스템을 구현했을 경우, 다음과 같은 질문을 해 볼 수 있음<br>&nbsp;&nbsp;&nbsp;&nbsp;-연말 정산 때 비거주자가 주의할 점을 알려 줘.<br>&nbsp;&nbsp;&nbsp;&nbsp;-2024년 개정 세법 중에 월세와 관련한 내용이 있을까?|

- (심화) 고급 RAG 기법 실험: 기본 RAG 구현을 완료했다면, 더 나아가 다양한 고급 기법들을 구현해 보고 성능이 나아지는지 확인
  - Hybrid searching, multi-query retrieval, contextual compression, reranking 등을 실험해 볼 수 있습니다.
- (심화) Hugging Face 외의 LLM API 실험: 여유가 있다면, OpenAI API 등 Hugging Face가 아닌 LLM API를 사용해 RAG 시스템을 만들어 보고, 성능을 비교

# 2 수행환경
- local에서 venv환경을 구성해서 수행
- https://github.com/mungmung1970/GenAI_RAG

## 2.1 폴더 구조
![image_01](images/image_01.png)

## 2.2 주요 설치 패키지
- 상세설치 내역은 requirements.txt파일 참조

## 2.3 벡터DB - Elastic Search/Kibana
![image_02](images/image_02.png)

## 2.4 실험모델
|구분|모델|비고|
|-|-|-|
|임베딩|text-embedding-3-small||
||bge-m3|다국어 처리가 우수하다고 하나 속도가 느림|
|Reranker|gpt-4o-mini||
|답변생성LLM|gpt-4o-mini|openAI API사용|
||deepseek-r1:1.5b|Local Ollama API사용(영어로 답변-속도 빠름, 한글 답변 실패)|
||mistral:7b|Local PC Ollama API사용(timeout - 시간 연장, MAX_CONTEXT_CHARS = 8000로제한-속도 매우 느림|
||llama3:8b-instruct-q4_0|Local  PC Ollama API사용-mistral과 동일 조건-속도 매우 느림, newline이 잘 되지 않아 잘못 이해할 소지가 있음|

## 2.5 Local LLM모델
- ollama list
  
|NAME|ID|SIZE|
|-|-|-|
|llama3:8b-instruct-q4_0|365c0bd3c000|4.7 GB|
|llama3:instruct|365c0bd3c000|4.7 GB|
|mistral:7b|6577803aa9a0|4.4 GB|
|deepseek-r1:1.5b|e0979632db5a|1.1 GB|
|deepseek-r1:14b|c333b7232bdb|9.0 GB|

# 3 수행내역 및 결과
## 3.1 데이터 로드(1loader.py)
- PyPDFLoader, PDFPlumberLoader 두개 방식 비교
- txt, html, json파일로 저장
- PDFPlumberLoader의 경우 표도 같이 생성하는 방식으로 저장하였으나 대부분의 표가 인식되지 않았고, 표의 셀순서가 맞지 않게 텍스트도 저장됨
- 따라서 최종PyPDFLoader를 사용하여json파일로 저장
<br>**상용 솔루션 중에는 표인식 및 Markdown 또는 HTML로 저장, 그림인식 및 캡션 저장 등이 가능한 것이 있는 것으로 알고 있음**
