import os
from langchain_community.llms import Ollama

# Ollama 객체 생성
# llm = Ollama(temperature=0, model="ob-llama2-13b", keep_alive=0)
llm = Ollama(model="llama2", temperature=0)


# 분석할 키워드 리스트
# context = '"프론트엔드 엔지니어는 이런 업무를 하시게 됩니다!,자사의 Web/Mobile Application 및 기업용 SaaS 플랫폼 설계 및 개발,최고의 고객경험과 안정적인 서비스 제공을 위한 개선""우리는 이런 분들과 함께하고 싶어요!"",사용자가 겪는 문제를 IT 기술을 통해 주도적으로 해결하려고 노력하는 사람,새로운 기술에 대해 거부감이 없고, 끊임없이 도전하길 좋아하는 사람,팀원들과 기술을 공유하고 함께 성장하는걸 중요하게 생각하는 사람,되는걸 만드는 것 보다 잘 되는 걸 만들기 위해 노력하는 사람,자신의 의견을 적극적으로 말할 수 있는사람""우리팀은 이런 기술들을 활용합니다!"",Front-end: React.js, Next.js, TypeScript, JavaScript, Zustand, Recoil, Redux, emotion.js, SCSS, Webpack,Back-end: TypeScript (Nest.js, Express.js in Node.js),Native App: Flutter, React-Native,Block Chain: Klyatn, Polygon, Luniverse, LINE Blockchain, Solidity,Database: MySQL, MongoDB, DynamoDB,Infra: AWS (EC2, RDS, Lambda, SQS, S3, Amplify, Code Depoly, etc.), Terraform, k8s,Co-work: Github(Mono-Repo, Git-Flow), Slack, Notion, Figma,,개발 경력 5년 이상 혹은 그에 준하는 역량을 갖추신 분,HTML, CSS 및 javascript (ES5 및 ES6+)에 대한 충분한 이해와 개발 경험이 있으신 분,React.js 기반 모던 프론트엔드 생태계에 대한 충분한 이해와 개발 경험이 있으신 분,React-Query, Redux, Zustand, Recoil 등 상태관리 라이브러리에 대한 충분한 이해와 개발 경험이 있으신 분,RESTful API 서버와의 비동기 통신을 이해하고 있으신 분,SPA 및 반응형, 모바일 웹에 대한 이해와 경험이 있으신 분,Styled Component, SCSS, LESS 등 CSS 전처리기 개발 경험이 있으신 분,Webpack, Babel 등을 활용한 프론트엔드 환경 셋팅 경험이 있으신 분,Git을 이용한 버전관리에 대한 이해가 있으신 분,,컴퓨터공학 전공 및 관련학과 우대,디자인 컴포넌트 시스템 환경에서 개발 경험이 있으신 분,웹/모바일 서비스 성능 최적화 경험이 있으신 분,HTTP 통신 및 웹 보안에 대한 이해가 있으신 분,프론트엔드 테스트 및 배포 자동화 경험이 있으신 분,SSR(Server Side Rendering) 개발 경험이 있으신 분,Flutter 등의 Cross Platform 개발 및 Hybrid App 개발 경험이 있으신 분"'
context = ",자율주행 연구 및 디바이스 시그널 데이터를 처리하는 분석 서비스 백엔드 설계 및 개발,대용량 데이터 스트리밍 처리 및 프로세싱 기능 개발,1. 자격 요건 ,Spring Framework, JVM 기반 SW 설계 경험을 가진 분 (3년 차 이상),대규모 트래픽 처리 가능한 시스템을 설계하고 운영해본 경험이 있으신 분,고가용성의 확장 가능한 시스템에 대해 끊임 없이 고민할 수 있는 사람,변화를 두려워하지 않고, 서비스를 지속적으로 발전시키고 싶은 분,커뮤니케이션을 사랑하며, 개발 문화를 개선하고 함께 발전하고자 하시는 분2. 주요 기술 ,Java, Spring boot, JPA, QueryDSL, MySQL, Redis, RabbitMQ, Kubernetes3. 업무 툴 ,GitHub, GitHub Action, Jira, Notion,,컴퓨터공학 전공자,Kubernetes 기반 개발 경험,오픈소스 활용 경험,새로운 기술에 대한 끊임없는 도전과 성장을 할 수 있는 분"

# 프롬프트 작성
summary_query = f"""
Based on the content of the selected job postings, can you create a dictionary that shows what kind of people this company is trying to hire by grouping the main fields as keys and the skills connected to the main fields as list-type values?
i don't need a "Expanation". just provide dict.

format : 
 - main field
 - related skills

…
text:
<<<
{context}
>>>
"""

# Ollama 모델 호출 및 결과 출력
result = llm.invoke(summary_query)
print(result)
