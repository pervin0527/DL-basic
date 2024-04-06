import os
from langchain_community.llms import Ollama


llm = Ollama(temperature=0, model="ob-llama2-13b", keep_alive=0)

context = "미국의 역사에 대해 한국어로 설명해라."
summary_query = f"""Please briefly summarise the text separated by <<< and >>> into five key points. Please output it in the following format
format:
1. summary1
2. summary2
…
text:
<<<
{context}
>>>
"""
result = llm.invoke(summary_query)
print(result) 