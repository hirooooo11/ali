import os
import sys
from openai import OpenAI
from retriever import Retriever


client = OpenAI(
    # 若没有配置环境变量，请将下行替换为：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY", "your_qwen_api_key_here"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def call_qwen(context_str, user_question):
    prompt = f"""You are a data analysis assistant.
Answer the question ONLY based on the provided context.
Context:
{context_str}
Question:
{user_question}
Answer:
要求:
• 只能基于提供的context 回答
• 不允许编造信息
• 回答控制在150-200字"""


    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return completion.choices[0].message.content


def main():
    print("Ad Conversion Assistant Ready.")
    

    retriever = Retriever()



    while True:
        try:
            user_question = input("\nAsk a question:\n> ")
            if not user_question.strip():
                continue
                

            retrieved = retriever.search(user_question, top_k=3)
            if not retrieved:
                continue
                

            context_passages = []
            sources = set()
            for item in retrieved:
                context_passages.append(f"[{item['source']}] {item['content']}")
                sources.add(item['source'])
                
            context_str = "\n".join(context_passages)
            

            answer = call_qwen(context_str, user_question)
            

            print(f"\nAnswer:\n{answer}")
            print("\nSources:")
            for s in sources:
                print(f"- {s}")
                

        except KeyboardInterrupt:
            sys.exit(0)

if __name__ == "__main__":
    main()