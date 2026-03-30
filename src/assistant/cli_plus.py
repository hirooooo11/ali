import os
import sys
import argparse
from openai import OpenAI
from retriever import Retriever


client = OpenAI(
    # 若没有配置环境变量，请将下行替换为：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY", "your_qwen_api_key_here"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def call_qwen(retrieved_passages, user_question):
    prompt = f"""You are a professional data analysis assistant.
Answer the question ONLY based on the provided context.
Requirements:
- Be concise and clear
- Use business-friendly language
- Do NOT make up information
- If unsure, say "Not enough information"
Context:
{retrieved_passages}
Question:
{user_question}
Answer:"""


    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return completion.choices[0].message.content


def process_question(retriever, user_question, output_file=None):
    retrieved = retriever.search(user_question, top_k=3)
    
    if not retrieved:
        answer = "Not enough information"
        sources_list = []
    else:
        context_passages = []
        source_dict = {}
        for item in retrieved:
            context_passages.append(f"[{item['source']}] {item['content']}")
            

            src = item['source']
            title = item.get('title', 'General')
            if src not in source_dict:
                source_dict[src] = title
            

        retrieved_passages = "\n".join(context_passages)
        answer = call_qwen(retrieved_passages, user_question)
        
        sources_list = []
        for src, title_str in source_dict.items():
            sources_list.append(f"- {src} ({title_str})")


    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Question\n{user_question}\n\n")
            f.write(f"# Answer\n{answer}\n\n")
            f.write("# Sources\n")
            for s in sources_list:
                f.write(f"{s}\n")
    else:
        print(f"\nAnswer:\n{answer}")
        print("\nSources:")
        for s in sources_list:
            print(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    retriever = Retriever()


    if args.question:
        process_question(retriever, args.question, args.output)
    

    else:
        print("Ad Conversion Assistant Ready.")
        while True:
            try:
                user_question = input("\nAsk a question:\n> ")
                if not user_question.strip():
                    continue
                    
                process_question(retriever, user_question)

            except KeyboardInterrupt:
                sys.exit(0)

if __name__ == "__main__":
    main()