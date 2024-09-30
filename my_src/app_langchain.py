import streamlit as st
import json
import time
import os

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import AzureChatOpenAI

# Initialize the LLM
llm = AzureChatOpenAI(
    azure_endpoint = os.getenv("BASE_URL"),
    openai_api_version = os.getenv("API_VERSION"),
    deployment_name = os.getenv("DEPLOYMENT_NAME"),
    openai_api_key = os.getenv("API_KEY"),
    temperature = 0.2
)

# Create a prompt template for LangChain
prompt = PromptTemplate(template="{prompt}", input_variables=["prompt"])

# Create a LangChain pipeline
llm_chain = prompt | llm | JsonOutputParser()

# Create a prompt template for LangChain
prompt = PromptTemplate(template="{prompt}", input_variables=["prompt"])

def make_api_call(messages, max_tokens, is_final_answer=False):
    # print('--- make_api_call ---\n-- messages:', json.dumps(messages, indent=4))

    MAX_ATTEMPTS = 3
    for attempt in range(MAX_ATTEMPTS):
        try:
            # Create a prompt content string
            prompt_content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            print("-- prompt content:\n", prompt_content)

            # Generate response
            response = llm_chain.invoke({"prompt": prompt_content})
            # print('-- raw response:\n', response)

            # if it is not an array
            if not isinstance(response, list):
                response = [response]

            return response
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                if is_final_answer:
                    return {"title": "Error",
                            "content": f"Failed to generate final answer after {MAX_ATTEMPTS} attempts. Error: {str(e)}"}
                else:
                    return {"title": "Error", "content": f"Failed to generate step after {MAX_ATTEMPTS} attempts. Error: {str(e)}",
                            "next_action": "final_answer"}
            time.sleep(1)  # Wait for 1 second before retrying

def generate_response(prompt):
    messages = [
        {"role": "system", "content": """You are an expert AI assistant with advanced reasoning capabilities. Your task is to provide detailed, step-by-step explanations of your thought process. For each step:

1. Provide a clear, concise title describing the current reasoning phase.
2. Elaborate on your thought process in the content section.
3. Decide whether to continue reasoning or provide a final answer.

Response Format of Each Step:
Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' or 'final_answer')

Key Instructions:
- Employ at least 5 distinct reasoning steps.
- Acknowledge your limitations as an AI and explicitly state what you can and cannot do.
- Actively explore and evaluate alternative answers or approaches.
- Critically assess your own reasoning; identify potential flaws or biases.
- When re-examining, employ a fundamentally different approach or perspective.
- Utilize at least 3 diverse methods to derive or verify your answer.
- Incorporate relevant domain knowledge and best practices in your reasoning.
- Quantify certainty levels for each step and the final conclusion when applicable.
- Consider potential edge cases or exceptions to your reasoning.
- Provide clear justifications for eliminating alternative hypotheses. 

Example of a valid JSON response containing multiple steps as a JSON array:
...json
[{
    "title": "Initial Problem Analysis",
    "content": "To approach this problem effectively, I'll first break down the given information into key components. This involves identifying...[detailed explanation]... By structuring the problem this way, we can systematically address each aspect.",
    "next_action": "continue"
},
...
]
...
"""},
        {"role": "user", "content": prompt},
        {"role": "assistant",
         "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]

    steps = []
    step_count = 0
    total_thinking_time = 0
    final_content = None

    while True:
        start_time = time.time()
        more_steps = make_api_call(messages, 300)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        for step_data in more_steps:
            step_count += 1

            steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))
            messages.append({"role": "assistant", "content": json.dumps(step_data)})

            if step_data['next_action'] == 'final_answer':
                final_content = step_data['content']
                break

        if final_content != None:
            break

        # Yield after each step for Streamlit to update
        yield steps, None  # We're not yielding the total time until the end

    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the complete final answer based on your reasoning above."})

    start_time = time.time()
    final_steps = make_api_call(messages, 200, is_final_answer=True)
    # print('---------- final_steps:', json.dumps(final_steps))
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    for step_data in final_steps:
        if step_data['next_action'] == 'final_answer':
            final_content = step_data['content']
            break
        step_count += 1
        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))
        messages.append({"role": "assistant", "content": json.dumps(step_data)})

    steps.append(("Final Answer", final_content, thinking_time))

    yield steps, total_thinking_time

def main():
    st.set_page_config(page_title="LangChain OpenAI Reasoning Chains", page_icon="🧠", layout="wide")

    st.title("Using LangChain to create reasoning chains")

    st.markdown("""
    This is a prototype using LangChain's AzureChatOpenAI model to create reasoning chains for improved output accuracy.
    """)

    # Text input for user query
    user_query = st.text_input("Enter your query:", placeholder="e.g., How many 'R's are in the word strawberry?")

    if user_query:
        st.write("Generating response...")

        # Create empty elements to hold the generated text and total time
        response_container = st.empty()
        time_container = st.empty()

        # Generate and display the response
        for steps, total_thinking_time in generate_response(user_query):
            with response_container.container():
                for i, (title, content, thinking_time) in enumerate(steps):
                    if title.startswith("Final Answer"):
                        st.markdown(f"### {title}")
                        st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                    else:
                        with st.expander(title, expanded=True):
                            st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)

            # Only show total time when it's available at the end
            if total_thinking_time is not None:
                time_container.markdown(f"**Total thinking time: {total_thinking_time:.2f} seconds**")

def debug_main(user_query):
    # Generate and display the response
    for steps, total_thinking_time in generate_response(user_query):
        for i, (title, content, thinking_time) in enumerate(steps):
            if title.startswith("Final Answer"):
                print(f"### {title}")
                print(content.replace('\n', '<br>'))
            else:
                with st.expander(title, expanded=True):
                    print(content.replace('\n', '<br>'))

        # Only show total time when it's available at the end
        if total_thinking_time is not None:
            print(f"**Total thinking time: {total_thinking_time:.2f} seconds**")

if __name__ == "__main__":
    # debug_main("How many letters in word reasoning?") # for testing
    main()

# testing question examples:
# - Answer in Chinese: 农夫需要把狼、羊和白菜都带过河，但每次只能带一样物品，而且狼和羊不能单独相处，羊和白菜也不能单独相处，问农夫该如何过河？
# - Compare two real numbers: 7.11 and 7.5
# - How many letters in word xxx?
# - How many letter r in word xxx?
