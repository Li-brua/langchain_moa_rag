import time
import together
from together import Together

api = "b202d017492901648e70a4b6a30b07949b88b7fb705ceca9a25fba7909aed06b"

client = Together(api_key=api)

reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B-Chat",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct",
]

aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"

aggregator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""
layers = 3


def get_final_system_prompt(system_prompt, results):
    """Construct a system prompt for layers 2+ that includes the previous responses to synthesize."""
    return (
        system_prompt
        + "\n"
        + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])
    )


def run_llm(model, user_prompt, prev_response=None):
    """Run a single LLM call with a model while accounting for previous responses + rate limits."""
    messages = (
        [
            {
                "role": "system",
                "content": get_final_system_prompt(
                    aggregator_system_prompt, prev_response
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        if prev_response
        else [{"role": "user", "content": user_prompt}]
    )
    for sleep_time in [1, 2, 4]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=512,
            )
            print("Model: ", model)
            return response.choices[0].message.content
        except together.error.RateLimitError as e:
            print(e)
            time.sleep(sleep_time)
    return None


def moa_generate(user_prompt):
    """Run the main loop of the MOA process and return the final result."""
    results = [run_llm(model, user_prompt) for model in reference_models]

    for _ in range(1, layers - 1):
        results = [run_llm(model, user_prompt, prev_response=results) for model in reference_models]

    finalStream = client.chat.completions.create(
        model=aggregator_model,
        messages=[
            {
                "role": "system",
                "content": get_final_system_prompt(aggregator_system_prompt, results),
            },
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )

    final_response = ""
    for chunk in finalStream:
        final_response += chunk.choices[0].delta.content or ""
    # print(final_response)
    return final_response
    # return finalStream


# Example usage:
if __name__ == "__main__":
    user_prompt = "What are 3 fun things to do in SF?"
    # user_prompt = str()
    final_result = moa_generate(user_prompt)
    print(final_result)