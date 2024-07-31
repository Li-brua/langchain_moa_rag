from langchain.vectorstores.chroma import Chroma
from chain import moaChat
from get_embedding_function import embedding_function

from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from util_output_parser import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
# from tools import get_control_name
CHROMA_PATH = "chroma"


def main():
    query_text = ''' question="How much total money does a player start with in Monopoly? (Answer with the number only)",'''
    #expect result 1500
    # question = "How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
    # expected_response = "10 points",
    query_rag(query_text)


def get_prompt():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant.",
            ),
            ("assistant", "{assistant}"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    return prompt

def query_rag(query_text: str):
    # Prepare the DB.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function())

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    llms = moaChat()
    llm_with_tools = llms.bind(functions=[])

    agent = (
            {
                "input": lambda x: x["user"],
                "assistant": lambda x: x["assistant"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | get_prompt()
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=[],  verbose=False)
    response_text = agent_executor.invoke({"assistant":context_text,"user":query_text})

    print(response_text['output'])
    return response_text


if __name__ == "__main__":
    main()
