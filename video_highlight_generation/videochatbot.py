import os

from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.agents.json_chat.prompt import TEMPLATE_TOOL_RESPONSE
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import ValidationError

SYSTEM_PROMPT = 'Assistant is a large language model with access to Tools.  Assistant should prioritize using a Tool to get up-to-date information.\n\nAssistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.\n\nAssistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.\n\nOverall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.'
FORMAT_PROMPT = 'TOOLS\n------\nAssistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:\n\n{tools}\n\nRESPONSE FORMAT INSTRUCTIONS\n----------------------------\n\nWhen responding to me, please output a response in one of two formats:\n\n**Option 1:**\nUse this if you want the human to use a tool.\nMarkdown code snippet formatted in the following schema:\n\n```json\n{{\n    "action": string, \\ The action to take. Must be one of {tool_names}\n    "action_input": string \\ The input to the action\n}}\n```\n\n**Option #2:**\nUse this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:\n\n```json\n{{\n    "action": "Final Answer",\n    "action_input": string \\ You should put what you want to return to use here\n}}\n```\n\nUSER\'S INPUT\n--------------------\nHere is the user\'s input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):\n\n{input}'
if not os.getenv("NVIDIA_API_KEY", False):
    raise EnvironmentError("NVIDIA_API_KEY not set")
try:
    llm = ChatNVIDIA(model="mixtral_8x7b",
                     # llm = ChatNVIDIA(model="playground_llama2_code_34b",
                     # llm = ChatNVIDIA(model='playground_steerlm_llama_70b',
                     temperature=0.1,
                     max_tokens=1000,
                     nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
                     streaming=True)
    print('Initialized NVIDIA LLM mixtral 8x7b model')
except ValidationError as ex:
    print(f"Exception in initialize_custom_agent. The API key provided is invalid\n: {ex}")
    raise ValueError(f"Exception in initialize_custom_agent. The API key provided is invalid\n: {ex}")
llm_with_stop = llm.bind(stop=["\nObservation"])
chat_template = ChatPromptTemplate.from_messages(
    [SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[],
                                                       template=SYSTEM_PROMPT)),
     MessagesPlaceholder(variable_name='chat_history', optional=True),
     HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input', 'tool_names', 'tools'],
                                                      template=FORMAT_PROMPT)),
     MessagesPlaceholder(variable_name='agent_scratchpad')]
)
prompt = chat_template
agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_messages(
                x["intermediate_steps"], template_tool_response=TEMPLATE_TOOL_RESPONSE
            )
        )
        | prompt
        | llm_with_stop
        | JSONAgentOutputParser()
)

# TODO define tools (it should interact with the DB
