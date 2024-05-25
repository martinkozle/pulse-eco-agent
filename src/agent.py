import datetime
from typing import cast

from devtools import debug
from langchain import hub
from langchain.agents import AgentExecutor, BaseMultiActionAgent, create_json_chat_agent
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.environ import OLLAMA_BASE_URL, OLLAMA_MODEL
from src.tools.datetime import (
    datetime_plus_days,
    now_plus_hours,
)
from src.tools.pulse_eco import (
    get_average_city_air_quality_on_date,
    get_average_location_air_quality_on_date,
    get_current_city_air_quality,
)

USER_CITY = "Skopje"

current_date = datetime.datetime.now(tz=datetime.UTC).date()

prompt = hub.pull("hwchase17/structured-chat-agent")

prompt.messages[0].prompt.template = (
    "Respond to the human as helpfully and accurately as possible."
    " You have access to the following tools:\n"
    "\n"
    "{tools}\n"
    "\n"
    "Use a json blob to specify a tool by providing an action key (tool name)"
    " and an action_input key (tool input).\n"
    "\n"
    'Valid "action" values: "Final Answer" or {tool_names}\n'
    "\n"
    "Provide only ONE action per $JSON_BLOB, as shown:\n"
    "\n"
    "```json\n"
    "{{\n"
    '  "action": $TOOL_NAME,\n'
    '  "action_input": $INPUT\n'
    "}}\n"
    "```\n"
    "\n"
    "Where $INPUT can also be an object with multiple keys for every tool parameter.\n"
    "\n"
    "\n"
    "Follow this format:\n"
    "\n"
    "Question: input question to answer\n"
    "Thought: consider previous and subsequent steps\n"
    "Action:\n"
    "```json\n"
    "$JSON_BLOB\n"
    "```\n"
    "Observation: action result\n"
    "... (repeat Thought/Action/Observation N times)\n"
    "Thought: I know what to respond\n"
    "Action:\n"
    "```json\n"
    "{{\n"
    '  "action": "Final Answer",\n'
    '  "action_input": "Final response to human"\n'
    "}}\n"
    "\n"
    f"Today's date is {current_date}.\n"
    f"The user is from the city of {USER_CITY}.\n"
    "Always first use Thought to plan your next steps and then use Action.\n"
    "You don't always have to use a tool,"
    ' but you must always end your turn with the "Final Answer" tool.\n'
    "Begin! Reminder to ALWAYS respond with a valid json blob of a single action."
    ' Do not forget the "```json"!'
    " Use tools if necessary. Respond directly if appropriate."
    " Format is Action:```json\n$JSON_BLOB```then Observation"
)

debug(prompt)

llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    temperature=0.8,
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

tools = [
    now_plus_hours,
    datetime_plus_days,
    get_current_city_air_quality,
    get_average_city_air_quality_on_date,
    get_average_location_air_quality_on_date,
]

agent = create_json_chat_agent(
    llm, tools, prompt, stop_sequence=["<|eot_id|>", "Observation:"]
)

agent_executor = AgentExecutor(
    agent=cast(BaseMultiActionAgent, agent),
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

chat_history: list[BaseMessage] = []


while (inp := input("Human: ")) not in {"!exit", "!quit", "/exit", "/quit"}:
    attempt = 0
    while True:
        try:
            response = agent_executor.invoke({
                "input": inp,
                "chat_history": chat_history,
            })
        except ValueError:
            if attempt >= 3:
                raise
            print("Output parser error, retrying...")
            attempt += 1
            continue
        break
    output = response["output"]
    chat_history += [HumanMessage(content=inp), AIMessage(content=output)]
    print("AI:", output)

print(chat_history)
print(
    "\n".join(f"{message.type.title()}: {message.content}" for message in chat_history)
)
