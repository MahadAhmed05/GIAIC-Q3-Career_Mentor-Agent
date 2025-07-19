import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
# === Load environment variables ===
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file.")

# === Set up Gemini-compatible client ===
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# === Tool function ===
def get_career_roadmap(field: str) -> str:
    roadmaps = {
        "software engineering": "1. Learn Python or JavaScript\n2. Study data structures and algorithms\n3. Build real projects\n4. Contribute to open source\n5. Apply to internships/jobs",
        "data science": "1. Learn Python and statistics\n2. Study machine learning basics\n3. Practice on datasets (e.g. Kaggle)\n4. Build a portfolio\n5. Apply for junior roles",
        "medicine": "1. Take pre-med courses\n2. Prepare for MCAT\n3. Attend medical school\n4. Do clinical rotations\n5. Specialize",
    }
    return roadmaps.get(field.lower(), f"No roadmap found for '{field}'. Try asking about 'software engineering' or 'data science'.")

# === Agents ===
CareerAgent = Agent(
    name="CareerAgent",
    instructions="Ask the user about their interests and suggest suitable career paths based on that."
)

SkillAgent = Agent(
    name="SkillAgent",
    instructions="Provide a detailed skill roadmap for a chosen career field using the 'get_career_roadmap' tool.",
    tools={"get_career_roadmap": get_career_roadmap}
)

JobAgent = Agent(
    name="JobAgent",
    instructions="List real-world job roles and example job titles related to the user's chosen field."
)

# === Chat start ===
@cl.on_chat_start
async def start():
    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)
    cl.user_session.set("current_agent", CareerAgent)
    await cl.Message(content="👋 Welcome to Career Mentor AI!\nTell me about your interests and I'll help you explore a career path.").send()

# === Message handling ===
@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content": message.content})

    user_input = message.content.lower()

    # === Agent handoff logic ===
    if "skill" in user_input or "learn" in user_input or "roadmap" in user_input:
        agent = SkillAgent
    elif "job" in user_input or "role" in user_input or "title" in user_input:
        agent = JobAgent
    else:
        agent = CareerAgent

    cl.user_session.set("current_agent", agent)

    msg = cl.Message(content="")
    await msg.send()

    try:
        result = Runner.run_streamed(agent, history, run_config=cast(RunConfig, config))
        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, "delta"):
                await msg.stream_token(event.data.delta)

        history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("chat_history", history)

    except Exception as e:
        await msg.update(content=f"❌ Error: {str(e)}")
        print(f"Error: {e}")
