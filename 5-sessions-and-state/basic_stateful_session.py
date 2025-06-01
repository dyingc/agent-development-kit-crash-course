import uuid

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from question_answering_agent import question_answering_agent
from pydantic import BaseModel, Field
from typing import Annotated
import os

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "question_answering_app")
USER_ID = os.getenv("USER_ID", "brandon_hancock")
USER_NAME = os.getenv("USER_NAME", "Brandon Hancock")

# Create a new session service to store state
session_service_stateful = InMemorySessionService()

initial_state = {
    "user_name": USER_NAME,
    "user_preferences": """
        I like to play Heroes of Might and Magic III, Chess, Jogging, and Tennis.
        My favorite food is Chinese.
        我很喜欢刘慈欣的《三体》。于和伟演的很不错，也很忠实于原作。其实，85版的《射雕》也不错的，尤其是里面的主题曲。
        Loves it when people like what he shares.
    """,
}

# Create a NEW session
class TRIAL(BaseModel):
    dob: Annotated[str, Field(description="Date of Birth in YYYY-MM-DD format")]
    session_id: Annotated[str, Field(description="Unique session identifier in UUID format")]

def create_new_session(trial: TRIAL):
    state = initial_state.copy()
    state["dob"] = trial.dob # Add the date of birth to the session state
    SESSION_ID = trial.session_id
    session_service_stateful.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=state,
    )
    print("CREATED NEW SESSION:")
    print(f"\tSession ID: {SESSION_ID}")
    print(f"\tUser ID: {USER_ID}")
    print(f"\tApp Name: {APP_NAME}")
    print(f"\tBirth Date: {trial.dob}\n\n")

# Create multiple sessions with different birth dates
trial1 = TRIAL(dob="1995-05-15", session_id=str(uuid.uuid4()))
create_new_session(trial1)
trial2 = TRIAL(dob="1985-10-20", session_id=str(uuid.uuid4()))
create_new_session(trial2)
trial3 = TRIAL(dob="2000-01-01", session_id=str(uuid.uuid4()))
create_new_session(trial3)

runner = Runner(
    agent=question_answering_agent,
    app_name=APP_NAME,
    session_service=session_service_stateful,
)

new_message = types.Content(
    role="user", parts=[types.Part(text=f"My name is {os.environ['USER_FIRST_NAME']} and what's my favorite TV show and music? What you can deduce, from my year-of-birth?")],
)

chosen_trial = trial2  # Choose the second trial for this example

for event in runner.run(
    user_id=USER_ID,
    session_id=chosen_trial.session_id,
    new_message=new_message,
):
    if event.is_final_response():
        if event.content and event.content.parts:
            print(f"Final Response: {event.content.parts[0].text}")

print("==== Session Event Exploration ====")
session = session_service_stateful.get_session(
    app_name=APP_NAME, user_id=USER_ID, session_id=chosen_trial.session_id
)

# Log final Session state
print("=== Final Session State ===")
for key, value in session.state.items():
    print(f"{key}: {value}")

pass