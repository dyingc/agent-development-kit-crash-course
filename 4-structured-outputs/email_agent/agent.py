from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field, EmailStr
from typing import List, Annotated, Optional
from langchain.output_parsers import PydanticOutputParser

# --- Define Output Schema ---
# Note: Nested structures like lists of objects are not supported by the agent
# class Receiver(BaseModel):
#     name: Annotated[Optional[str], Field(default=None, description="Name of the email recipient")]
#     email: Annotated[EmailStr, Field(description="Email address of the recipient")]

class EmailContent(BaseModel):
    subject: Annotated[str, Field(
        description="The subject line of the email. Should be concise and descriptive.")
    ]
    body: Annotated[str, Field(
        description="The main content of the email. Should be well-formatted with proper greeting, paragraphs, and signature.")
    ]
    # receivers: Annotated[List[Receiver], Field(
    #     default_factory=list, description="List of email recipients with their names and email addresses"
    # )] <<<---- nested structure is not supported!!
    receiver_name: Annotated[Optional[str], Field(
        default=None, description="Name of the primary recipient for personalized greeting"
    )]
    receiver_email: Annotated[str, Field(
        description="Email address of the primary recipient. Note, the format must be: <email_id>@<domain>"
    )]

parser = PydanticOutputParser(name="EmailContentParser",
                              pydantic_object=EmailContent)
format_instructions = parser.get_format_instructions()
print(f"Output format instructions: {format_instructions}")

instruction = f"""
        You are an Email Generation Assistant.
        Your task is to generate a professional email based on the user's request.

        GUIDELINES:
        - Create an appropriate subject line (concise and relevant)
        - Write a well-structured email body with:
            * Professional greeting
            * Clear and concise main content
            * Appropriate closing
            * Your name as signature
        - Suggest relevant attachments if applicable (empty list if none needed)
        - Email tone should match the purpose (formal for business, friendly for colleagues)
        - Keep emails concise but complete

        IMPORTANT: Output format:
        <output_format>
        {format_instructions}
        </output_format>

        DO NOT include any explanations or additional text outside the JSON response.
    """

# --- Create Email Generator Agent ---
root_agent = LlmAgent(
    name="email_agent",
    model="gemini-2.0-flash",
    instruction=instruction,
    description="Generates professional emails with structured subject and body",
    output_schema=EmailContent,
    output_key="email",
)
