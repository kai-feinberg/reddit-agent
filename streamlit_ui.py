from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import httpx
import os

import streamlit as st
import json
import logfire

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from ai_agent import ai_agent, Deps

# Load environment variables if needed
from dotenv import load_dotenv
load_dotenv()

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content) 

    elif part.part_kind == 'tool-call':
        args = json.loads(part.args.args_json)
        st.session_state.tool_calls[part.tool_call_id] = args

    # tool-return
    elif part.part_kind == 'tool-return':
        tool_args = st.session_state.tool_calls.get(part.tool_call_id, {})
        with st.expander(f"Used tool: {part.tool_name}", expanded=False):
            st.markdown("**Tool Call Arguments:**")
            st.code(json.dumps(tool_args, indent=2), language="json")

            st.markdown("**Tool Output:**")                 
            st.code(json.dumps(part.content, indent=2), language="json")


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    reddit_client_id = os.getenv('REDDIT_CLIENT_ID', None)
    reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET', None)

    deps = Deps(
        client = httpx.AsyncClient(), 
        reddit_client_id=reddit_client_id, 
        reddit_client_secret=reddit_client_secret)

    # Run the agent in a stream
    try:
        async with ai_agent.run_stream(
            user_input,
            deps=deps,
            message_history= st.session_state.messages[:-1],  # pass entire conversation so far
        ) as result:
            # We'll gather partial text to show incrementally
            partial_text = ""
            message_placeholder = st.empty()

            # Render partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                partial_text += chunk
                message_placeholder.markdown(partial_text)


            # Add the final response to the messages
            st.session_state.messages.append(
                ModelResponse(parts=[TextPart(content=partial_text)])
            )

            # Now that the stream is finished, we have a final result.
            # Add new messages from this run, excluding user-prompt messages
            # THIS ADDS THE TOOL PARTS AFTER THE RESPONSE SO WE CAN DISPLAY THEM AT THE END OF THE RESPONSE
            filtered_messages = [msg for msg in result.new_messages() 
                            if not (hasattr(msg, 'parts') and 
                                    any(part.part_kind == 'user-prompt' for part in msg.parts))]
            st.session_state.messages.extend(filtered_messages)

            ## now we display tools and tool usage from this response ...
            new_messages = result.new_messages()
            for msg in new_messages:
                if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
                    for part in msg.parts:
                        if part.part_kind == 'tool-call' or part.part_kind == 'tool-return':
                            display_message_part(part)

            
            #display all messages for debugging
            # st.write(st.session_state.messages)
    finally:
        await deps.client.aclose()


async def main():
    st.title("Reddit Search Analyzer")
    st.write("Ask me anything that I can search on Reddit!")


    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "tool_calls" not in st.session_state:
        st.session_state.tool_calls = {}

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What would you like to know about today?")

    if user_input:     

        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input)


if __name__ == "__main__":
    asyncio.run(main())