
# from dotenv import load_dotenv
# from httpx import AsyncClient
# import streamlit as st
# import asyncio
# import json
# import os
# from openai import AsyncOpenAI, OpenAI
# from pydantic_ai.messages import ModelResponse, ModelRequest, UserPromptPart
# from pydantic_ai.models.openai import OpenAIModel, OpenAIStreamTextResponse

# # imports the search agent and its dependencies
# from ai_agent import ai_agent, Deps

# load_dotenv()

# async def prompt_ai(prompt, messages):
#     async with AsyncClient() as client:
#         reddit_client_id = os.getenv('REDDIT_CLIENT_ID', None)
#         reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET', None)

#         deps = Deps(client=client, reddit_client_id=reddit_client_id, reddit_client_secret=reddit_client_secret)
        
#         if messages != []:
#             async with ai_agent.run_stream(
#                 prompt, deps=deps, message_history=messages.all_messages()
#             ) as result:
            
#                 # Stream the text response
#                 async for message in result.stream_text(delta=True):
#                     st.write(type(message))
#                     yield message
                
#                 st.write(ai_agent.all_messages())
#         else:
#             async with ai_agent.run_stream(
#                 prompt, deps=deps
#             ) as result:
#                 # Stream the text response
#                 async for message in result.stream_text(delta=True):
#                     yield OpenAIStreamTextResponse(message)
                
#                 st.write(ai_agent.all_messages())

            
# async def main():
#     st.title("AI Chatbot with agents")

#     # React to user input
#     if prompt := st.chat_input("What would you like to find today?"):
#         # Display user message in chat message container
#         st.chat_message("user").markdown(prompt)
#         # Add user message to chat history
#         # st.session_state.messages.append(ModelRequest(content=prompt))

#         # Display assistant response in chat message container
#         response_content = ""
#         messages= []
         
#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()  # Placeholder for updating the message
#             # Run the async generator to fetch responses
#             async for chunk in prompt_ai(prompt, messages):
#                 st.write(chunk)
#                 response_content += (chunk)
#                 # Update the placeholder with the current response content
#                 message_placeholder.markdown(response_content)
      
#         # st.session_state.messages.append(ModelResponse(content=response_content))
        
# if __name__ == "__main__":
#     asyncio.run(main())

from dotenv import load_dotenv
from httpx import AsyncClient
import streamlit as st
import asyncio
import json
import os
from openai import AsyncOpenAI, OpenAI
from pydantic_ai.messages import ModelResponse, ModelRequest, UserPromptPart
from pydantic_ai.models.openai import OpenAIModel

# imports the search agent and its dependencies
from ai_agent import ai_agent, Deps

load_dotenv()

async def prompt_ai(messages):
    async with AsyncClient() as client:
        reddit_client_id = os.getenv('REDDIT_CLIENT_ID', None)
        reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET', None)

        deps = Deps(client=client, reddit_client_id=reddit_client_id, reddit_client_secret=reddit_client_secret)

        async with ai_agent.run_stream(
            messages[-1].content, deps=deps, message_history=messages[:-1]
        ) as result:
            if "tool_usage" not in st.session_state:
                st.session_state.tool_usage = []

            # Stream the text response
            async for message in result.stream_text(delta=True):
                yield message
            
            # Process tool usage
            tool_calls = []

            st.write(ai_agent.all_messages())

            for msg in result._all_messages:
                if msg.role == 'model-structured-response' and hasattr(msg, 'calls'):
                    for call in msg.calls:
                        tool_info = {
                            'id': call.tool_id,
                            'tool': call.tool_name,
                            'arguments': call.args.args_json,
                            'response': None
                        }
                        tool_calls.append(tool_info)
                elif msg.role == 'tool-return':
                    for tool in tool_calls:
                        if tool['id'] == msg.tool_id:
                            tool['response'] = msg.content
                            break
            
            if tool_calls:
                st.session_state.tool_usage.append({
                    'timestamp': str(result.timestamp()),
                    'tool_calls': tool_calls
                })
                
                yield "\n\n---\n**Tools Used in this Response:**\n"
                for idx, tool in enumerate(tool_calls, 1):
                    with st.expander(f"{idx}. {tool['tool']}", expanded=False):
                        st.markdown("**Arguments:**")
                        st.code(tool['arguments'], language='json')
                        st.markdown("**Tool Output:**")
                        # st.markdown(
                        #     f"""<div style="height: 200px; overflow-y: auto; border: 1px solid #ccc; padding: 10px;">
                        #     <pre>{tool['response']}</pre>
                        #     </div>""", 
                        #     unsafe_allow_html=True
                        # )
                        with stylable_container(
                            "codeblock",
                            """
                            code {
                                white-space: pre-wrap !important;
                            }
                            """,
                        ):
                            st.code(tool['response'], language='json')

async def main():
    st.title("AI Chatbot with agents")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []    

    # Display tool usage history in sidebar
    if "tool_usage" in st.session_state and st.session_state.tool_usage:
        st.sidebar.header("Tool Usage History")
        for idx, usage in enumerate(st.session_state.tool_usage):
            with st.sidebar.expander(f"Interaction {idx + 1}"):
                st.json(usage)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        # role = message.role
        if isinstance(message, ModelResponse):
            with st.chat_message("ai"):
                st.markdown(message.content)
        elif isinstance(message, UserPromptPart):
            with st.chat_message("human"):
                st.markdown(message.content)
        
        # with st.chat_message("human" if isinstance(message, UserPromptPart) else "ai"):
        #     st.markdown(message.content)

    # React to user input
    if prompt := st.chat_input("What would you like to find today?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(ModelRequest(content=prompt))

        # Display assistant response in chat message container
        response_content = ""
         
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # Placeholder for updating the message
            # Run the async generator to fetch responses
            async for chunk in prompt_ai(st.session_state.messages):
                response_content += chunk
                # Update the placeholder with the current response content
                message_placeholder.markdown(response_content)
      
        st.session_state.messages.append(ModelResponse(content=response_content))
        
if __name__ == "__main__":
    asyncio.run(main())