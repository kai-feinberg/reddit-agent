from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import logfire
from devtools import debug
from httpx import AsyncClient
from dotenv import load_dotenv

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, ModelRetry, RunContext
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

load_dotenv()
llm = os.getenv('LLM_MODEL', 'gpt-4o')


model = OpenAIModel(llm) 

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
# logfire.configure(send_to_logfire='if-token-present')

# Configure Logfire with more detailed settings
logfire.configure(
    token =os.getenv('LOGFIRE_TOKEN', None),
    send_to_logfire='if-token-present',
    service_name="web-search-agent",
    service_version="1.0.0",
    environment=os.getenv('ENVIRONMENT', 'development'),

)

# Class for dependencies for agent (will be injected from ui)
@dataclass
class Deps:
    client: AsyncClient
    brave_api_key: str | None


ai_agent = Agent(
    model,
    system_prompt=
        '''You are an expert at researching the web to answer user questions. 
        Format your response in markdown and provide citations as well as the sources at the end of the response. 
        You also have the ability to fetch the transcript of Youtube Videos. Use this functionality to generate notes in markdown format based on the content of a youtube video.

        When taking notes use the following guidelines:
            Please summarize the following information as structured notes. Focus on capturing key points, omitting unnecessary details, and using bullet points or short paragraphs for readability. Prioritize clarity and conciseness by highlighting:
                1. **Main topics or sections**
                2. **Key points, insights, or findings**
                3. **Supporting details** (only if essential)
                4. **Action items or next steps** (if applicable)
                5. **Dates, names, or specific terms** (only if relevant)

                Format the notes in bullet points or short, clear sentences. Avoid repetition or filler words. Aim for a summary that is easy to scan and ideal for quick reference.
    
        ''',
    deps_type=Deps,
    retries=2
)


@ai_agent.tool
async def search_web(
    ctx: RunContext[Deps], web_query: str
) -> str:
    """
     
    Search the web given a query defined to answer the user's question.

    Args:
        ctx: The context.
        web_query: The query for the web search.

    Returns:
        str: The search results as a formatted string.
    """
    if ctx.deps.brave_api_key is None:
        return "This is a test web search result. Please provide a Brave API key to get real search results."

    headers = {
        'X-Subscription-Token': ctx.deps.brave_api_key,
        'Accept': 'application/json',
    }
    
    with logfire.span('calling Brave search API', query=web_query) as span:
        r = await ctx.deps.client.get(
            'https://api.search.brave.com/res/v1/web/search',
            params={
                'q': web_query,
                'count': 5,
                'text_decorations': True,
                'search_lang': 'en'
            },
            headers=headers
        )
        r.raise_for_status()
        data = r.json()
        span.set_attribute('response', data)

    results = []
    
    # Add web results in a nice formatted way
    web_results = data.get('web', {}).get('results', [])
    for item in web_results[:3]:
        title = item.get('title', '')
        description = item.get('description', '')
        url = item.get('url', '')
        if title and description:
            results.append(f"Title: {title}\nSummary: {description}\nSource: {url}\n")

    return "\n".join(results) if results else "No results found for the query."

@ai_agent.tool
async def get_youtube_transcript(
    ctx: RunContext[Deps], video_url: str
)-> str:
    """
     
    Get the transcript of a YouTube video. Use this to take generate notes in markdown format based on the content of a youtube video.

    Args:
        ctx: The context.
        video_url: The URL of the YouTube video.

    Returns:
        str: The transcript of the video.
    """
    with logfire.span('getting YouTube transcript', video_url=video_url) as span:
        if not video_url.startswith('https://www.youtube.com/watch?v='):
            return "Invalid YouTube video URL. Please provide a valid YouTube video URL."
        video_id = video_url.split('v=')[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        formatted_transcript = formatter.format_transcript(transcript)
        span.set_attribute('transcript', formatted_transcript)
    
    return formatted_transcript if formatted_transcript else "No transcript found for the video."

async def main():
    # async with AsyncClient() as client:
    #     brave_api_key = os.getenv('BRAVE_API_KEY', None)
    #     deps = Deps(client=client, brave_api_key=brave_api_key)

    #     result = await web_search_agent.run(
    #         'Give me some articles talking about the new release of React 19.', deps=deps
    #     )
        
    #     debug(result)
    #     print('Response:', result.data)
    print("this doesn't matter")

if __name__ == '__main__':
    asyncio.run(main())