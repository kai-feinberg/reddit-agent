from __future__ import annotations as _annotations

import praw
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict
import httpx

import logfire
from devtools import debug
from dotenv import load_dotenv

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, ModelRetry, RunContext

load_dotenv()
# llm = os.getenv('LLM_MODEL', 'google/gemini-2.0-flash-exp:free')
llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')


model = OpenAIModel(
    llm,
    # base_url= 'https://openrouter.ai/api/v1',
    # api_key= os.getenv('OPEN_ROUTER_API_KEY')
    api_key= os.getenv('OPENAI_API_KEY')
) 

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
    client: httpx.AsyncClient
    reddit_client_id: str | None
    reddit_client_secret: str | None


ai_agent = Agent(
    model,
    system_prompt=
        '''
        You are a helpful assistant
        ''',
    deps_type=Deps,
    retries=2
)


@ai_agent.tool
async def find_subreddits(ctx: RunContext[Deps], query: str) -> Dict[str, Any]:
    """
    Search Reddit with a given query and return results as a dictionary.

    Args:
        ctx: The context containing dependencies such as Reddit credentials.
        query: The subreddit to search for.

    Returns:
        A dictionary containing subreddits that match the query.
    """
    reddit = praw.Reddit(
        client_id=ctx.deps.reddit_client_id,
        client_secret=ctx.deps.reddit_client_secret,
        user_agent='A search method for Reddit to surface the most relevant posts'
    )

    # Fetch search results
    for subreddit in reddit.subreddits.search(query, limit=5):
        return {"subreddit": subreddit.display_name}

@ai_agent.tool
async def search_reddit(ctx: RunContext[Deps], query: str, subreddit: str = "all") -> Dict[str, Any]:
    """
    Search Reddit with a given query and return results as a dictionary.

    Args:
        ctx: The context containing dependencies such as Reddit credentials.
        query: The search query.
        subreddit: The subreddit to search in (default is "all").

    Returns:
        A dictionary containing the search results with relevant posts and their comments.
    """
    reddit = praw.Reddit(
        client_id=ctx.deps.reddit_client_id,
        client_secret=ctx.deps.reddit_client_secret,
        user_agent='A search method for Reddit to surface the most relevant posts'
    )

    # Fetch search results
    search_results = reddit.subreddit(subreddit).search(query, limit=5)

    # Process results into a JSON-like structure
    result_list = []
    for post in search_results:
        if not post.selftext:  # Skip posts without text
            continue

        # Fetch and sort comments by score
        comments = sorted(
            post.comments.list(),  # Flatten the comment tree
            key=lambda comment: comment.score if isinstance(comment, praw.models.Comment) else 0,
            reverse=True
        )

        # Extract relevant fields for comments, limiting to the first 15 processed comments
        processed_comments = [
            {
                "author": comment.author.name if comment.author else None,
                "score": comment.score,
                "body": comment.body[:1800]  # Limit the comment body length
            }
            for comment in comments if isinstance(comment, praw.models.Comment) and comment.body != "[removed]"
        ][:8]  # Take only the first 8 comments

        # Append post details to the results list
        result_list.append({
            "title": post.title,
            "subreddit": str(post.subreddit),
            "score": post.score,
            "num_comments": post.num_comments,
            "selftext": post.selftext,
            "url": post.url,
            "comments": processed_comments
        })

    # Return as a dictionary
    return {"results": result_list}

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