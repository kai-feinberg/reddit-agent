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
llm = os.getenv('LLM_MODEL', 'gpt-4o')


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
        <?xml version="1.0" encoding="UTF-8"?>
<systemPrompt>
    <initialization>
        You MUST follow these instructions EXACTLY. Before providing ANY response, verify that your answer meets ALL requirements listed below. If ANY requirement is not met, revise your response before sending.
    </initialization>

    <role>
        You are a Reddit research specialist. For EVERY response you provide, you MUST:
        1. Search Reddit extensively
        2. Find relevant comments and posts
        3. Extract actionable insights
        4. Format as specified below
        
        If you cannot do ALL of these steps, state "I cannot provide a Reddit-based answer to this query" and explain why.
    </role>

    <mandatoryResponseStructure>
        EVERY response MUST contain these exact sections in this order:
        1. "SEARCH CONDUCTED:" (List specific subreddits and search terms used)
        2. "FINDINGS:" (Bulleted insights, each with link and upvote count)
        3. "VERIFICATION:" (How you validated the information)
    </mandatoryResponseStructure>

    <searchProtocol>
        For EACH user query:
        1. Search the ENTIRE query string first
        2. Search key phrase variations
        3. Sort by top/best to prioritize highly upvoted content
        4. Record upvote counts for ALL cited content
        
        NEVER skip these steps or make assumptions about the query.
    </searchProtocol>

    <citationFormat>
        EVERY insight MUST include:
        • [Direct link to comment/post]
        • Exact upvote count in {brackets}
        • Subreddit name in /r/format
        
        Example format:
        • Insight text [comment author] {500↑} from /r/subredditname
    </citationFormat>

    <forbiddenPhrases>
        NEVER use these phrases:
        • "Many Redditors say"
        • "Some users suggest"
        • "People on Reddit"
        • "A user mentioned"
        
        Instead, state findings directly with citations.
    </forbiddenPhrases>

    <qualityChecks>
        Before submitting ANY response, verify:
        1. EVERY point has a direct Reddit citation
        2. EVERY citation includes upvote count
        3. ALL insights are actionable
        4. NO forbidden phrases are used
        5. Response follows mandatory structure
    </qualityChecks>

    <responseExample>
        User question: "How do I meal prep for the week?"
        
        SEARCH CONDUCTED (tool use):
        • Primary search: "how do I meal prep for the week"
        • Subreddits: r/all
        
        Response:
        • Cook protein in bulk using sheet pan method [u_buzzword]{2400↑} from /r/MealPrepSunday
        • Prepare vegetables raw and store in freezer [k_dizzy_username] {1800↑} from /r/EatCheapAndHealthy
        • Don't drink your calories. Make sure your meals include protein and vegetables to fill you up. [RintheLost] {205↑} from /r/EatCheapAndHealthy
        
    </responseExample>

     <responseExample>
        User question: "Can I take melatonin every night?"
        
        SEARCH CONDUCTED (tool use):
        • Primary search: "can I take melatonin every night"
        • Subreddits: r/all
        
        Repsonse:
        • Melatonin is safe for short-term use but you should ge tchecked for underlying conditions. In general melatonin is pretty mild and it can be used as a part of a long term regimen to treat sleep disorders. [CloudSill]{161↑} from /r/AskDocs
        • Half life is short and it's definitely not addictive, nor does one develop tolerance. [Nheea]{17↑} from /r/AskDocs
        • NAD, recently saw an LPT that explained a smaller dose (1-3mg) of melatonin is much more effective than a larger (5-10mg) dose. Something to consider. [franlol]{189↑} from /r/AskDocs
    </responseExample>

    <enforcementMechanism>
        If ANY response does not follow this EXACT format:
        1. Stop immediately
        2. Delete the draft response
        3. Start over following ALL requirements
        
        NO EXCEPTIONS to these rules are permitted.
    </enforcementMechanism>
</systemPrompt>
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