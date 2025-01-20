from __future__ import annotations as _annotations

import asyncpraw
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict
import httpx
import logging
import traceback
import logfire
from devtools import debug
from dotenv import load_dotenv

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, ModelRetry, RunContext

load_dotenv()
# llm = os.getenv('LLM_MODEL', 'anthropic/claude-3.5-haiku')
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
    brave_api_key: str | None


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
        You are a Reddit research specialist. ONLY use reddit for your information. Do not make use of any data you have been trained on. Your job is simply to intelligently summarize and extract insights from the results of your tools.
        For EVERY response you provide, you MUST:

        1. Search Reddit extensively
        2. Find relevant comments and posts
        3. Extract actionable insights
        4. Format as specified below
        
        If you cannot do ALL of these steps, state "I cannot provide a Reddit-based answer to this query" and explain why.
    </role>

    <mandatoryResponseStructure>
        EVERY response MUST contain these exact sections in this order in markdown format:
        1. relevant quote snippet from comment or post
        2. User name of comment author and link back to post [brackets](link to post)
        3. Upvote count in { brackets }

        Each comment should be its own bullet.
    </mandatoryResponseStructure>

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
    retries=0 ## TODO CHANGE THIS TO 2
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ai_agent.tool
async def search_reddit(ctx: RunContext[Deps], query: str) -> Dict[str, Any]:
    """
    Search Reddit with a given query and return results as a dictionary.

    Args:
        ctx: The context containing dependencies such as Reddit credentials.
        query: The search query.
    Returns:
        A dictionary containing the search results with relevant posts and their comments.
    """
    logger.info(f"Starting Reddit search with query: {query}")
    
    # Check Brave API key
    if ctx.deps.brave_api_key is None:
        logger.warning("No Brave API key provided - returning test results")
        return {"results": [], "error": "Please provide a Brave API key to get real search results"}

    headers = {
        'X-Subscription-Token': ctx.deps.brave_api_key,
        'Accept': 'application/json',
    }
    
    try:
        # Search using Brave API
        logger.info("Calling Brave search API")
        # Modified this part to use the client directly without context manager
        r = await ctx.deps.client.get(
            'https://api.search.brave.com/res/v1/web/search',
            params={
                'q': query + "reddit",  # Better filtering for Reddit-specific results
                'count': 3,
                'text_decorations': True,
                'search_lang': 'en'
            },
            headers=headers
        )
        data = r.json()
        logger.info(f"Received {len(data.get('web', {}).get('results', []))} results from Brave")
        
        # Extract Reddit URLs
        reddit_urls = [
            item.get('url') 
            for item in data.get('web', {}).get('results', [])
            if item.get('url', '').startswith('https://www.reddit.com/r/')
        ]
        
        if not reddit_urls:
            logger.warning("No valid Reddit URLs found in search results")
            return {"results": [], "error": "No relevant Reddit posts found"}

        # Initialize Reddit client
        logger.info("Initializing Reddit client")
        reddit = asyncpraw.Reddit(
            client_id=ctx.deps.reddit_client_id,
            client_secret=ctx.deps.reddit_client_secret,
            user_agent='A search method for Reddit to surface the most relevant posts'
        )

        result_data = []
        for url in reddit_urls:
            try:
                logger.info(f"Processing Reddit post: {url}")
                submission = await reddit.submission(url=url)
                        
                # Sort and process comments
                comments = sorted(
                            submission.comments.list(),  # Flatten the comment tree
                            key=lambda comment: comment.score if isinstance(comment, asyncpraw.models.Comment) else 0,
                            reverse=True
                )  # sort comments to get the top upvoted comments            
                processed_comments = []
                
                for comment in comments[:8]:  # Limit to top 8 comments
                    if not isinstance(comment, asyncpraw.models.Comment) or comment.body == "[removed]" or comment.body == "[deleted]":
                        continue
                        
                    try:
                        author_name = comment.author.name if comment.author else "[deleted]"
                        processed_comments.append({
                            "author": author_name,
                            "score": comment.score,
                            "body": comment.body[:1800]  # Limit comment length
                        })
                    except AttributeError as e:
                        logger.error(f"Error processing comment: {e}")
                        continue

                # Add post data
                result_data.append({
                    "title": submission.title,
                    "subreddit": str(submission.subreddit),
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "selftext": submission.selftext[:2000],  # Limit selftext length
                    "url": submission.url,
                    "comments": processed_comments
                })
                
            except Exception as e:
                logger.error(f"Error processing submission {url}: {e}")
                continue

        logger.info(f"Successfully processed {len(result_data)} Reddit posts")
        await reddit.close()
        return {"results": result_data}

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

        return {"results": [], "error": f"Unexpected error: {str(e)}"}