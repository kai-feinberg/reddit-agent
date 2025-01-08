# Pydantic AI agents with Streamlit UI:

## Description
This is a template repository for creating basic chat applications with pydantic ai agents. This is a great frontend and starting point that allows you to focus on building useful agents rather than detailed frontends.

## Prerequisites
Python 3.11+
OpenAI API key (if using GPT models)
Ollama (optional, for local LLM usage)
Brave Search API key

## How to run

Set up environment variables:
Rename .env.example to .env.
Edit .env with your API keys and preferences:
OPENAI_API_KEY=your_openai_api_key  # Only needed if using GPT models
BRAVE_API_KEY=your_brave_api_key # only if using web search capabilities

### Method 1: Docker (Reccomended)

Run `docker compose up --build` to build and run the docker container

Once built you can run `docker compose up` in the future

Access your application at [http://localhost:8501/](http://localhost:8501/)

### Method 2: Local

`pip install -r requirements.txt`

`streamlit run streamlit_ui.py`

Access your application at [http://localhost:8501/](http://localhost:8501/)

## Repository structure

### `Streamlit_ui.py`
The Streamlit app is created to provide a UI with text streaming from the LLM. You shouldn't need to change anything to have a perfectly functioning chat app. The app will respond with a generated message and will use tools you provide as it deems necessary (you can edit these tools in `ai_agent.py`) 


### `Web_search_agent`
This is where you will write your agent with pydantic ai. By default the model will respond to messages with chat gpt-4o. There are two example tools, one to search the web with brave's api and one to fetch the transcript of a youtube video.

You can define your system prompt during agent creation

    
        ai_agent = Agent(
            model,
            system_prompt="INSERT HERE",
            ...
    

Define your tools with the @ai_agent.tool decorator. You can use any python libraries you like (but make sure to update requirements.txt). If you would like to integrate logfire logging from pydantic ai you can create and use a LOGFIRE_TOKEN from [https://logfire.pydantic.dev/docs/#logfire](https://logfire.pydantic.dev/docs/#logfire).

For each tool make sure to use the provided docstring to tell the ai agent what the tool is and what data is required/what data is returned.

For example (the first line is the description of the agent tool)

    """
        Get the transcript of a YouTube video. Use this to take generate notes in markdown format based on the content of a youtube video.

        Args:
            ctx: The context.
            video_url: The URL of the YouTube video.

        Returns:
            str: The transcript of the video.
    """

## Next steps

    - [ ] Integrate authorization via supabase
    - [ ] Store/load chat history so you can load previous chats
    - [ ] Allow for users to input their own api keys