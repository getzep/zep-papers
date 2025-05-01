import os
import json
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from zep_cloud.client import AsyncZep
from zep_cloud import Message
from openai import AsyncOpenAI
import asyncio


async def main():
    # Load environment variables
    load_dotenv()

    # Initialize Zep and OpenAI clients
    zep = AsyncZep(api_key=os.getenv("ZEP_API_KEY"), base_url="https://api.getzep.com/api/v2")
    oai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if __name__ == "__main__":
    asyncio.run(main())