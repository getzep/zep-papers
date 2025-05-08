import os
import json
from collections import defaultdict
from time import time

import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from zep_cloud.client import AsyncZep
from zep_cloud import Message, EntityEdge, EntityNode
from openai import AsyncOpenAI
import asyncio

async def locomo_response(llm_client, context: str, question: str) -> str:
    system_prompt = """
        You are a helpful expert assistant answering questions from lme_experiment users based on the provided context.
        """

    prompt = f"""
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories:

    {context}

    Question: {question}
    Answer:
    """


    response = await llm_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}],
                temperature=0,
            )
    result = response.choices[0].message.content or ''

    return result

async def process_qa(qa, search_result, oai_client):
    start = time()
    query = qa.get('question')
    gold_answer = qa.get('answer') or qa.get('adversarial_answer')
    if gold_answer is None:
        return

    zep_answer = await locomo_response(oai_client, search_result.get('context'), query)

    duration_ms = (time() - start) * 1000

    return {'question': query, 'answer': zep_answer, 'golden_answer': gold_answer, 'duration_ms': duration_ms}



async def main():
    # Load environment variables
    load_dotenv()

    # Initialize OpenAI client
    oai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    locomo_df = pd.read_json('data/locomo.json')

    with open('data/zep_locomo_search_results.json', 'r') as file:
        zep_locomo_search_results = json.load(file)

    # Get context for each question
    num_users = 10

    zep_responses = {}
    for group_idx in range(num_users):
        qa_set = locomo_df['qa'].iloc[group_idx]
        group_id = f"locomo_experiment_user_{group_idx}"
        search_results = zep_locomo_search_results.get(group_id)

        tasks = [
            process_qa(qa, search_result, oai_client)
            for qa, search_result in zip(qa_set, search_results, strict=True)
        ]

        responses = await asyncio.gather(*tasks)
        zep_responses[group_id] = responses


    os.makedirs("data", exist_ok=True)

    print(zep_responses)

    with open("data/zep_locomo_responses.json", "w") as f:
        json.dump(zep_responses, f, indent=2)
        print('Save search results')





if __name__ == "__main__":
    asyncio.run(main())