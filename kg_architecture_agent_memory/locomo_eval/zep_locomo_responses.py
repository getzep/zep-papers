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
    # CONTEXT:
    You have access to facts and entities from a conversation.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. Always convert relative time references to specific dates, months, or years.
    6. Be as specific as possible when talking about people, places, and events
    7. Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.
    
    Clarification:
    When interpreting memories, use the timestamp to determine when the described event happened, not when someone talked about the event.
    
    Example:
    
    Memory: (2023-03-15T16:33:00Z) I went to the vet yesterday.
    Question: What day did I go to the vet?
    Correct Answer: March 15, 2023
    Explanation:
    Even though the phrase says "yesterday," the timestamp shows the event was recorded as happening on March 15th. Therefore, the actual vet visit happened on that date, regardless of the word "yesterday" in the text.


    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Context:

    {context}

    Question: {question}
    Answer:
    """

    response = await llm_client.chat.completions.create(
                model='gpt-4.1-mini',
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