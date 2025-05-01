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

TEMPLATE = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts and their valid date ranges. If the fact is about an event, the event takes place during this time.
# format: FACT (Date range: from - to)
<FACTS>
{facts}
</FACTS>

# These are the most relevant entities
# ENTITY_NAME: entity summary
<ENTITIES>
{entities}
</ENTITIES>
"""

def format_edge_date_range(edge: EntityEdge) -> str:
    # return f"{datetime(edge.valid_at).strftime('%Y-%m-%d %H:%M:%S') if edge.valid_at else 'date unknown'} - {(edge.invalid_at.strftime('%Y-%m-%d %H:%M:%S') if edge.invalid_at else 'present')}"
    return f"{edge.valid_at if edge.valid_at else 'date unknown'} - {(edge.invalid_at if edge.invalid_at else 'present')}"


def compose_search_context(edges: list[EntityEdge], nodes: list[EntityNode]) -> str:
    facts = [f'  - {edge.fact} ({format_edge_date_range(edge)})' for edge in edges]
    entities = [f'  - {node.name}: {node.summary}' for node in nodes]
    return TEMPLATE.format(facts='\n'.join(facts), entities='\n'.join(entities))


async def main():
    # Load environment variables
    load_dotenv()

    # Initialize Zep and OpenAI clients
    zep = AsyncZep(api_key=os.getenv("ZEP_API_KEY"), base_url="https://api.getzep.com/api/v2")
    oai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    locomo_df = pd.read_json('data/locomo.json')

    # Get context for each question
    num_users = 10

    zep_search_results = defaultdict(list)
    for user_idx in range(num_users):
        qa_set = locomo_df['qa'].iloc[user_idx]
        user_id = f"locomo_experiment_user_{user_idx}"

        for qa in qa_set:
            start = time()
            query = qa.get('question')

            search_results = await asyncio.gather(
                zep.graph.search(query=query, user_id=user_id, scope='nodes', reranker='rrf', limit=20),
                zep.graph.search(query=query, user_id=user_id, scope='edges', reranker='cross_encoder', limit=20))

            nodes = search_results[0].nodes
            edges = search_results[1].edges

            context = compose_search_context(edges, nodes)
            duration_ms = (time() - start) * 1000

            zep_search_results[user_id].append({'context': context, 'duration_ms': duration_ms})

    os.makedirs("data", exist_ok=True)

    print(zep_search_results)

    with open("data/zep_locomo_search_results.json", "w") as f:
        json.dump(dict(zep_search_results), f, indent=2)
        print('Save search results')





if __name__ == "__main__":
    asyncio.run(main())