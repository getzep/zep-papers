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

    # Download JSON data
    url = "https://raw.githubusercontent.com/snap-research/locomo/refs/heads/main/data/locomo10.json"
    response = requests.get(url)
    data = response.json()
    locomo_df = pd.read_json(url)

    # Save JSON locally
    os.makedirs("../data", exist_ok=True)
    with open("../data/locomo.json", "w") as f:
        json.dump(data, f, indent=2)

    print("JSON saved to ./data/locomo.json")

    # Initialize Zep client
    zep = AsyncZep(api_key=os.getenv("ZEP_API_KEY"), base_url="https://api.getzep.com/api/v2")

    # Process each user/session
    num_users = 10
    max_session_count = 35

    for user_idx in range(num_users):
        conversation = locomo_df['conversation'].iloc[user_idx]
        user_id = f"locomo_experiment_user_{user_idx}"
        session_id = f"locomo_experiment_session_{user_idx}"
        print(user_id)

        first_name = conversation['speaker_a']

        try:
            await zep.user.add(user_id=user_id, first_name=first_name)
            await zep.memory.add_session(user_id=user_id, session_id=session_id)
        except Exception:
            pass

        for session_idx in range(max_session_count):
            session_key = f'session_{session_idx}'
            print(session_key)
            session = conversation.get(session_key)
            if session is None:
                continue

            for msg in session:
                session_date = conversation.get(f'session_{session_idx}_date_time') + ' UTC'
                date_format = '%I:%M %p on %d %B, %Y UTC'
                date_string = datetime.strptime(session_date, date_format).replace(tzinfo=timezone.utc)
                iso_date = date_string.isoformat()

                speaker = msg.get('speaker')
                role_type = 'user' if speaker == first_name else 'norole'

                await zep.memory.add(
                    session_id=session_id,
                    messages=[
                        Message(
                            role=speaker,
                            role_type=role_type,
                            content=msg.get('text'),
                            created_at=iso_date
                        )
                    ]
                )

if __name__ == "__main__":
    asyncio.run(main())
