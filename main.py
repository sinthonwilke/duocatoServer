from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from typing import List
import os

load_dotenv()
app = FastAPI()
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
)

SYSTEM_PROMPT = \
"""
You will be assigned a role by following these instructions:

You are an English teacher who always finds a topic to start conversations with the student in the chat app. The English level skill depends on the mode that the student gave in front of the message: Like this [mode={mode level, which have easy, medium, hard}]" easy mode is like you talking with a child not more than 13 years. mode medium is like you talking to teenagers age between 14 - 21 years and mode hard is like you talking to people that are 21++ years.

- Remember to talk like a human does.
- No numbered points.
- Trying answer short.
- Don't explanation things.
- Always find random topic so student can answering.

Example format: [mode={mode level, which can be easy, medium, or hard}], followed by the message that the student speaks.
"""

class ReqMessage(BaseModel):
    text: str
    isSentByUser: bool
    mode: str

@app.post("/")
async def post(reqMessage: List[ReqMessage]):
    res = await chatGPT(reqMessage)    
    return {"message": res}
  
async def chatGPT(reqMessage: List[ReqMessage]):
    messages = [
        {"role": "user", "content": f"[mode={message.mode}] {message.text}"} if message.isSentByUser else 
        {"role": "system", "content": message.text}
        for message in reqMessage
    ]

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT.strip()},
            *messages
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return res.choices[0].message.content.strip()