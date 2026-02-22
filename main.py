#!/usr/bin/env python3
"""
********************************************************
    Project: Frames AI
    Version: 2.0
    Author: CBOMBS
    Date:   November 8th, 2025
    Script: main.py
    Description: An AI animation assitant that helps build storyboards 
                 and generate motion poses from user commands

    Example Usage:
        source .venv/bin/activate
        deactivate
        python3 main.py

        pixelate man.pn
        pixelate 64x64 man.png
        walk man.png
        run
        sit 
        jump
        turn

    Notes:
        memory.json
*********************************************************
"""
import os, json, re, sys, signal
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from app.pose.pose_engine import *

# --- Setup ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ Missing OPEN_API_KEY in .env file!")

model = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=api_key
)

memory_file = "memory.json"
pose_engine = PoseEngine()

def save_memory():
    with open(memory_file, "w") as f:
        json.dump([{"role": m.type, "content": m.content} for m in memory], f, indent=2)

# --- Load memory --- 
if os.path.exists(memory_file):
    with open(memory_file, "r") as f:
        raw = json.load(f)
        memory = [SystemMessage(m["content"]) if m["role"] == "system"
                  else HumanMessage(m["content"]) if m["role"] == "user"
                  else AIMessage(m["content"]) for m in raw]
else:
    memory = [SystemMessage(
        "You are Frames AI — a creative assistant that helps build "
        "storyboards and generate motion poses from user commands."
    )]

print("\n\n🎞️ Frames AI ready!")

# --- Simple greeting/chat handler ---
def handle_small_talk(user_input):
    greetings = ["hello", "hi", "hey", "how's it going", "good morning", "good evening"]
    if any(g.lower() in user_input.lower() for g in greetings):
        return "🤖 Frames: Hey there! How can I help you today?"
    return None

# --- Handle abrupt exits ---
def handle_exit(sig=None, frame=None):
    save_memory()
    print("\n💾 Memory saved. 👋 Goodbye!")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)   # Ctrl+C
signal.signal(signal.SIGTERM, handle_exit)  # kill command

# --- Main chat loop --- 
while True:
    user_input = input("🎙️ You: ")
    if user_input.lower() in ["exit", "quit"]:
        handle_exit()


    # First, check if it's small talk
    chat_response = handle_small_talk(user_input)
    if chat_response:
        print(chat_response)
        memory.append(HumanMessage(user_input))
        memory.append(SystemMessage(chat_response))
        save_memory()
        continue

    # Add user input to memory
    memory.append(HumanMessage(user_input))

    # 🧩 Pose detection
    pose_keyword = pose_engine.detect_pose(user_input)
    if pose_keyword:
        pose_response = pose_engine.handle_pose(pose_keyword, user_input)
        print(pose_response)
        # Add to memory so AI can refer to it contextually
        memory.append(SystemMessage(pose_response))
        save_memory()
        continue

    # Normal AI response
    response = model.invoke(memory)
    ai_message = response.content
    print("🤖 Frames:", ai_message)
    memory.append(AIMessage(ai_message))

    # Auto save memory
    save_memory()
