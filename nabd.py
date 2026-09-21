from pathlib import Path
import random
import html
import json
import time

from google import genai
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="NABD",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# GEMINI
# =========================================================

try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )
except Exception:
    client = None


def generate_ai_question(
    skill,
    difficulty,
    language="English"
):

    if client is None:
        return None

    if language == "Arabic":

        prompt = f"""
Create one educational multiple-choice math question.

Skill: {skill}
Difficulty: {difficulty}

The question must be in Arabic.
Provide exactly 4 answer choices.
Provide the correct answer.
Provide a clear step-by-step solution.

Return ONLY valid JSON in this format:
{{
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "answer": "...",
    "solution": "..."
}}
"""

    else:

        prompt = f"""
Create one educational multiple-choice math question.

Skill: {skill}
Difficulty: {difficulty}

The question must be in English.
Provide exactly 4 answer choices.
Provide the correct answer.
Provide a clear step-by-step solution.

Return ONLY valid JSON in this format:
{{
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "answer": "...",
    "solution": "..."
}}
"""

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
