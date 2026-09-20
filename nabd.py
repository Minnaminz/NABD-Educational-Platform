from pathlib import Path
import random
import html
import json

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
import json
from google import genai
try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )
except Exception:
    client = None

        ```python
def generate_ai_question(skill, difficulty, language="English"):
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

    # Try up to 3 times if Gemini is temporarily unavailable
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json"
                }
            )

            return json.loads(response.text)

        except Exception as e:
            error_message = str(e)

            # Retry temporary 503/high-demand errors
            if "503" in error_message or "UNAVAILABLE" in error_message:
                import time
                time.sleep(2)
                continue

            st.error(f"AI Error: {e}")
            return None

    st.error("Gemini is temporarily busy. Please try again in a few seconds.")
    return None
```

# =========================================================
# PATHS + DATA
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "nabd_data.csv"

try:
    df = pd.read_csv(DATA_FILE)

except Exception as e:
    st.error("Unable to load nabd_data.csv")
    st.code(str(e))
    st.stop()


required_columns = [
    "student_id",
    "concept",
    "skill",
    "error_type",
    "correct"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error("Missing columns in nabd_data.csv")
    st.write(missing_columns)
    st.stop()


# =========================================================
# NaN PROTECTION
# =========================================================

df["student_id"] = df["student_id"].fillna(0)

df["concept"] = (
    df["concept"]
    .fillna("Unknown")
    .astype(str)
)

df["skill"] = (
    df["skill"]
    .fillna("Unknown")
    .astype(str)
)

df["error_type"] = (
    df["error_type"]
    .fillna("Unknown")
    .astype(str)
)

df["correct"] = pd.to_numeric(
    df["correct"],
    errors="coerce"
).fillna(0).astype(int)


# =========================================================
# TRANSLATIONS
# =========================================================

T = {

    "English": {

        "home": "Home",
        "assessment": "Assessment",
        "snapshot": "Learning Snapshot",
        "errors": "Error Analysis",
        "practice": "Smart Practice",
        "ai_questions": "AI Question Generator",
        "path": "Learning Path",
        "reassessment": "Reassessment",

        "language": "Language",
        "english": "English",
        "arabic": "العربية",

        "hero_small": "PERSONALIZED LEARNING • AI • DATA",
        "hero_title": "Understand your learning.",
        "hero_title2": "Improve with purpose.",

        "hero_desc":
            "NABD transforms assessment results into meaningful "
            "learning insights and targeted practice.",

        "start": "Start Assessment",
        "continue": "Continue Learning",

        "journey": "Your Learning Journey",

        "assess": "Assess",
        "assess_desc": "Measure your current performance.",

        "analyze": "Analyze",
        "analyze_desc": "Discover possible learning gaps.",

        "practice_title": "Practice",
        "practice_desc": "Focus on skills that need attention.",

        "reassess_title": "Reassess",
        "reassess_desc": "Measure your progress.",

        "questions": "Questions",
        "skills": "Skills",
        "model": "ML Model",
        "decision_tree": "Decision Tree",

        "placement": "Placement Assessment",

        "placement_desc":
            "Answer the questions to estimate your current "
            "learning level.",

        "question": "Question",
        "of": "of",
        "choose": "Choose your answer",

        "previous": "Previous",
        "next": "Next",
        "finish": "Finish Assessment",

        "score": "Score",
        "level": "Estimated Level",

        "advanced": "Advanced",
        "proficient": "Proficient",
        "developing": "Developing",
        "beginner": "Beginner",

        "snapshot_title": "Learning Snapshot",
        "overall": "Overall Accuracy",
        "assessed_skills": "Skills Assessed",
        "skill_performance": "Skill Performance",

        "practice_needed":
            "Skills That May Need Practice",

        "no_practice":
            "No skill fell below the practice threshold.",

        "recommended": "Practice recommended",

        "error_title": "Error Analysis",
        "wrong": "incorrect answer(s)",
        "your_answer": "Your answer",
        "correct_answer": "Correct answer",
        "potential": "Potential error pattern",
        "solution": "Step-by-step solution",

        "perfect":
            "Excellent! No incorrect answers.",

        "smart_title": "Smart Practice",
        "choose_skill": "Choose a skill to practice",
        "generate": "Generate Practice Set",
        "learning_point": "Learning Point",
        "finish_practice": "Finish Practice",
        "practice_complete": "Practice complete!",

        "more_practice":
            "More practice may be helpful before reassessment.",

        "ready_reassess":
            "This skill appears ready for reassessment.",

        "ai_title": "AI Question Generator",

        "ai_desc":
            "Generate personalized practice questions with AI.",

        "ai_skill": "Choose a skill",
        "ai_difficulty": "Choose difficulty",
        "ai_count": "Number of questions",
        "ai_generate": "Generate AI Questions",
        "ai_generated": "AI-generated questions",

        "ai_error":
            "Unable to generate questions right now.",

        "path_title": "Your Learning Path",

        "path_desc":
            "NABD connects assessment, analysis, practice, "
            "and reassessment.",

        "step1": "Assess your current level",
        "step2": "Analyze your skill performance",
        "step3": "Practice targeted skills",
        "step4": "Reassess your progress",

        "reassessment_title": "Reassessment",

        "reassessment_desc":
            "Take a new set of questions to compare your performance.",

        "before": "Before",
        "after": "After",
        "change": "Change",
        "points": "pts",

        "completed":
            "Reassessment completed.",

        "footer":
            "NABD • Personalized Learning Platform • "
            "Student Project Prototype",

        "data_error":
            "There is a problem with the dataset.",

        "complete_first":
            "Complete the assessment first."
    },


    "العربية": {

        "home": "الرئيسية",
        "assessment": "التقييم",
        "snapshot": "ملخص التعلم",
        "errors": "تحليل الأخطاء",
        "practice": "التدريب الذكي",
        "ai_questions":
            "مولّد الأسئلة بالذكاء الاصطناعي",
        "path": "مسار التعلم",
        "reassessment": "إعادة التقييم",

        "language": "اللغة",
        "english": "English",
        "arabic": "العربية",

        "hero_small":
            "تعلم شخصي • ذكاء اصطناعي • تحليل بيانات",

        "hero_title":
            "افهم طريقة تعلّمك.",

        "hero_title2":
            "وتحسّن بذكاء.",

        "hero_desc":
            "يحوّل NABD نتائج التقييم إلى معلومات مفيدة "
            "وتدريبات مخصصة تساعد الطالب على التركيز "
            "على المهارات التي تحتاج إلى ممارسة.",

        "start": "ابدأ التقييم",
        "continue": "تابع التعلم",

        "journey": "رحلة التعلم",

        "assess": "قيّم",
        "assess_desc": "قِس مستواك الحالي.",

        "analyze": "حلّل",
        "analyze_desc":
            "اكتشف المهارات التي قد تحتاج إلى دعم.",

        "practice_title": "تدرّب",
        "practice_desc":
            "ركّز على المهارات التي تحتاج إلى ممارسة.",

        "reassess_title": "أعد التقييم",
        "reassess_desc":
            "قِس تطورك بعد التدريب.",

        "questions": "الأسئلة",
        "skills": "المهارات",
        "model": "نموذج التعلم الآلي",
        "decision_tree": "شجرة القرار",

        "placement": "التقييم التشخيصي",

        "placement_desc":
            "أجب عن الأسئلة لتحديد مستواك الحالي بشكل تقريبي.",

        "question": "السؤال",
        "of": "من",
        "choose": "اختر إجابتك",

        "previous": "السابق",
        "next": "التالي",
        "finish": "إنهاء التقييم",

        "score": "النتيجة",
        "level": "المستوى التقديري",

        "advanced": "متقدم",
        "proficient": "متقن",
        "developing": "في طور التطور",
        "beginner": "مبتدئ",

        "snapshot_title": "ملخص التعلم",
        "overall": "الدقة الإجمالية",
        "assessed_skills": "المهارات التي تم تقييمها",
        "skill_performance": "أداء المهارات",

        "practice_needed":
            "المهارات التي قد تحتاج إلى ممارسة",

        "no_practice":
            "لم تنخفض أي مهارة عن حد الممارسة المحدد.",

        "recommended": "ممارسة مقترحة",

        "error_title": "تحليل الأخطاء",
        "wrong": "إجابة غير صحيحة",
        "your_answer": "إجابتك",
        "correct_answer": "الإجابة الصحيحة",
        "potential": "نمط الخطأ المحتمل",
        "solution": "الحل خطوة بخطوة",

        "perfect":
            "ممتاز! لا توجد إجابات غير صحيحة.",

        "smart_title": "التدريب الذكي",
        "choose_skill": "اختر مهارة للتدريب",
        "generate": "إنشاء مجموعة تدريب",
        "learning_point": "نقطة التعلم",
        "finish_practice": "إنهاء التدريب",
        "practice_complete": "اكتمل التدريب!",

        "more_practice":
            "قد تحتاج إلى المزيد من التدريب قبل إعادة التقييم.",

        "ready_reassess":
            "يبدو أن هذه المهارة جاهزة لإعادة التقييم.",

        "ai_title":
            "مولّد الأسئلة بالذكاء الاصطناعي",

        "ai_desc":
            "أنشئ أسئلة تدريبية مخصصة باستخدام الذكاء الاصطناعي.",

        "ai_skill": "اختر المهارة",
        "ai_difficulty": "اختر مستوى الصعوبة",
        "ai_count": "عدد الأسئلة",

        "ai_generate":
            "إنشاء أسئلة بالذكاء الاصطناعي",

        "ai_generated":
            "الأسئلة التي أنشأها الذكاء الاصطناعي",

        "ai_error":
            "تعذر إنشاء الأسئلة حاليًا.",

        "path_title": "مسار التعلم",

        "path_desc":
            "يربط NABD بين التقييم والتحليل والتدريب "
            "وإعادة التقييم.",

        "step1": "قيّم مستواك الحالي",
        "step2": "حلّل أداء مهاراتك",
        "step3": "تدرّب على المهارات المستهدفة",
        "step4": "أعد تقييم تقدمك",

        "reassessment_title": "إعادة التقييم",

        "reassessment_desc":
            "أجب عن مجموعة جديدة من الأسئلة لمقارنة أدائك.",

        "before": "قبل",
        "after": "بعد",
        "change": "التغير",
        "points": "نقطة",

        "completed":
            "اكتملت إعادة التقييم.",

        "footer":
            "NABD • منصة تعلم شخصي • نموذج مشروع طلابي",

        "data_error":
            "هناك مشكلة في بيانات المشروع.",

        "complete_first":
            "أكمل التقييم أولاً."
    }
}


# =========================================================
# SESSION STATE
# =========================================================

if "lang" not in st.session_state:
    st.session_state.lang = "English"

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "assessment_questions" not in st.session_state:
    st.session_state.assessment_questions = []

if "assessment_answers" not in st.session_state:
    st.session_state.assessment_answers = {}

if "assessment_index" not in st.session_state:
    st.session_state.assessment_index = 0

if "assessment_submitted" not in st.session_state:
    st.session_state.assessment_submitted = False

if "before_score" not in st.session_state:
    st.session_state.before_score = None

if "after_score" not in st.session_state:
    st.session_state.after_score = None

if "practice_questions" not in st.session_state:
    st.session_state.practice_questions = []

if "practice_answers" not in st.session_state:
    st.session_state.practice_answers = {}

if "reassessment_questions" not in st.session_state:
    st.session_state.reassessment_questions = []

if "reassessment_answers" not in st.session_state:
    st.session_state.reassessment_answers = {}

if "reassessment_submitted" not in st.session_state:
    st.session_state.reassessment_submitted = False

if "ai_questions" not in st.session_state:
    st.session_state.ai_questions = []


# =========================================================
# LANGUAGE
# =========================================================

L = T[st.session_state.lang]

is_arabic = st.session_state.lang == "العربية"

direction = "rtl" if is_arabic else "ltr"
text_align = "right" if is_arabic else "left"


# =========================================================
# CSS
# =========================================================

st.markdown(
    f"""
<style>

html, body, [class*="css"] {{
    font-family: "Segoe UI", Arial, sans-serif;
}}

.main {{
    direction: {direction};
}}

.block-container {{
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(
        180deg,
        #101828 0%,
        #18263d 100%
    );
}}

[data-testid="stSidebar"] * {{
    color: white !important;
}}

.hero {{
    background:
        radial-gradient(
            circle at 85% 20%,
            rgba(102,126,234,.35),
            transparent 30%
        ),
        radial-gradient(
            circle at 15% 90%,
            rgba(118,75,162,.30),
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #111827 0%,
            #1e293b 55%,
            #312e81 100%
        );

    border-radius: 32px;
    padding: 55px;
    color: white;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}}

.hero::after {{
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,.12);
    right: 5%;
    top: 15%;
}}

.hero-small {{
    font-size: 13px;
    letter-spacing: 2px;
    opacity: .75;
    font-weight: 700;
}}

.hero-title {{
    font-size: 54px;
    line-height: 1.05;
    font-weight: 900;
    margin-top: 15px;
}}

.hero-title span {{
    background: linear-gradient(
        90deg,
        #93c5fd,
        #c4b5fd
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.hero-desc {{
    font-size: 18px;
    line-height: 1.7;
    opacity: .85;
    max-width: 700px;
    margin-top: 18px;
}}

.section-title {{
    font-size: 28px;
    font-weight: 850;
    margin-top: 30px;
    margin-bottom: 18px;
}}

.stat-card {{
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 22px;
    padding: 25px;
    min-height: 135px;
    box-shadow: 0 8px 25px rgba(15,23,42,.06);
}}

.stat-label {{
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
}}

.stat-value {{
    font-size: 32px;
    font-weight: 900;
    color: #111827;
    margin-top: 8px;
}}

.journey-card {{
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 24px;
    padding: 25px;
    min-height: 185px;
    box-shadow: 0 8px 25px rgba(15,23,42,.05);
}}

.journey-number {{
    font-size: 13px;
    font-weight: 800;
    color: #6366f1;
}}

.journey-title {{
    font-size: 21px;
    font-weight: 800;
    margin-top: 10px;
}}

.journey-desc {{
    color: #64748b;
    line-height: 1.6;
    margin-top: 8px;
}}

.question-card {{
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 26px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(15,23,42,.06);
}}

.question-card h2,
.question-card h3,
.question-card h4,
.question-card p {{
    color: #111827 !important;
}}

.skill-card {{
    background: linear-gradient(
        135deg,
        #f8fafc,
        #eef2ff
    );

    border: 1px solid #e0e7ff;
    border-radius: 22px;
    padding: 22px;
    margin-bottom: 15px;
    color: #111827 !important;
}}

.skill-card span,
.skill-card div {{
    color: #111827 !important;
}}

.big-score {{
    font-size: 58px;
    font-weight: 900;
    color: #4f46e5;
}}

.path-card {{
    background: white;
    border-radius: 24px;
    border: 1px solid #e5e7eb;
    padding: 28px;
    box-shadow: 0 8px 25px rgba(15,23,42,.05);
}}

.footer {{
    text-align: center;
    color: #94a3b8;
    padding: 30px;
}}

div.stButton > button {{
    border-radius: 14px;
    min-height: 48px;
    font-weight: 700;
}}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# MODEL
# =========================================================

model = None

try:

    model_data = df[
        ["concept", "skill"]
    ].copy()

    X = pd.get_dummies(
        model_data,
        dtype=int
    )

    y = df["error_type"]

    if len(y.unique()) >= 2:

        model = DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )

        model.fit(X, y)

except Exception:
    model = None


def predict_error(concept, skill):

    if model is None:

        return (
            "نمط الخطأ غير متاح"
            if is_arabic
            else "Potential error pattern unavailable"
        )

    row = pd.DataFrame([
        {
            "concept": str(concept),
            "skill": str(skill)
        }
    ])

    combined = pd.concat(
        [
            df[["concept", "skill"]],
            row
        ],
        ignore_index=True
    )

    encoded = pd.get_dummies(
        combined,
        dtype=int
    )

    encoded = encoded.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(
        encoded.tail(1)
    )[0]

    return str(prediction)


# =========================================================
# QUESTIONS
# =========================================================

QUESTIONS = [

    {
        "id": 1,
        "concept": "Algebra",
        "skill": "Linear Equations",
        "difficulty": "Easy",
        "question_en": "Solve: x + 5 = 12",
        "question_ar": "أوجد قيمة س: س + ٥ = ١٢",
        "options": ["5", "6", "7", "8"],
        "answer": "7",
        "solution_en": "x + 5 = 12\nx = 12 - 5\nx = 7",
        "solution_ar": "س + ٥ = ١٢\nس = ١٢ - ٥\nس = ٧"
    },

    {
        "id": 2,
        "concept": "Algebra",
        "skill": "Linear Equations",
        "difficulty": "Easy",
        "question_en": "Solve: x - 4 = 9",
        "question_ar": "أوجد قيمة س: س - ٤ = ٩",
        "options": ["11", "12", "13", "14"],
        "answer": "13",
        "solution_en": "x - 4 = 9\nx = 9 + 4\nx = 13",
        "solution_ar": "س - ٤ = ٩\nس = ٩ + ٤\nس = ١٣"
    },

    {
        "id": 3,
        "concept": "Algebra",
        "skill": "Linear Equations",
        "difficulty": "Easy",
        "question_en": "Solve: 2x = 10",
        "question_ar": "أوجد قيمة س: ٢س = ١٠",
        "options": ["2", "5", "8", "10"],
        "answer": "5",
        "solution_en": "2x = 10\nx = 10 ÷ 2\nx = 5",
        "solution_ar": "٢س = ١٠\nس = ١٠ ÷ ٢\nس = ٥"
    },

    {
        "id": 4,
        "concept": "Algebra",
        "skill": "Linear Equations",
        "difficulty": "Easy",
        "question_en": "Solve: x + 8 = 15",
        "question_ar": "أوجد قيمة س: س + ٨ = ١٥",
        "options": ["5", "6", "7", "8"],
        "answer": "7",
        "solution_en": "x + 8 = 15\nx = 15 - 8\nx = 7",
        "solution_ar": "س + ٨ = ١٥\nس = ١٥ - ٨\nس = ٧"
    },

    {
        "id": 5,
        "concept": "Algebra",
        "skill": "Linear Equations",
        "difficulty": "Easy",
        "question_en": "Solve: x - 7 = 3",
        "question_ar": "أوجد قيمة س: س - ٧ = ٣",
        "options": ["4", "10", "11", "12"],
        "answer": "10",
        "solution_en": "x - 7 = 3\nx = 3 + 7\nx = 10",
        "solution_ar": "س - ٧ = ٣\nس = ٣ + ٧\nس = ١٠"
    },

    {
        "id": 6,
        "concept": "Algebra",
        "skill": "Linear Equations",
        "difficulty": "Medium",
        "question_en": "Solve: 3x + 2 = 14",
        "question_ar": "أوجد قيمة س: ٣س + ٢ = ١٤",
        "options": ["3", "4", "5", "6"],
        "answer": "4",
        "solution_en": "3x + 2 = 14\n3x = 12\nx = 4",
        "solution_ar": "٣س + ٢ = ١٤\n٣س = ١٢\nس = ٤"
    },

    {
        "id": 7,
        "concept": "Algebra",
        "skill": "Linear Equations",
        "difficulty": "Medium",
        "question_en": "Solve: 5x - 5 = 20",
        "question_ar": "أوجد قيمة س: ٥س - ٥ = ٢٠",
        "options": ["4", "5", "6", "7"],
        "answer": "5",
        "solution_en": "5x - 5 = 20\n5x = 25\nx = 5",
        "solution_ar": "٥س - ٥ = ٢٠\n٥س = ٢٥\nس = ٥"
    },

    {
        "id": 8,
        "concept": "Algebra",
        "skill": "Linear Equations",
        "difficulty": "Medium",
        "question_en": "Solve: 2x + 6 = 18",
        "question_ar": "أوجد قيمة س: ٢س + ٦ = ١٨",
        "options": ["4", "5", "6", "7"],
        "answer": "6",
        "solution_en": "2x + 6 = 18\n2x = 12\nx = 6",
        "solution_ar": "٢س + ٦ = ١٨\n٢س = ١٢\nس = ٦"
    },

    {
        "id": 9,
        "concept": "Algebra",
        "skill": "Linear Equations",
        "difficulty": "Hard",
        "question_en": "Solve: 4x - 7 = 21",
        "question_ar": "أوجد قيمة س: ٤س - ٧ = ٢١",
        "options": ["6", "7", "8", "9"],
        "answer": "7",
        "solution_en": "4x - 7 = 21\n4x = 28\nx = 7",
        "solution_ar": "٤س - ٧ = ٢١\n٤س = ٢٨\nس = ٧"
    },

    {
        "id": 10,
        "concept": "Algebra",
        "skill": "Linear Equations",
        "difficulty": "Hard",
        "question_en": "Solve: 6x + 4 = 40",
        "question_ar": "أوجد قيمة س: ٦س + ٤ = ٤٠",
        "options": ["5", "6", "7", "8"],
        "answer": "6",
        "solution_en": "6x + 4 = 40\n6x = 36\nx = 6",
        "solution_ar": "٦س + ٤ = ٤٠\n٦س = ٣٦\nس = ٦"
    },

    {
        "id": 11,
        "concept": "Arithmetic",
        "skill": "Multiplication",
        "difficulty": "Easy",
        "question_en": "What is 3 × 4?",
        "question_ar": "ما حاصل ٣ × ٤؟",
        "options": ["7", "10", "12", "14"],
        "answer": "12",
        "solution_en": "3 × 4 = 12",
        "solution_ar": "٣ × ٤ = ١٢"
    },

    {
        "id": 12,
        "concept": "Arithmetic",
        "skill": "Multiplication",
        "difficulty": "Easy",
        "question_en": "What is 5 × 6?",
        "question_ar": "ما حاصل ٥ × ٦؟",
        "options": ["20", "25", "30", "35"],
        "answer": "30",
        "solution_en": "5 × 6 = 30",
        "solution_ar": "٥ × ٦ = ٣٠"
    },

    {
        "id": 13,
        "concept": "Arithmetic",
        "skill": "Multiplication",
        "difficulty": "Easy",
        "question_en": "What is 7 × 2?",
        "question_ar": "ما حاصل ٧ × ٢؟",
        "options": ["12", "14", "16", "18"],
        "answer": "14",
        "solution_en": "7 × 2 = 14",
        "solution_ar": "٧ × ٢ = ١٤"
    },

    {
        "id": 14,
        "concept": "Arithmetic",
        "skill": "Multiplication",
        "difficulty": "Easy",
        "question_en": "What is 9 × 3?",
        "question_ar": "ما حاصل ٩ × ٣؟",
        "options": ["18", "21", "27", "30"],
        "answer": "27",
        "solution_en": "9 × 3 = 27",
        "solution_ar": "٩ × ٣ = ٢٧"
    },

    {
        "id": 15,
        "concept": "Arithmetic",
        "skill": "Multiplication",
        "difficulty": "Medium",
        "question_en": "What is 12 × 8?",
        "question_ar": "ما حاصل ١٢ × ٨؟",
        "options": ["86", "96", "106", "116"],
        "answer": "96",
        "solution_en": "12 × 8 = 96",
        "solution_ar": "١٢ × ٨ = ٩٦"
    },

    {
        "id": 16,
        "concept": "Arithmetic",
        "skill": "Multiplication",
        "difficulty": "Medium",
        "question_en": "What is 14 × 6?",
        "question_ar": "ما حاصل ١٤ × ٦؟",
        "options": ["74", "84", "94", "104"],
        "answer": "84",
        "solution_en": "14 × 6 = 84",
        "solution_ar": "١٤ × ٦ = ٨٤"
    },

    {
        "id": 17,
        "concept": "Arithmetic",
        "skill": "Multiplication",
        "difficulty": "Medium",
        "question_en": "What is 15 × 7?",
        "question_ar": "ما حاصل ١٥ × ٧؟",
        "options": ["95", "105", "115", "125"],
        "answer": "105",
        "solution_en": "15 × 7 = 105",
        "solution_ar": "١٥ × ٧ = ١٠٥"
    },

    {
        "id": 18,
        "concept": "Arithmetic",
        "skill": "Multiplication",
        "difficulty": "Hard",
        "question_en": "What is 24 × 12?",
        "question_ar": "ما حاصل ٢٤ × ١٢؟",
        "options": ["248", "288", "298", "308"],
        "answer": "288",
        "solution_en": "24 × 12 = 288",
        "solution_ar": "٢٤ × ١٢ = ٢٨٨"
    },

    {
        "id": 19,
        "concept": "Arithmetic",
        "skill": "Multiplication",
        "difficulty": "Hard",
        "question_en": "What is 36 × 15?",
        "question_ar": "ما حاصل ٣٦ × ١٥؟",
        "options": ["440", "520", "540", "560"],
        "answer": "540",
        "solution_en": "36 × 15 = 540",
        "solution_ar": "٣٦ × ١٥ = ٥٤٠"
    },

    {
        "id": 20,
        "concept": "Arithmetic",
        "skill": "Multiplication",
        "difficulty": "Hard",
        "question_en": "What is 48 × 11?",
        "question_ar": "ما حاصل ٤٨ × ١١؟",
        "options": ["518", "528", "538", "548"],
        "answer": "528",
        "solution_en": "48 × 11 = 528",
        "solution_ar": "٤٨ × ١١ = ٥٢٨"
    },

    {
        "id": 21,
        "concept": "Arithmetic",
        "skill": "Percentages",
        "difficulty": "Easy",
        "question_en": "What is 10% of 50?",
        "question_ar": "ما قيمة ١٠٪ من ٥٠؟",
        "options": ["2", "5", "10", "15"],
        "answer": "5",
        "solution_en": "10% of 50 = 0.10 × 50 = 5",
        "solution_ar": "١٠٪ من ٥٠ = ٠٫١ × ٥٠ = ٥"
    },

    {
        "id": 22,
        "concept": "Arithmetic",
        "skill": "Percentages",
        "difficulty": "Easy",
        "question_en": "What is 50% of 20?",
        "question_ar": "ما قيمة ٥٠٪ من ٢٠؟",
        "options": ["5", "10", "15", "20"],
        "answer": "10",
        "solution_en": "50% of 20 = 0.50 × 20 = 10",
        "solution_ar": "٥٠٪ من ٢٠ = ٠٫٥ × ٢٠ = ١٠"
    },

    {
        "id": 23,
        "concept": "Arithmetic",
        "skill": "Percentages",
        "difficulty": "Easy",
        "question_en": "What is 25% of 40?",
        "question_ar": "ما قيمة ٢٥٪ من ٤٠؟",
        "options": ["5", "10", "15", "20"],
        "answer": "10",
        "solution_en": "25% of 40 = 0.25 × 40 = 10",
        "solution_ar": "٢٥٪ من ٤٠ = ٠٫٢٥ × ٤٠ = ١٠"
    },

    {
        "id": 24,
        "concept": "Arithmetic",
        "skill": "Percentages",
        "difficulty": "Medium",
        "question_en": "What is 20% of 150?",
        "question_ar": "ما قيمة ٢٠٪ من ١٥٠؟",
        "options": ["20", "25", "30", "35"],
        "answer": "30",
        "solution_en": "20% of 150 = 0.20 × 150 = 30",
        "solution_ar": "٢٠٪ من ١٥٠ = ٠٫٢ × ١٥٠ = ٣٠"
    },

    {
        "id": 25,
        "concept": "Arithmetic",
        "skill": "Percentages",
        "difficulty": "Medium",
        "question_en": "What is 15% of 200?",
        "question_ar": "ما قيمة ١٥٪ من ٢٠٠؟",
        "options": ["20", "30", "40", "50"],
        "answer": "30",
        "solution_en": "15% of 200 = 0.15 × 200 = 30",
        "solution_ar": "١٥٪ من ٢٠٠ = ٠٫١٥ × ٢٠٠ = ٣٠"
    },

    {
        "id": 26,
        "concept": "Arithmetic",
        "skill": "Percentages",
        "difficulty": "Medium",
        "question_en": "What is 30% of 90?",
        "question_ar": "ما قيمة ٣٠٪ من ٩٠؟",
        "options": ["17", "27", "37", "47"],
        "answer": "27",
        "solution_en": "30% of 90 = 0.30 × 90 = 27",
        "solution_ar": "٣٠٪ من ٩٠ = ٠٫٣ × ٩٠ = ٢٧"
    },

    {
        "id": 27,
        "concept": "Arithmetic",
        "skill": "Percentages",
        "difficulty": "Hard",
        "question_en":
            "An $80 item is discounted by 25%. What is the discount?",
        "question_ar":
            "سعر منتج ٨٠ ريالًا، وتم تخفيضه بنسبة ٢٥٪. كم قيمة الخصم؟",
        "options": ["10", "15", "20", "25"],
        "answer": "20",
        "solution_en":
            "25% of 80 = 0.25 × 80 = 20",
        "solution_ar":
            "٢٥٪ من ٨٠ = ٠٫٢٥ × ٨٠ = ٢٠"
    },

    {
        "id": 28,
        "concept": "Arithmetic",
        "skill": "Percentages",
        "difficulty": "Hard",
        "question_en":
            "A number increases from 100 to 120. What is the percentage increase?",
        "question_ar":
            "زاد عدد من ١٠٠ إلى ١٢٠. ما نسبة الزيادة؟",
        "options": ["10%", "15%", "20%", "25%"],
        "answer": "20%",
        "solution_en":
            "Increase = 120 - 100 = 20\n"
            "20 ÷ 100 × 100 = 20%",
        "solution_ar":
            "الزيادة = ١٢٠ - ١٠٠ = ٢٠\n"
            "٢٠ ÷ ١٠٠ × ١٠٠ = ٢٠٪"
    },

    {
        "id": 29,
        "concept": "Arithmetic",
        "skill": "Percentages",
        "difficulty": "Hard",
        "question_en": "What is 35% of 240?",
        "question_ar": "ما قيمة ٣٥٪ من ٢٤٠؟",
        "options": ["74", "84", "94", "104"],
        "answer": "84",
        "solution_en":
            "35% of 240 = 0.35 × 240 = 84",
        "solution_ar":
            "٣٥٪ من ٢٤٠ = ٠٫٣٥ × ٢٤٠ = ٨٤"
    },

    {
        "id": 30,
        "concept": "Arithmetic",
        "skill": "Percentages",
        "difficulty": "Hard",
        "question_en":
            "A $200 item is reduced by 15%. What is the new price?",
        "question_ar":
            "سعر منتج ٢٠٠ ريال وتم تخفيضه بنسبة ١٥٪. ما السعر الجديد؟",
        "options": ["160", "170", "180", "185"],
        "answer": "170",
        "solution_en":
            "15% of 200 = 30\n200 - 30 = 170",
        "solution_ar":
            "١٥٪ من ٢٠٠ = ٣٠\n٢٠٠ - ٣٠ = ١٧٠"
    }
]


# =========================================================
# HELPERS
# =========================================================

def get_question_text(q):

    return (
        q["question_ar"]
        if is_arabic
        else q["question_en"]
    )


def get_solution(q):

    return (
        q["solution_ar"]
        if is_arabic
        else q["solution_en"]
    )


def get_level(score):

    if score >= 90:
        return L["advanced"]

    if score >= 75:
        return L["proficient"]

    if score >= 50:
        return L["developing"]

    return L["beginner"]


def start_assessment():

    st.session_state.assessment_questions = random.sample(
        QUESTIONS,
        15
    )

    st.session_state.assessment_answers = {}

    st.session_state.assessment_index = 0

    st.session_state.assessment_submitted = False

    st.session_state.before_score = None


def calculate_assessment_score():

    questions = st.session_state.assessment_questions

    correct = 0

    for q in questions:

        if (
            st.session_state.assessment_answers.get(
                q["id"]
            )
            == q["answer"]
        ):
            correct += 1

    if not questions:
        return 0

    return round(
        correct / len(questions) * 100
    )


# =========================================================
# AI QUESTION GENERATOR
# =========================================================

def generate_ai_questions(
    skill,
    difficulty,
    count,
    language
):

    if client is None:
        raise RuntimeError(
            "Gemini API is not configured."
        )

    prompt = f"""
You are an educational question generator
for a school learning platform called NABD.

Generate exactly {count} multiple-choice mathematics
questions.

Skill:
{skill}

Difficulty:
{difficulty}

Language:
{language}

Requirements:

1. Create exactly {count} questions.
2. Every question must focus on the selected skill.
3. Every question must have exactly 4 answer choices.
4. There must be exactly one correct answer.
5. The correct answer must appear exactly in the options.
6. Include a short step-by-step solution.
7. Do not repeat questions.
8. Make the difficulty match the requested level.
9. Do not include unsafe, inappropriate, or non-educational content.
10. Return ONLY valid JSON.

Return an array where every item has:

question
options
answer
solution
"""

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string"
                },
                "options": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "answer": {
                    "type": "string"
                },
                "solution": {
                    "type": "string"
                }
            },
            "required": [
                "question",
                "options",
                "answer",
                "solution"
            ]
        }
    }

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": schema
        }
    )

    parsed = getattr(
        response,
        "parsed",
        None
    )

    if parsed is None:

        parsed = json.loads(
            response.text
        )

    valid_questions = []

    for item in parsed:

        if not isinstance(item, dict):
            continue

        question = str(
            item.get("question", "")
        ).strip()

        options = item.get(
            "options",
            []
        )

        answer = str(
            item.get("answer", "")
        ).strip()

        solution = str(
            item.get("solution", "")
        ).strip()

        if (
            question
            and isinstance(options, list)
            and len(options) == 4
            and answer
            and solution
            and answer in [
                str(option)
                for option in options
            ]
        ):

            valid_questions.append(
                {
                    "question": question,
                    "options": [
                        str(option)
                        for option in options
                    ],
                    "answer": answer,
                    "solution": solution
                }
            )

    return valid_questions


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
        font-size:34px;
        font-weight:900;
        margin-bottom:5px;
        ">
        🧠 NABD
        </div>

        <div style="
        opacity:.65;
        font-size:12px;
        letter-spacing:1px;
        ">
        PERSONALIZED LEARNING
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    language_choice = st.radio(
        L["language"],
        ["English", "العربية"],
        index=(
            0
            if st.session_state.lang == "English"
            else 1
        )
    )

    if language_choice != st.session_state.lang:

        st.session_state.lang = language_choice

        st.rerun()

    st.divider()

    pages = [

        ("🏠", L["home"], "Home"),

        ("📝", L["assessment"], "Assessment"),

        ("📊", L["snapshot"], "Learning Snapshot"),

        ("🔎", L["errors"], "Error Analysis"),

        ("🎯", L["practice"], "Smart Practice"),

        ("🤖", L["ai_questions"], "AI Questions"),

        ("🛤️", L["path"], "Learning Path"),

        ("🔄", L["reassessment"], "Reassessment")
    ]

    for icon, label, page_key in pages:

        if st.button(
            f"{icon}  {label}",
            use_container_width=True
        ):

            st.session_state.page = page_key

            st.rerun()


# =========================================================
# HOME
# =========================================================

if st.session_state.page == "Home":

    st.markdown(
        f"""
        <div class="hero">

        <div class="hero-small">
        {L["hero_small"]}
        </div>

        <div class="hero-title">
        🧠 NABD<br>
        <span>{L["hero_title2"]}</span>
        </div>

        <div class="hero-desc">
        {L["hero_desc"]}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="stat-card">

            <div class="stat-label">
            {L["questions"]}
            </div>

            <div class="stat-value">
            30
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="stat-card">

            <div class="stat-label">
            {L["skills"]}
            </div>

            <div class="stat-value">
            3
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="stat-card">

            <div class="stat-label">
            {L["model"]}
            </div>

            <div class="stat-value"
            style="font-size:24px;">

            {L["decision_tree"]}

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="section-title">
        {L["journey"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    cols = st.columns(4)

    journey = [

        (
            "01",
            "📝",
            L["assess"],
            L["assess_desc"]
        ),

        (
            "02",
            "🔎",
            L["analyze"],
            L["analyze_desc"]
        ),

        (
            "03",
            "🎯",
            L["practice_title"],
            L["practice_desc"]
        ),

        (
            "04",
            "📈",
            L["reassess_title"],
            L["reassess_desc"]
        )
    ]

    for col, item in zip(
        cols,
        journey
    ):

        number, icon, title, desc = item

        with col:

            st.markdown(
                f"""
                <div class="journey-card">

                <div class="journey-number">
                {number}
                </div>

                <div style="
                font-size:30px;
                margin-top:8px;
                ">
                {icon}
                </div>

                <div class="journey-title">
                {title}
                </div>

                <div class="journey-desc">
                {desc}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    if st.button(
        f"🚀  {L['start']}",
        use_container_width=True
    ):

        start_assessment()

        st.session_state.page = "Assessment"

        st.rerun()


# =========================================================
# ASSESSMENT
# =========================================================

elif st.session_state.page == "Assessment":

    st.title(
        f"📝 {L['placement']}"
    )

    st.write(
        L["placement_desc"]
    )

    if not st.session_state.assessment_questions:

        start_assessment()

    if not st.session_state.assessment_submitted:

        questions = (
            st.session_state.assessment_questions
        )

        index = (
            st.session_state.assessment_index
        )

        current = questions[index]

        total = len(questions)

        progress = (
            (index + 1) / total
        )

        st.progress(progress)

        st.caption(
            f"{L['question']} "
            f"{index + 1} "
            f"{L['of']} "
            f"{total}"
        )

        st.markdown(
            f"""
            <div class="question-card">

            <div style="
            color:#6366f1;
            font-weight:800;
            font-size:13px;
            ">

            {html.escape(str(current["skill"]))}
            •
            {html.escape(str(current["difficulty"]))}

            </div>

            <h2>
            {html.escape(get_question_text(current))}
            </h2>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        previous_answer = (
            st.session_state.assessment_answers.get(
                current["id"]
            )
        )

        answer = st.radio(
            L["choose"],
            current["options"],
            index=(
                current["options"].index(
                    previous_answer
                )
                if previous_answer
                in current["options"]
                else None
            ),
            key=f"question_{current['id']}"
        )

        st.session_state.assessment_answers[
            current["id"]
        ] = answer

        col1, col2 = st.columns(2)

        with col1:

            if index > 0:

                if st.button(
                    f"← {L['previous']}",
                    use_container_width=True
                ):

                    st.session_state.assessment_index -= 1

                    st.rerun()

        with col2:

            if index < total - 1:

                if st.button(
                    f"{L['next']} →",
                    use_container_width=True
                ):

                    st.session_state.assessment_index += 1

                    st.rerun()

            else:

                if st.button(
                    f"✓ {L['finish']}",
                    use_container_width=True
                ):

                    score = (
                        calculate_assessment_score()
                    )

                    st.session_state.before_score = score

                    st.session_state.assessment_submitted = True

                    st.rerun()

    else:

        score = (
            st.session_state.before_score
        )

        level = get_level(score)

        st.markdown(
            f"""
            <div class="hero">

            <div class="hero-small">
            NABD ASSESSMENT RESULT
            </div>

            <div class="big-score"
            style="color:white;">
            {score}%
            </div>

            <div style="
            font-size:24px;
            font-weight:800;
            margin-top:5px;
            ">

            {L["level"]}: {level}

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            f"📊 {L['snapshot']}",
            use_container_width=True
        ):

            st.session_state.page = (
                "Learning Snapshot"
            )

            st.rerun()


# =========================================================
# SNAPSHOT
# =========================================================

elif st.session_state.page == "Learning Snapshot":

    st.title(
        f"📊 {L['snapshot_title']}"
    )

    if st.session_state.before_score is None:

        st.warning(
            L["complete_first"]
        )

        st.stop()

    score = (
        st.session_state.before_score
    )

    questions = (
        st.session_state.assessment_questions
    )

    answers = (
        st.session_state.assessment_answers
    )

    skill_scores = {}

    for skill in sorted(
        set(
            q["skill"]
            for q in questions
        )
    ):

        skill_questions = [
            q
            for q in questions
            if q["skill"] == skill
        ]

        correct = sum(
            1
            for q in skill_questions
            if answers.get(q["id"])
            == q["answer"]
        )

        skill_scores[skill] = round(
            correct /
            len(skill_questions)
            * 100
        )

    weak_skills = [
        skill
        for skill, value
        in skill_scores.items()
        if value < 70
    ]

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
            <div class="stat-card">

            <div class="stat-label">
            {L["overall"]}
            </div>

            <div class="stat-value">
            {score}%
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="stat-card">

            <div class="stat-label">
            {L["level"]}
            </div>

            <div class="stat-value"
            style="font-size:25px;">

            {get_level(score)}

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="stat-card">

            <div class="stat-label">
            {L["assessed_skills"]}
            </div>

            <div class="stat-value">
            {len(skill_scores)}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="section-title">
        {L["skill_performance"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    for skill, value in skill_scores.items():

        st.markdown(
            f"""
            <div class="skill-card">

            <div style="
            display:flex;
            justify-content:space-between;
            font-weight:800;
            color:#111827 !important;
            ">

            <span>
            {html.escape(str(skill))}
            </span>

            <span>
            {value}%
            </span>

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            value / 100
        )

    st.markdown(
        f"""
        <div class="section-title">
        {L["practice_needed"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    if weak_skills:

        for skill in weak_skills:

            st.warning(
                f"🎯 {L['recommended']}: {skill}"
            )

    else:

        st.success(
            f"✓ {L['no_practice']}"
        )

    st.markdown(
        f"""
        <div class="section-title">
        📈 {L["journey"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            L["before"],
            f"{score}%"
        )

    with c2:

        st.metric(
            L["practice_title"],
            "→"
        )

    with c3:

        if (
            st.session_state.after_score
            is None
        ):

            st.metric(
                L["after"],
                "—"
            )

        else:

            st.metric(
                L["after"],
                f"{st.session_state.after_score}%"
            )


# =========================================================
# ERROR ANALYSIS
# =========================================================

elif st.session_state.page == "Error Analysis":

    st.title(
        f"🔎 {L['error_title']}"
    )

    if not st.session_state.assessment_submitted:

        st.warning(
            L["complete_first"]
        )

        st.stop()

    questions = (
        st.session_state.assessment_questions
    )

    answers = (
        st.session_state.assessment_answers
    )

    wrong_questions = [
        q
        for q in questions
        if answers.get(q["id"])
        != q["answer"]
    ]

    if not wrong_questions:

        st.success(
            f"✓ {L['perfect']}"
        )

    else:

        st.write(
            f"{len(wrong_questions)} "
            f"{L['wrong']}"
        )

        for q in wrong_questions:

            selected = answers.get(
                q["id"],
                "—"
            )

            st.markdown(
                f"""
                <div class="question-card">

                <h3>
                {html.escape(get_question_text(q))}
                </h3>

                <p>
                <b>{L["your_answer"]}:</b>
                {html.escape(str(selected))}
                </p>

                <p>
                <b>{L["correct_answer"]}:</b>
                {html.escape(str(q["answer"]))}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            predicted = predict_error(
                q["concept"],
                q["skill"]
            )

            st.info(
                f"🔎 {L['potential']}: {predicted}"
            )

            with st.expander(
                f"💡 {L['solution']}"
            ):

                st.code(
                    get_solution(q),
                    language="text"
                )


# =========================================================
# SMART PRACTICE
# =========================================================

elif st.session_state.page == "Smart Practice":

    st.title(
        f"🎯 {L['smart_title']}"
    )

    skills = sorted(
        set(
            q["skill"]
            for q in QUESTIONS
        )
    )

    selected_skill = st.selectbox(
        L["choose_skill"],
        skills
    )

    if st.button(
        f"✨ {L['generate']}",
        use_container_width=True
    ):

        pool = [
            q
            for q in QUESTIONS
            if q["skill"] == selected_skill
        ]

        st.session_state.practice_questions = (
            random.sample(
                pool,
                min(6, len(pool))
            )
        )

        st.session_state.practice_answers = {}

        st.rerun()

    if st.session_state.practice_questions:

        questions = (
            st.session_state.practice_questions
        )

        for i, q in enumerate(questions):

            st.markdown(
                f"""
                <div class="question-card">

                <div style="
                color:#6366f1;
                font-weight:800;
                ">
                {L["question"]} {i + 1}
                </div>

                <h3>
                {html.escape(get_question_text(q))}
                </h3>

                </div>
                """,
                unsafe_allow_html=True
            )

            answer = st.radio(
                L["choose"],
                q["options"],
                index=None,
                key=f"practice_answer_{q['id']}"
            )

            st.session_state.practice_answers[
                q["id"]
            ] = answer

            with st.expander(
                f"💡 {L['learning_point']}"
            ):

                st.code(
                    get_solution(q),
                    language="text"
                )

        if st.button(
            f"✓ {L['finish_practice']}",
            use_container_width=True
        ):

            correct = sum(
                1
                for q in questions
                if st.session_state.practice_answers.get(
                    q["id"]
                ) == q["answer"]
            )

            score = round(
                correct /
                len(questions)
                * 100
            )

            st.success(
                f"{L['practice_complete']} "
                f"{score}%"
            )

            if score >= 80:

                st.info(
                    f"🚀 {L['ready_reassess']}"
                )

            else:

                st.warning(
                    f"🎯 {L['more_practice']}"
                )


# =========================================================
# AI QUESTION GENERATOR
# =========================================================

elif st.session_state.page == "AI Questions":

    st.title(
        f"🤖 {L['ai_title']}"
    )

    st.write(
        L["ai_desc"]
    )

    skills = sorted(
        set(
            q["skill"]
            for q in QUESTIONS
        )
    )

    difficulties = [
        "Easy",
        "Medium",
        "Hard"
    ]

    col1, col2, col3 = st.columns(3)

    with col1:

        selected_skill = st.selectbox(
            L["ai_skill"],
            skills
        )

    with col2:

        selected_difficulty = st.selectbox(
            L["ai_difficulty"],
            difficulties
        )

    with col3:

        question_count = st.selectbox(
            L["ai_count"],
            [3, 5, 6, 8, 10],
            index=1
        )

    if st.button(
        f"✨ {L['ai_generate']}",
        use_container_width=True
    ):

        language = (
            "Arabic"
            if is_arabic
            else "English"
        )

        with st.spinner(
            "Generating questions..."
            if not is_arabic
            else "جاري إنشاء الأسئلة..."
        ):

            try:

                generated_questions = (
                    generate_ai_questions(
                        selected_skill,
                        selected_difficulty,
                        question_count,
                        language
                    )
                )

                if not generated_questions:

                    st.error(
                        L["ai_error"]
                    )

                else:

                    st.session_state.ai_questions = (
                        generated_questions
                    )

                    st.rerun()

            except Exception as e:

                st.error(
                    L["ai_error"]
                )

                st.code(
                    str(e)
                )

    if st.session_state.ai_questions:

        st.markdown(
            f"""
            <div class="section-title">
            🤖 {L["ai_generated"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        for i, q in enumerate(
            st.session_state.ai_questions
        ):

            question_text = html.escape(
                str(q["question"])
            )

            st.markdown(
                f"""
                <div class="question-card">

                <div style="
                color:#6366f1;
                font-weight:800;
                ">
                {L["question"]} {i + 1}
                </div>

                <h3>
                {question_text}
                </h3>

                </div>
                """,
                unsafe_allow_html=True
            )

            answer = st.radio(
                L["choose"],
                q["options"],
                index=None,
                key=f"ai_question_{i}"
            )

            if answer:

                if answer == q["answer"]:

                    st.success(
                        "✓ Correct!"
                        if not is_arabic
                        else "✓ إجابة صحيحة!"
                    )

                else:

                    st.error(
                        "✗ Incorrect"
                        if not is_arabic
                        else "✗ إجابة غير صحيحة"
                    )

                    st.write(
                        f"**{L['correct_answer']}:** "
                        f"{q['answer']}"
                    )

                with st.expander(
                    f"💡 {L['solution']}"
                ):

                    st.code(
                        q["solution"],
                        language="text"
                    )


# =========================================================
# LEARNING PATH
# =========================================================

elif st.session_state.page == "Learning Path":

    st.title(
        f"🛤️ {L['path_title']}"
    )

    st.write(
        L["path_desc"]
    )

    path_items = [

        (
            "01",
            "📝",
            L["assess"],
            L["step1"]
        ),

        (
            "02",
            "🔎",
            L["analyze"],
            L["step2"]
        ),

        (
            "03",
            "🎯",
            L["practice_title"],
            L["step3"]
        ),

        (
            "04",
            "📈",
            L["reassess_title"],
            L["step4"]
        )
    ]

    for number, icon, title, desc in path_items:

        st.markdown(
            f"""
            <div class="path-card">

            <div style="
            display:flex;
            align-items:center;
            gap:20px;
            ">

            <div style="
            font-size:32px;
            font-weight:900;
            color:#6366f1;
            ">
            {number}
            </div>

            <div>

            <div style="
            font-size:25px;
            font-weight:850;
            ">
            {icon} {title}
            </div>

            <div style="
            color:#64748b;
            margin-top:5px;
            ">
            {desc}
            </div>

            </div>

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")


# =========================================================
# REASSESSMENT
# =========================================================

elif st.session_state.page == "Reassessment":

    st.title(
        f"🔄 {L['reassessment_title']}"
    )

    if st.session_state.before_score is None:

        st.warning(
            L["complete_first"]
        )

        st.stop()

    st.write(
        L["reassessment_desc"]
    )

    if not st.session_state.reassessment_questions:

        st.session_state.reassessment_questions = (
            random.sample(
                QUESTIONS,
                10
            )
        )

        st.session_state.reassessment_answers = {}

        st.session_state.reassessment_submitted = False

    if not st.session_state.get(
        "reassessment_submitted",
        False
    ):

        questions = (
            st.session_state.reassessment_questions
        )

        for i, q in enumerate(questions):

            st.markdown(
                f"""
                <div class="question-card">

                <div style="
                color:#6366f1;
                font-weight:800;
                ">
                {L["question"]} {i + 1}
                </div>

                <h3>
                {html.escape(get_question_text(q))}
                </h3>

                </div>
                """,
                unsafe_allow_html=True
            )

            answer = st.radio(
                L["choose"],
                q["options"],
                index=None,
                key=f"reassessment_{q['id']}"
            )

            st.session_state.reassessment_answers[
                q["id"]
            ] = answer

        if st.button(
            f"✓ {L['finish']}",
            use_container_width=True
        ):

            correct = sum(
                1
                for q in questions
                if st.session_state.reassessment_answers.get(
                    q["id"]
                ) == q["answer"]
            )

            score = round(
                correct /
                len(questions)
                * 100
            )

            st.session_state.after_score = score

            st.session_state.reassessment_submitted = True

            st.rerun()

    else:

        before = (
            st.session_state.before_score
        )

        after = (
            st.session_state.after_score
        )

        change = after - before

        st.markdown(
            f"""
            <div class="hero">

            <div class="hero-small">
            {L["reassessment_title"].upper()}
            </div>

            <div style="
            font-size:54px;
            font-weight:900;
            margin-top:10px;
            ">
            {after}%
            </div>

            <div style="
            font-size:20px;
            opacity:.8;
            ">
            {L["after"]}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                L["before"],
                f"{before}%"
            )

        with c2:

            st.metric(
                L["after"],
                f"{after}%"
            )

        with c3:

            st.metric(
                L["change"],
                f"{change:+d} {L['points']}"
            )

        st.success(
            f"✓ {L['completed']}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    f"""
    <div class="footer">
        {L["footer"]}<br>
        © 2026 MINNA MOHAMMED — NABD Educational Platform. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
