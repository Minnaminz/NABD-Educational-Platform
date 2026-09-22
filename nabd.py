from pathlib import Path
import random
import html
import json
import time
import textwrap

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


RETRYABLE_AI_ERRORS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "DEADLINE_EXCEEDED",
    "INTERNAL"
)


def is_retryable_ai_error(error):
    message = str(error).upper()
    return any(
        error_code in message
        for error_code in RETRYABLE_AI_ERRORS
    )


def generate_ai_question(
    skill,
    difficulty,
    language="English"
):
    if client is None:
        return None

    prompt = f"""
Create one educational multiple-choice math question.

Skill: {skill}
Difficulty: {difficulty}
Language: {language}

The question must be in {language}.
Provide exactly 4 answer choices.
Provide exactly one correct answer.
Provide a clear step-by-step solution.

IMPORTANT:
- Return plain text only inside the JSON fields.
- Do not use HTML tags.
- Do not use Markdown formatting.
- Do not use code blocks.
- Do not include <div>, <p>, <span>, <br>, or any other HTML.

Return ONLY valid JSON in this format:
{{
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "answer": "...",
    "solution": "..."
}}
"""

    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json"
                }
            )

            return json.loads(response.text)

        except Exception as e:
            if is_retryable_ai_error(e) and attempt < 3:
                time.sleep(2 ** attempt)
                continue

            return None

    return None


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

        "practice_needed": "Skills That May Need Practice",
        "no_practice": "No skill fell below the practice threshold.",
        "recommended": "Practice recommended",

        "error_title": "Error Analysis",
        "wrong": "incorrect answer(s)",
        "your_answer": "Your answer",
        "correct_answer": "Correct answer",
        "potential": "Potential error pattern",
        "solution": "Step-by-step solution",

        "perfect": "Excellent! No incorrect answers.",

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

        "ai_busy":
            "Gemini is temporarily busy. Please wait a few seconds and try again.",

        "ai_setup":
            "Gemini is not configured. Please check your Streamlit secret.",

        "ai_connection":
            "There was a connection problem with the AI service. Please try again.",

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

        "completed": "Reassessment completed.",

        "footer":
            "NABD • Personalized Learning Platform • "
            "Student Project Prototype",

        "data_error": "There is a problem with the dataset.",
        "complete_first": "Complete the assessment first.",

        "theme": "Theme",
        "dark": "Dark",
        "light": "Light",

        "activities": "Learning Activities",
        "activities_title": "NABD Learning Activities",
        "activities_desc":
            "Short challenges that make practice more engaging.",
        "challenge": "NABD Challenge",
        "challenge_desc":
            "Answer 5 questions and test your skills.",
        "start_challenge": "Start Challenge",
        "challenge_complete": "Challenge complete!",
        "challenge_score": "Challenge Score",
        "challenge_again": "Try Again",
        "challenge_locked": "Start the challenge to begin.",

        "ai_status": "AI status",
        "ai_ready": "Gemini is ready",
        "ai_unavailable": "Gemini is not configured",
        "appearance": "Appearance",

        "perfect_title": "Perfect score! 🏆",
        "perfect_desc":
            "Outstanding work! You showed a very strong understanding of the skills tested.",

        "excellent_title": "Outstanding work! 🌟",
        "excellent_desc":
            "You showed a strong understanding of the math skills in this assessment.",

        "good_title": "Great job! 💪",
        "good_desc":
            "You are building a solid understanding. Keep practicing and you can become even stronger.",

        "developing_title": "Nice effort! 🚀",
        "developing_desc":
            "You are making progress. Review the questions you missed and try again.",

        "beginner_title": "Every attempt helps you learn! 🌱",
        "beginner_desc":
            "Use your mistakes as clues, practice the key skills, and give it another try.",

        "focus_area": "Focus area",
        "keep_going": "Keep going!",
        "practice_progress": "Practice is part of the learning process.",

        "improved": "You improved! Keep building on this progress. 🎉",
        "same_score":
            "Your score stayed the same. That is okay — use your mistakes to guide your next practice session. 🌱",
        "review_again":
            "This attempt gives you useful information about what to review next. Keep going! 💪",

        "correct_feedback": "Correct! Great work! ✓",
        "incorrect_feedback":
            "Not quite — use the solution to learn from this question."
    },

    "العربية": {
        "home": "الرئيسية",
        "assessment": "التقييم",
        "snapshot": "ملخص التعلم",
        "errors": "تحليل الأخطاء",
        "practice": "التدريب الذكي",
        "ai_questions": "مولّد الأسئلة بالذكاء الاصطناعي",
        "path": "مسار التعلم",
        "reassessment": "إعادة التقييم",

        "language": "اللغة",
        "english": "English",
        "arabic": "العربية",

        "hero_small": "تعلم شخصي • ذكاء اصطناعي • تحليل بيانات",
        "hero_title": "افهم طريقة تعلّمك.",
        "hero_title2": "وتحسّن بذكاء.",

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
        "analyze_desc": "اكتشف المهارات التي قد تحتاج إلى دعم.",

        "practice_title": "تدرّب",
        "practice_desc":
            "ركّز على المهارات التي تحتاج إلى ممارسة.",

        "reassess_title": "أعد التقييم",
        "reassess_desc": "قِس تطورك بعد التدريب.",

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

        "practice_needed": "المهارات التي قد تحتاج إلى ممارسة",
        "no_practice":
            "لم تنخفض أي مهارة عن حد الممارسة المحدد.",
        "recommended": "ممارسة مقترحة",

        "error_title": "تحليل الأخطاء",
        "wrong": "إجابة غير صحيحة",
        "your_answer": "إجابتك",
        "correct_answer": "الإجابة الصحيحة",
        "potential": "نمط الخطأ المحتمل",
        "solution": "الحل خطوة بخطوة",

        "perfect": "ممتاز! لا توجد إجابات غير صحيحة.",

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

        "ai_title": "مولّد الأسئلة بالذكاء الاصطناعي",
        "ai_desc":
            "أنشئ أسئلة تدريبية مخصصة باستخدام الذكاء الاصطناعي.",
        "ai_skill": "اختر المهارة",
        "ai_difficulty": "اختر مستوى الصعوبة",
        "ai_count": "عدد الأسئلة",
        "ai_generate": "إنشاء أسئلة بالذكاء الاصطناعي",
        "ai_generated": "الأسئلة التي أنشأها الذكاء الاصطناعي",

        "ai_error": "تعذر إنشاء الأسئلة حاليًا.",

        "ai_busy":
            "Gemini مشغول مؤقتًا. انتظر بضع ثوانٍ ثم حاول مرة أخرى.",

        "ai_setup":
            "لم يتم إعداد Gemini. تأكد من إضافة المفتاح في Streamlit Secrets.",

        "ai_connection":
            "حدثت مشكلة في الاتصال بخدمة الذكاء الاصطناعي. حاول مرة أخرى.",

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

        "completed": "اكتملت إعادة التقييم.",

        "footer":
            "NABD • منصة تعلم شخصي • نموذج مشروع طلابي",

        "data_error": "هناك مشكلة في بيانات المشروع.",
        "complete_first": "أكمل التقييم أولاً.",

        "theme": "المظهر",
        "dark": "داكن",
        "light": "فاتح",

        "activities": "الفعاليات التعليمية",
        "activities_title": "فعاليات NABD التعليمية",
        "activities_desc":
            "تحديات قصيرة تجعل التدريب أكثر تفاعلاً ومتعة.",
        "challenge": "تحدي NABD",
        "challenge_desc":
            "أجب عن ٥ أسئلة واختبر مهاراتك.",
        "start_challenge": "ابدأ التحدي",
        "challenge_complete": "اكتمل التحدي!",
        "challenge_score": "نتيجة التحدي",
        "challenge_again": "حاول مرة أخرى",
        "challenge_locked": "ابدأ التحدي للبدء.",

        "ai_status": "حالة الذكاء الاصطناعي",
        "ai_ready": "Gemini جاهز",
        "ai_unavailable": "Gemini غير مهيأ",
        "appearance": "المظهر",

        "perfect_title": "نتيجة كاملة! 🏆",
        "perfect_desc":
            "أداء رائع جدًا! أظهرت فهمًا قويًا جدًا للمهارات التي تم اختبارها.",

        "excellent_title": "أداء رائع! 🌟",
        "excellent_desc":
            "أظهرت فهمًا قويًا لمهارات الرياضيات في هذا التقييم.",

        "good_title": "أحسنت! 💪",
        "good_desc":
            "أنت تبني فهمًا جيدًا. استمر في التدريب وستصبح أقوى.",

        "developing_title": "محاولة جميلة! 🚀",
        "developing_desc":
            "أنت تتقدم. راجع الأسئلة التي أخطأت فيها وحاول مرة أخرى.",

        "beginner_title": "كل محاولة تساعدك على التعلم! 🌱",
        "beginner_desc":
            "استخدم أخطاءك كدليل، وتدرّب على المهارات الأساسية، ثم حاول مرة أخرى.",

        "focus_area": "مجال التركيز",
        "keep_going": "استمر!",
        "practice_progress": "التدريب جزء طبيعي من رحلة التعلم.",

        "improved": "تحسنت نتيجتك! استمر في البناء على هذا التقدم. 🎉",
        "same_score":
            "بقيت نتيجتك كما هي. لا بأس — استخدم أخطاءك لتحديد ما تحتاج إلى التدريب عليه. 🌱",
        "review_again":
            "هذه المحاولة أعطتك معلومات مفيدة عمّا تحتاج إلى مراجعته. استمر! 💪",

        "correct_feedback": "إجابة صحيحة! أحسنت! ✓",
        "incorrect_feedback":
            "ليست الإجابة الصحيحة — استخدم الحل للتعلم من هذا السؤال."
    }
}


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "lang": "English",
    "theme": "Dark",

    "challenge_questions": [],
    "challenge_answers": {},
    "challenge_submitted": False,
    "challenge_score": None,

    "page": "Home",

    "assessment_questions": [],
    "assessment_answers": {},
    "assessment_index": 0,
    "assessment_submitted": False,
    "assessment_celebrated": False,

    "before_score": None,
    "after_score": None,

    "practice_questions": [],
    "practice_answers": {},
    "practice_submitted": False,
    "practice_score": None,

    "reassessment_questions": [],
    "reassessment_answers": {},
    "reassessment_submitted": False,

    "ai_questions": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# LANGUAGE
# =========================================================

L = T[st.session_state.lang]

is_arabic = st.session_state.lang == "العربية"

direction = "rtl" if is_arabic else "ltr"
text_align = "right" if is_arabic else "left"


# =========================================================
# COLORS
# =========================================================

if st.session_state.theme == "Dark":

    COLORS = {
        "page": "#0b1120",
        "surface": "#111827",
        "surface_alt": "#172033",
        "surface_soft": "#1e293b",
        "text": "#f8fafc",
        "muted": "#cbd5e1",
        "border": "#334155",

        "accent": "#818cf8",
        "accent_2": "#c084fc",
        "accent_dark": "#3730a3",

        "hero_1": "#111827",
        "hero_2": "#1e293b",
        "hero_3": "#312e81",

        "button_text": "#ffffff",
        "input_bg": "#0f172a",
        "code_bg": "#020617",

        "sidebar_1": "#0b1020",
        "sidebar_2": "#171b3a",
        "sidebar_text": "#f8fafc",
        "sidebar_muted": "#cbd5e1",

        "shadow": "rgba(0,0,0,.28)"
    }

else:

    COLORS = {
        "page": "#f5f7fb",
        "surface": "#ffffff",
        "surface_alt": "#f8fafc",
        "surface_soft": "#eef2ff",
        "text": "#111827",
        "muted": "#475569",
        "border": "#dbe2ea",

        "accent": "#4f46e5",
        "accent_2": "#9333ea",
        "accent_dark": "#312e81",

        "hero_1": "#172554",
        "hero_2": "#312e81",
        "hero_3": "#581c87",

        "button_text": "#ffffff",
        "input_bg": "#ffffff",
        "code_bg": "#f1f5f9",

        "sidebar_1": "#ffffff",
        "sidebar_2": "#eef2ff",
        "sidebar_text": "#111827",
        "sidebar_muted": "#475569",

        "shadow": "rgba(15,23,42,.10)"
    }


# =========================================================
# CSS
# =========================================================

st.markdown(
    textwrap.dedent(
    f"""
    <style>

    :root {{
        color-scheme: {"dark" if st.session_state.theme == "Dark" else "light"};
    }}

    html, body, [class*="css"] {{
        font-family: "Segoe UI", Arial, sans-serif;
    }}

    body {{
        background: {COLORS["page"]} !important;
    }}

    .stApp {{
        background:
            radial-gradient(
                circle at 90% 5%,
                rgba(99,102,241,.09),
                transparent 28%
            ),
            {COLORS["page"]} !important;
        color: {COLORS["text"]} !important;
    }}

    .main {{
        direction: {direction};
        background: transparent !important;
    }}

    .block-container {{
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }}

    /* =====================================================
       SIDEBAR
       ===================================================== */

    [data-testid="stSidebar"] {{
        background:
            radial-gradient(
                circle at 20% 10%,
                rgba(129,140,248,.20),
                transparent 28%
            ),
            linear-gradient(
                180deg,
                {COLORS["sidebar_1"]} 0%,
                {COLORS["sidebar_2"]} 100%
            ) !important;

        border-right: 1px solid {COLORS["border"]};
        direction: {direction};
    }}

    [data-testid="stSidebar"] * {{
        color: {COLORS["sidebar_text"]} !important;
    }}

    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {{
        color: {COLORS["sidebar_muted"]} !important;
    }}

    [data-testid="stSidebar"] hr {{
        border-color: {COLORS["border"]} !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] {{
        background: transparent !important;
        padding: 0 !important;
    }}

    [data-testid="stSidebar"] .stRadio label {{
        border-radius: 12px;
        padding: 5px 8px;
    }}

    /* =====================================================
       GLOBAL TEXT
       ===================================================== */

    .stMarkdown,
    .stText,
    label,
    p,
    li,
    .stCaption {{
        color: {COLORS["text"]} !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS["text"]} !important;
        font-weight: 850 !important;
    }}

    .section-title {{
        font-size: 28px;
        font-weight: 900;
        color: {COLORS["text"]} !important;
        margin-top: 30px;
        margin-bottom: 18px;
    }}

    /* =====================================================
       HERO
       ===================================================== */

    .hero {{
        background:
            radial-gradient(
                circle at 85% 20%,
                rgba(129,140,248,.35),
                transparent 30%
            ),
            radial-gradient(
                circle at 15% 90%,
                rgba(192,132,252,.28),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                {COLORS["hero_1"]} 0%,
                {COLORS["hero_2"]} 55%,
                {COLORS["hero_3"]} 100%
            );

        border-radius: 30px;
        padding: 52px;
        color: #ffffff !important;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 55px {COLORS["shadow"]};
    }}

    .hero::after {{
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,.18);
        right: -55px;
        top: -55px;
    }}

    .hero,
    .hero * {{
        color: #ffffff !important;
    }}

    .hero-small {{
        font-size: 13px;
        letter-spacing: 2px;
        opacity: .82;
        font-weight: 800;
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
            #e9d5ff
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .hero-desc {{
        font-size: 18px;
        line-height: 1.7;
        max-width: 700px;
        margin-top: 18px;
        color: #ffffff !important;
    }}

    /* =====================================================
       CARDS
       ===================================================== */

    .stat-card,
    .journey-card,
    .question-card,
    .path-card,
    .ai-question-card {{
        background: {COLORS["surface"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        color: {COLORS["text"]} !important;
        box-shadow: 0 12px 30px {COLORS["shadow"]};
    }}

    .stat-card {{
        border-radius: 22px;
        padding: 25px;
        min-height: 135px;
    }}

    .stat-card * {{
        color: {COLORS["text"]} !important;
    }}

    .stat-label {{
        color: {COLORS["muted"]} !important;
        font-size: 14px;
        font-weight: 700;
    }}

    .stat-value {{
        font-size: 32px;
        font-weight: 900;
        color: {COLORS["text"]} !important;
        margin-top: 8px;
    }}

    .journey-card {{
        border-radius: 24px;
        padding: 25px;
        min-height: 185px;
        transition: transform .18s ease, box-shadow .18s ease;
    }}

    .journey-card:hover,
    .stat-card:hover,
    .path-card:hover,
    .ai-question-card:hover,
    .activity-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 18px 38px {COLORS["shadow"]};
    }}

    .journey-card * {{
        color: {COLORS["text"]} !important;
    }}

    .journey-number {{
        font-size: 13px;
        font-weight: 800;
        color: {COLORS["accent"]} !important;
    }}

    .journey-title {{
        font-size: 21px;
        font-weight: 850;
        margin-top: 10px;
    }}

    .journey-desc {{
        color: {COLORS["muted"]} !important;
        line-height: 1.6;
        margin-top: 8px;
    }}

    .question-card {{
        border-radius: 22px;
        padding: 25px;
        margin-top: 15px;
    }}

    .question-card h2,
    .question-card h3,
    .question-card p,
    .question-card span {{
        color: {COLORS["text"]} !important;
    }}

    .skill-card {{
        background:
            linear-gradient(
                135deg,
                {COLORS["surface_alt"]},
                {COLORS["surface_soft"]}
            ) !important;

        border: 1px solid {COLORS["border"]} !important;
        border-radius: 22px;
        padding: 22px;
        margin-bottom: 15px;
        color: {COLORS["text"]} !important;
    }}

    .skill-card * {{
        color: {COLORS["text"]} !important;
    }}

    .path-card {{
        border-radius: 24px;
        padding: 28px;
    }}

    .path-card * {{
        color: {COLORS["text"]} !important;
    }}

    .big-score {{
        font-size: 58px;
        font-weight: 900;
        color: {COLORS["accent"]} !important;
    }}

    /* =====================================================
       ENCOURAGEMENT
       ===================================================== */

    .encouragement-card {{
        background:
            linear-gradient(
                135deg,
                {COLORS["surface"]},
                {COLORS["surface_soft"]}
            ) !important;

        border: 1px solid {COLORS["border"]} !important;
        border-left: 5px solid {COLORS["accent"]} !important;
        border-radius: 22px;
        padding: 24px 26px;
        margin: 20px 0;
        box-shadow: 0 12px 30px {COLORS["shadow"]};
    }}

    .encouragement-card * {{
        color: {COLORS["text"]} !important;
    }}

    .encouragement-title {{
        font-size: 23px;
        font-weight: 900;
        margin-bottom: 7px;
    }}

    .encouragement-text {{
        color: {COLORS["muted"]} !important;
        font-size: 16px;
        line-height: 1.65;
    }}

    .focus-card {{
        background: {COLORS["surface_alt"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 18px;
        padding: 17px 20px;
        margin-top: 12px;
    }}

    .focus-card * {{
        color: {COLORS["text"]} !important;
    }}

    /* =====================================================
       AI
       ===================================================== */

    .ai-header {{
        background:
            radial-gradient(
                circle at 90% 15%,
                rgba(192,132,252,.25),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #1e3a8a,
                #312e81
            );

        border-radius: 24px;
        padding: 28px 30px;
        margin-bottom: 22px;
        box-shadow: 0 16px 36px {COLORS["shadow"]};
    }}

    .ai-header,
    .ai-header * {{
        color: #ffffff !important;
    }}

    .ai-header h1 {{
        margin: 0;
        font-size: 34px;
        font-weight: 900;
    }}

    .ai-header p {{
        margin-top: 10px;
        margin-bottom: 0;
        font-size: 17px;
    }}

    .ai-control-label {{
        color: #ffffff !important;
        background: linear-gradient(
            135deg,
            #1e3a8a,
            #4f46e5
        );
        padding: 9px 13px;
        border-radius: 10px;
        font-weight: 750;
        margin-bottom: 8px;
    }}

    .ai-generated-title {{
        background: linear-gradient(
            135deg,
            #1e3a8a,
            #4f46e5
        );
        color: #ffffff !important;
        border-radius: 14px;
        padding: 14px 18px;
        font-size: 24px;
        font-weight: 850;
        margin-top: 24px;
    }}

    .ai-generated-title * {{
        color: #ffffff !important;
    }}

    .ai-question-card {{
        border-radius: 20px;
        padding: 20px;
        margin-top: 20px;
    }}

    .ai-question-number {{
        color: {COLORS["accent"]} !important;
        font-weight: 800;
        font-size: 14px;
        margin-bottom: 10px;
    }}

    .ai-question-card .question-text {{
        color: {COLORS["text"]} !important;
        font-size: 20px;
        font-weight: 850;
        line-height: 1.55;
        white-space: pre-wrap;
    }}

    /* =====================================================
       INPUTS
       ===================================================== */

    [data-testid="stRadio"] {{
        background: {COLORS["surface"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 14px;
        padding: 10px 14px;
        margin-top: 8px;
    }}

    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label *,
    [data-testid="stRadio"] p {{
        color: {COLORS["text"]} !important;
    }}

    [data-baseweb="select"] > div {{
        background: {COLORS["input_bg"]} !important;
        border-color: {COLORS["border"]} !important;
        color: {COLORS["text"]} !important;
        border-radius: 12px !important;
    }}

    [data-baseweb="select"] * {{
        color: {COLORS["text"]} !important;
    }}

    input,
    textarea {{
        background: {COLORS["input_bg"]} !important;
        color: {COLORS["text"]} !important;
        border-color: {COLORS["border"]} !important;
    }}

    /* =====================================================
       BUTTONS
       ===================================================== */

    div.stButton > button {{
        border-radius: 14px !important;
        min-height: 48px;
        font-weight: 800 !important;

        border: 1px solid rgba(129,140,248,.35) !important;

        background:
            linear-gradient(
                135deg,
                {COLORS["accent"]},
                {COLORS["accent_2"]}
            ) !important;

        color: #ffffff !important;

        box-shadow:
            0 8px 20px rgba(79,70,229,.18);

        transition:
            transform .16s ease,
            box-shadow .16s ease,
            filter .16s ease;
    }}

    div.stButton > button:hover {{
        transform: translateY(-2px);
        filter: brightness(1.07);
        box-shadow:
            0 12px 26px rgba(79,70,229,.28);
    }}

    div.stButton > button:active {{
        transform: translateY(0);
    }}

    div.stButton > button,
    div.stButton > button * {{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }}

    /* =====================================================
       ALERTS / EXPANDERS / CODE
       ===================================================== */

    [data-testid="stAlert"] {{
        border-radius: 14px !important;
    }}

    [data-testid="stExpander"] {{
        background: {COLORS["surface"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 14px !important;
    }}

    [data-testid="stExpander"] * {{
        color: {COLORS["text"]} !important;
    }}

    code,
    pre {{
        background: {COLORS["code_bg"]} !important;
        color: {COLORS["text"]} !important;
    }}

    /* =====================================================
       METRICS / PROGRESS
       ===================================================== */

    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {{
        color: {COLORS["text"]} !important;
    }}

    [data-testid="stProgressBar"] > div > div {{
        background:
            linear-gradient(
                90deg,
                {COLORS["accent"]},
                {COLORS["accent_2"]}
            ) !important;
    }}

    /* =====================================================
       ACTIVITIES
       ===================================================== */

    .activity-card {{
        background:
            linear-gradient(
                135deg,
                {COLORS["surface"]},
                {COLORS["surface_soft"]}
            ) !important;

        border: 1px solid {COLORS["border"]} !important;
        border-radius: 24px;
        padding: 24px;
        min-height: 190px;
        box-shadow: 0 12px 30px {COLORS["shadow"]};

        transition:
            transform .18s ease,
            box-shadow .18s ease;
    }}

    .activity-card * {{
        color: {COLORS["text"]} !important;
    }}

    .activity-icon {{
        font-size: 34px;
        margin-bottom: 10px;
    }}

    .activity-title {{
        font-size: 22px;
        font-weight: 900;
    }}

    .activity-desc {{
        color: {COLORS["muted"]} !important;
        line-height: 1.6;
        margin-top: 8px;
    }}

    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {{
        text-align: center;
        color: {COLORS["muted"]} !important;
        padding: 30px;
        margin-top: 30px;
        border-top: 1px solid {COLORS["border"]};
    }}

    /* =====================================================
       MOBILE
       ===================================================== */

    @media (max-width: 768px) {{

        .block-container {{
            padding-top: 1rem;
        }}

        .hero {{
            padding: 32px 24px;
            border-radius: 24px;
        }}

        .hero-title {{
            font-size: 38px;
        }}

        .hero-desc {{
            font-size: 16px;
        }}

        .section-title {{
            font-size: 24px;
        }}

        .big-score {{
            font-size: 48px;
        }}
    }}

    </style>
    """
    ),
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


def get_encouragement(score):

    if score == 100:
        return (
            L["perfect_title"],
            L["perfect_desc"]
        )

    if score >= 90:
        return (
            L["excellent_title"],
            L["excellent_desc"]
        )

    if score >= 75:
        return (
            L["good_title"],
            L["good_desc"]
        )

    if score >= 50:
        return (
            L["developing_title"],
            L["developing_desc"]
        )

    return (
        L["beginner_title"],
        L["beginner_desc"]
    )


def show_encouragement(score, focus_skill=None):

    title, description = get_encouragement(score)

    st.markdown(
        textwrap.dedent(
        f"""
        <div class="encouragement-card">

            <div class="encouragement-title">
                {title}
            </div>

            <div class="encouragement-text">
                {description}
            </div>

        </div>
        """
        ),
        unsafe_allow_html=True
    )

    if focus_skill:

        st.markdown(
            textwrap.dedent(
            f"""
            <div class="focus-card">

                <b>{L["focus_area"]}:</b>
                {html.escape(str(focus_skill))}

                <br>

                <span style="
                    color:{COLORS["muted"]} !important;
                ">
                    {L["practice_progress"]}
                </span>

            </div>
            """
            ),
            unsafe_allow_html=True
        )


def start_assessment():

    st.session_state.assessment_questions = random.sample(
        QUESTIONS,
        15
    )

    st.session_state.assessment_answers = {}

    st.session_state.assessment_index = 0

    st.session_state.assessment_submitted = False

    st.session_state.assessment_celebrated = False

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


def get_weakest_skill(questions, answers):

    skill_scores = {}

    for skill in sorted(
        set(q["skill"] for q in questions)
    ):

        skill_questions = [
            q for q in questions
            if q["skill"] == skill
        ]

        correct = sum(
            1
            for q in skill_questions
            if answers.get(q["id"]) == q["answer"]
        )

        skill_scores[skill] = round(
            correct / len(skill_questions) * 100
        )

    if not skill_scores:
        return None

    return min(
        skill_scores,
        key=skill_scores.get
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
11. Do not use HTML tags, Markdown, or code formatting.
12. Return plain text only inside question, options, answer, and solution.

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

    response = None

    for attempt in range(4):

        try:

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema
                }
            )

            break

        except Exception as e:

            if is_retryable_ai_error(e) and attempt < 3:
                time.sleep(2 ** attempt)
                continue

            raise

    if response is None:
        raise RuntimeError(
            "Gemini did not return a response."
        )

    parsed = getattr(
        response,
        "parsed",
        None
    )

    if parsed is None:
        parsed = json.loads(response.text)

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

        question = html.unescape(question)
        solution = html.unescape(solution)
        answer = html.unescape(answer)

        cleaned_options = []

        if isinstance(options, list):

            for option in options:

                cleaned_options.append(
                    html.unescape(
                        str(option).strip()
                    )
                )

        html_tags_to_remove = [
            "<div>",
            "</div>",
            "<p>",
            "</p>",
            "<span>",
            "</span>",
            "<br>",
            "<br/>",
            "<br />"
        ]

        for tag in html_tags_to_remove:

            replacement = (
                "\n"
                if "br" in tag
                else ""
            )

            question = question.replace(
                tag,
                replacement
            )

            solution = solution.replace(
                tag,
                replacement
            )

            answer = answer.replace(
                tag,
                ""
            )

            cleaned_options = [
                option.replace(
                    tag,
                    replacement
                )
                for option in cleaned_options
            ]

        question = question.strip()
        solution = solution.strip()
        answer = answer.strip()

        if (
            question
            and len(cleaned_options) == 4
            and answer
            and solution
            and answer in cleaned_options
        ):

            valid_questions.append(
                {
                    "question": question,
                    "options": cleaned_options,
                    "answer": answer,
                    "solution": solution
                }
            )

    return valid_questions


def get_ai_error_message(error):

    if client is None:
        return L["ai_setup"]

    if is_retryable_ai_error(error):
        return L["ai_busy"]

    message = str(error).lower()

    if (
        "api key" in message
        or "permission" in message
        or "authentication" in message
        or "unauthorized" in message
    ):
        return L["ai_setup"]

    if (
        "connection" in message
        or "timeout" in message
        or "network" in message
    ):
        return L["ai_connection"]

    return L["ai_error"]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        textwrap.dedent(
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
        """
        ),
        unsafe_allow_html=True
    )

    st.write("")

    language_choice = st.radio(
        L["language"],
        [
            "English",
            "العربية"
        ],
        index=(
            0
            if st.session_state.lang == "English"
            else 1
        ),
        key="language_selector"
    )

    if language_choice != st.session_state.lang:

        st.session_state.lang = language_choice
        st.rerun()

    theme_choice = st.radio(
        L["theme"],
        [
            "🌙 " + L["dark"],
            "☀️ " + L["light"]
        ],
        index=(
            0
            if st.session_state.theme == "Dark"
            else 1
        ),
        key="theme_selector"
    )

    new_theme = (
        "Dark"
        if theme_choice.startswith("🌙")
        else "Light"
    )

    if new_theme != st.session_state.theme:

        st.session_state.theme = new_theme
        st.rerun()

    ai_status_text = (
        "✓ " + L["ai_ready"]
        if client is not None
        else "⚠ " + L["ai_unavailable"]
    )

    st.caption(
        f"{L['ai_status']}: {ai_status_text}"
    )

    st.divider()

    pages = [

        ("🏠", L["home"], "Home"),

        ("📝", L["assessment"], "Assessment"),

        ("📊", L["snapshot"], "Learning Snapshot"),

        ("🔎", L["errors"], "Error Analysis"),

        ("🎯", L["practice"], "Smart Practice"),

        ("🤖", L["ai_questions"], "AI Questions"),

        ("🎉", L["activities"], "Activities"),

        ("🛤️", L["path"], "Learning Path"),

        ("🔄", L["reassessment"], "Reassessment")
    ]

    for icon, label, page_key in pages:

        if st.button(
            f"{icon}  {label}",
            use_container_width=True,
            key=f"sidebar_{page_key}"
        ):

            st.session_state.page = page_key
            st.rerun()


# =========================================================
# HOME
# =========================================================

if st.session_state.page == "Home":

    st.markdown(
        textwrap.dedent(
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
        """
        ),
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            textwrap.dedent(
            f"""
            <div class="stat-card">
                <div class="stat-label">
                    {L["questions"]}
                </div>

                <div class="stat-value">
                    30
                </div>
            </div>
            """
            ),
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            textwrap.dedent(
            f"""
            <div class="stat-card">
                <div class="stat-label">
                    {L["skills"]}
                </div>

                <div class="stat-value">
                    3
                </div>
            </div>
            """
            ),
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            textwrap.dedent(
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
            """
            ),
            unsafe_allow_html=True
        )

    st.markdown(
        textwrap.dedent(
        f"""
        <div class="section-title">
            {L["journey"]}
        </div>
        """
        ),
        unsafe_allow_html=True
    )

    cols = st.columns(4)

    journey = [

        ("01", "📝", L["assess"], L["assess_desc"]),

        ("02", "🔎", L["analyze"], L["analyze_desc"]),

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

    for col, item in zip(cols, journey):

        number, icon, title, desc = item

        with col:

            st.markdown(
                textwrap.dedent(
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
                """
                ),
                unsafe_allow_html=True
            )

    st.write("")

    if st.button(
        f"🚀  {L['start']}",
        use_container_width=True,
        key="home_start_assessment"
    ):

        start_assessment()

        st.session_state.page = "Assessment"

        st.rerun()


# =========================================================
# ASSESSMENT
# =========================================================

elif st.session_state.page == "Assessment":

    st.title(f"📝 {L['placement']}")

    st.write(L["placement_desc"])

    if not st.session_state.assessment_questions:
        start_assessment()

    if not st.session_state.assessment_submitted:

        questions = st.session_state.assessment_questions

        index = st.session_state.assessment_index

        current = questions[index]

        total = len(questions)

        progress = (index + 1) / total

        st.progress(progress)

        st.caption(
            f"{L['question']} "
            f"{index + 1} "
            f"{L['of']} "
            f"{total}"
        )

        st.markdown(
            textwrap.dedent(
            f"""
            <div class="question-card">

                <div style="
                    color:{COLORS["accent"]} !important;
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
            """
            ),
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
                current["options"].index(previous_answer)
                if previous_answer in current["options"]
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
                    use_container_width=True,
                    key="assessment_previous"
                ):

                    st.session_state.assessment_index -= 1
                    st.rerun()

        with col2:

            if index < total - 1:

                if st.button(
                    f"{L['next']} →",
                    use_container_width=True,
                    key="assessment_next"
                ):

                    st.session_state.assessment_index += 1
                    st.rerun()

            else:

                if st.button(
                    f"✓ {L['finish']}",
                    use_container_width=True,
                    key="assessment_finish"
                ):

                    score = calculate_assessment_score()

                    st.session_state.before_score = score
                    st.session_state.assessment_submitted = True

                    st.rerun()

    else:

        score = st.session_state.before_score

        level = get_level(score)

        weakest_skill = get_weakest_skill(
            st.session_state.assessment_questions,
            st.session_state.assessment_answers
        )

        st.markdown(
            textwrap.dedent(
            f"""
            <div class="hero">

                <div class="hero-small">
                    NABD ASSESSMENT RESULT
                </div>

                <div class="big-score"
                     style="color:white !important;">
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
            """
            ),
            unsafe_allow_html=True
        )

        if score >= 90 and not st.session_state.assessment_celebrated:

            st.balloons()

            st.session_state.assessment_celebrated = True

        show_encouragement(
            score,
            weakest_skill if score < 90 else None
        )

        if st.button(
            f"📊 {L['snapshot']}",
            use_container_width=True,
            key="assessment_snapshot"
        ):

            st.session_state.page = "Learning Snapshot"
            st.rerun()


# =========================================================
# SNAPSHOT
# =========================================================

elif st.session_state.page == "Learning Snapshot":

    st.title(f"📊 {L['snapshot_title']}")

    if st.session_state.before_score is None:

        st.warning(L["complete_first"])
        st.stop()

    score = st.session_state.before_score

    questions = st.session_state.assessment_questions

    answers = st.session_state.assessment_answers

    skill_scores = {}

    for skill in sorted(
        set(q["skill"] for q in questions)
    ):

        skill_questions = [
            q
            for q in questions
            if q["skill"] == skill
        ]

        correct = sum(
            1
            for q in skill_questions
            if answers.get(q["id"]) == q["answer"]
        )

        skill_scores[skill] = round(
            correct / len(skill_questions) * 100
        )

    weak_skills = [
        skill
        for skill, value in skill_scores.items()
        if value < 70
    ]

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            textwrap.dedent(
            f"""
            <div class="stat-card">

                <div class="stat-label">
                    {L["overall"]}
                </div>

                <div class="stat-value">
                    {score}%
                </div>

            </div>
            """
            ),
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            textwrap.dedent(
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
            """
            ),
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            textwrap.dedent(
            f"""
            <div class="stat-card">

                <div class="stat-label">
                    {L["assessed_skills"]}
                </div>

                <div class="stat-value">
                    {len(skill_scores)}
                </div>

            </div>
            """
            ),
            unsafe_allow_html=True
        )

    st.markdown(
        textwrap.dedent(
        f"""
        <div class="section-title">
            {L["skill_performance"]}
        </div>
        """
        ),
        unsafe_allow_html=True
    )

    for skill, value in skill_scores.items():

        st.markdown(
            textwrap.dedent(
            f"""
            <div class="skill-card">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    font-weight:800;
                    color:{COLORS["text"]} !important;
                ">

                    <span>
                        {html.escape(str(skill))}
                    </span>

                    <span>
                        {value}%
                    </span>

                </div>

            </div>
            """
            ),
            unsafe_allow_html=True
        )

        st.progress(value / 100)

    st.markdown(
        textwrap.dedent(
        f"""
        <div class="section-title">
            {L["practice_needed"]}
        </div>
        """
        ),
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
        textwrap.dedent(
        f"""
        <div class="section-title">
            📈 {L["journey"]}
        </div>
        """
        ),
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

        if st.session_state.after_score is None:

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

    st.title(f"🔎 {L['error_title']}")

    if not st.session_state.assessment_submitted:

        st.warning(L["complete_first"])
        st.stop()

    questions = st.session_state.assessment_questions

    answers = st.session_state.assessment_answers

    wrong_questions = [
        q
        for q in questions
        if answers.get(q["id"]) != q["answer"]
    ]

    if not wrong_questions:

        st.success(
            f"✓ {L['perfect']}"
        )

    else:

        st.write(
            f"{len(wrong_questions)} {L['wrong']}"
        )

        for q in wrong_questions:

            selected = answers.get(
                q["id"],
                "—"
            )

            st.markdown(
                textwrap.dedent(
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
                """
                ),
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

    st.title(f"🎯 {L['smart_title']}")

    skills = sorted(
        set(q["skill"] for q in QUESTIONS)
    )

    selected_skill = st.selectbox(
        L["choose_skill"],
        skills,
        key="practice_skill"
    )

    if st.button(
        f"✨ {L['generate']}",
        use_container_width=True,
        key="practice_generate_button"
    ):

        pool = [
            q
            for q in QUESTIONS
            if q["skill"] == selected_skill
        ]

        st.session_state.practice_questions = random.sample(
            pool,
            min(6, len(pool))
        )

        st.session_state.practice_answers = {}

        st.session_state.practice_submitted = False

        st.session_state.practice_score = None

        st.rerun()

    if st.session_state.practice_questions:

        questions = st.session_state.practice_questions

        if not st.session_state.practice_submitted:

            for i, q in enumerate(questions):

                st.markdown(
                    textwrap.dedent(
                    f"""
                    <div class="question-card">

                        <div style="
                            color:{COLORS["accent"]} !important;
                            font-weight:800;
                        ">
                            {L["question"]} {i + 1}
                        </div>

                        <h3>
                            {html.escape(get_question_text(q))}
                        </h3>

                    </div>
                    """
                    ),
                    unsafe_allow_html=True
                )

                previous = (
                    st.session_state.practice_answers.get(
                        q["id"]
                    )
                )

                answer = st.radio(
                    L["choose"],
                    q["options"],
                    index=(
                        q["options"].index(previous)
                        if previous in q["options"]
                        else None
                    ),
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
                use_container_width=True,
                key="finish_practice_button"
            ):

                correct = sum(
                    1
                    for q in questions
                    if (
                        st.session_state.practice_answers.get(
                            q["id"]
                        )
                        == q["answer"]
                    )
                )

                score = round(
                    correct / len(questions) * 100
                )

                st.session_state.practice_score = score

                st.session_state.practice_submitted = True

                st.rerun()

        else:

            score = st.session_state.practice_score

            weakest_skill = (
                selected_skill
                if score < 80
                else None
            )

            st.markdown(
                textwrap.dedent(
                f"""
                <div class="hero">

                    <div class="hero-small">
                        {L["practice_complete"]}
                    </div>

                    <div class="big-score"
                         style="color:white !important;">
                        {score}%
                    </div>

                </div>
                """
                ),
                unsafe_allow_html=True
            )

            show_encouragement(
                score,
                weakest_skill
            )

            if score >= 80:

                st.success(
                    f"🚀 {L['ready_reassess']}"
                )

            else:

                st.warning(
                    f"🎯 {L['more_practice']}"
                )

            if st.button(
                f"🔁 {L['generate']}",
                use_container_width=True,
                key="practice_again"
            ):

                pool = [
                    q
                    for q in QUESTIONS
                    if q["skill"] == selected_skill
                ]

                st.session_state.practice_questions = random.sample(
                    pool,
                    min(6, len(pool))
                )

                st.session_state.practice_answers = {}

                st.session_state.practice_submitted = False

                st.session_state.practice_score = None

                st.rerun()


# =========================================================
# AI QUESTION GENERATOR
# =========================================================

elif st.session_state.page == "AI Questions":

    st.markdown(
        textwrap.dedent(
        f"""
        <div class="ai-header">

            <h1>
                🤖 {L["ai_title"]}
            </h1>

            <p>
                {L["ai_desc"]}
            </p>

        </div>
        """
        ),
        unsafe_allow_html=True
    )

    skills = sorted(
        set(q["skill"] for q in QUESTIONS)
    )

    difficulties = [
        "Easy",
        "Medium",
        "Hard"
    ]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            textwrap.dedent(
            f"""
            <div class="ai-control-label">
                {L["ai_skill"]}
            </div>
            """
            ),
            unsafe_allow_html=True
        )

        selected_skill = st.selectbox(
            L["ai_skill"],
            skills,
            label_visibility="collapsed",
            key="ai_skill_select"
        )

    with col2:

        st.markdown(
            textwrap.dedent(
            f"""
            <div class="ai-control-label">
                {L["ai_difficulty"]}
            </div>
            """
            ),
            unsafe_allow_html=True
        )

        selected_difficulty = st.selectbox(
            L["ai_difficulty"],
            difficulties,
            label_visibility="collapsed",
            key="ai_difficulty_select"
        )

    with col3:

        st.markdown(
            textwrap.dedent(
            f"""
            <div class="ai-control-label">
                {L["ai_count"]}
            </div>
            """
            ),
            unsafe_allow_html=True
        )

        question_count = st.selectbox(
            L["ai_count"],
            [3, 5, 6, 8, 10],
            index=1,
            label_visibility="collapsed",
            key="ai_count_select"
        )

    if st.button(
        f"✨ {L['ai_generate']}",
        use_container_width=True,
        key="ai_generate_button"
    ):

        if client is None:

            st.error(
                L["ai_setup"]
            )

        else:

            language = (
                "Arabic"
                if is_arabic
                else "English"
            )

            spinner_text = (
                "جاري إنشاء الأسئلة..."
                if is_arabic
                else "Generating questions..."
            )

            with st.spinner(spinner_text):

                try:

                    generated_questions = generate_ai_questions(
                        selected_skill,
                        selected_difficulty,
                        question_count,
                        language
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
                        get_ai_error_message(e)
                    )

    if st.session_state.ai_questions:

        st.markdown(
            textwrap.dedent(
            f"""
            <div class="ai-generated-title">
                🤖 {L["ai_generated"]}
            </div>
            """
            ),
            unsafe_allow_html=True
        )

        for i, q in enumerate(
            st.session_state.ai_questions
        ):

            question_text = str(
                q.get("question", "")
            ).strip()

            question_text = html.unescape(
                question_text
            )

            question_text = (
                question_text
                .replace("<div>", "")
                .replace("</div>", "")
                .replace("<p>", "")
                .replace("</p>", "")
                .replace("<span>", "")
                .replace("</span>", "")
                .replace("<br>", "\n")
                .replace("<br/>", "\n")
                .replace("<br />", "\n")
            )

            question_text = html.escape(
                question_text
            )

            question_number = html.escape(
                str(L["question"])
            )

            st.markdown(
                textwrap.dedent(
                f"""
                <div class="ai-question-card">

                    <div class="ai-question-number">
                        {question_number} {i + 1}
                    </div>

                    <div class="question-text">
                        {question_text}
                    </div>

                </div>
                """
                ),
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
                        L["correct_feedback"]
                    )

                else:

                    st.error(
                        L["incorrect_feedback"]
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
# LEARNING ACTIVITIES
# =========================================================

elif st.session_state.page == "Activities":

    st.markdown(
        textwrap.dedent(
        f"""
        <div class="hero">

            <div class="hero-small">
                NABD • LEARN • PLAY • GROW
            </div>

            <div class="hero-title">
                🎉 {L["activities_title"]}
            </div>

            <div class="hero-desc">
                {L["activities_desc"]}
            </div>

        </div>
        """
        ),
        unsafe_allow_html=True
    )

    activity_cols = st.columns(3)

    activities = [

        (
            activity_cols[0],
            "⚡",
            L["challenge"],
            L["challenge_desc"]
        ),

        (
            activity_cols[1],
            "🧠",
            L["smart_title"],
            L["practice_desc"]
        ),

        (
            activity_cols[2],
            "🤖",
            L["ai_title"],
            L["ai_desc"]
        )
    ]

    for col, icon, title, desc in activities:

        with col:

            st.markdown(
                textwrap.dedent(
                f"""
                <div class="activity-card">

                    <div class="activity-icon">
                        {icon}
                    </div>

                    <div class="activity-title">
                        {title}
                    </div>

                    <div class="activity-desc">
                        {desc}
                    </div>

                </div>
                """
                ),
                unsafe_allow_html=True
            )

    st.write("")

    if st.button(
        f"🚀 {L['start_challenge']}",
        use_container_width=True,
        key="activity_start_challenge"
    ):

        st.session_state.challenge_questions = random.sample(
            QUESTIONS,
            min(5, len(QUESTIONS))
        )

        st.session_state.challenge_answers = {}

        st.session_state.challenge_submitted = False

        st.session_state.challenge_score = None

        st.rerun()

    if st.session_state.challenge_questions:

        st.markdown(
            textwrap.dedent(
            f"""
            <div class="section-title">
                ⚡ {L["challenge"]}
            </div>
            """
            ),
            unsafe_allow_html=True
        )

        if not st.session_state.challenge_submitted:

            questions = st.session_state.challenge_questions

            for i, q in enumerate(questions):

                st.markdown(
                    textwrap.dedent(
                    f"""
                    <div class="question-card">

                        <div style="
                            color:{COLORS["accent"]} !important;
                            font-weight:800;
                            margin-bottom:8px;
                        ">
                            {L["question"]} {i + 1}
                        </div>

                        <h3>
                            {html.escape(get_question_text(q))}
                        </h3>

                    </div>
                    """
                    ),
                    unsafe_allow_html=True
                )

                previous = (
                    st.session_state.challenge_answers.get(
                        q["id"]
                    )
                )

                answer = st.radio(
                    L["choose"],
                    q["options"],
                    index=(
                        q["options"].index(previous)
                        if previous in q["options"]
                        else None
                    ),
                    key=f"challenge_answer_{q['id']}"
                )

                st.session_state.challenge_answers[
                    q["id"]
                ] = answer

            if st.button(
                f"🏆 {L['finish']}",
                use_container_width=True,
                key="activity_finish_challenge"
            ):

                correct = sum(
                    1
                    for q in questions
                    if (
                        st.session_state.challenge_answers.get(
                            q["id"]
                        )
                        == q["answer"]
                    )
                )

                st.session_state.challenge_score = round(
                    correct / len(questions) * 100
                )

                st.session_state.challenge_submitted = True

                st.rerun()

        else:

            score = st.session_state.challenge_score

            st.markdown(
                textwrap.dedent(
                f"""
                <div class="hero">

                    <div class="hero-small">
                        {L["challenge_complete"]}
                    </div>

                    <div style="
                        font-size:58px;
                        font-weight:900;
                        margin-top:10px;
                    ">
                        {score}%
                    </div>

                    <div style="
                        font-size:20px;
                        opacity:.85;
                    ">
                        {L["challenge_score"]}
                    </div>

                </div>
                """
                ),
                unsafe_allow_html=True
            )

            if score >= 90:
                st.balloons()

            show_encouragement(score)

            if st.button(
                f"🔁 {L['challenge_again']}",
                use_container_width=True,
                key="activity_again"
            ):

                st.session_state.challenge_questions = random.sample(
                    QUESTIONS,
                    min(5, len(QUESTIONS))
                )

                st.session_state.challenge_answers = {}

                st.session_state.challenge_submitted = False

                st.session_state.challenge_score = None

                st.rerun()


# =========================================================
# LEARNING PATH
# =========================================================

elif st.session_state.page == "Learning Path":

    st.title(f"🛤️ {L['path_title']}")

    st.write(L["path_desc"])

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
            L["reassessment_title"],
            L["step4"]
        )
    ]

    for number, icon, title, desc in path_items:

        st.markdown(
            textwrap.dedent(
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
                        color:{COLORS["accent"]} !important;
                    ">
                        {number}
                    </div>

                    <div>

                        <div style="
                            font-size:25px;
                            font-weight:850;
                            color:{COLORS["text"]} !important;
                        ">
                            {icon} {title}
                        </div>

                        <div style="
                            color:{COLORS["muted"]} !important;
                            margin-top:5px;
                        ">
                            {desc}
                        </div>

                    </div>

                </div>

            </div>
            """
            ),
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

        st.session_state.reassessment_questions = random.sample(
            QUESTIONS,
            10
        )

        st.session_state.reassessment_answers = {}

        st.session_state.reassessment_submitted = False

    if not st.session_state.reassessment_submitted:

        questions = (
            st.session_state.reassessment_questions
        )

        for i, q in enumerate(questions):

            st.markdown(
                textwrap.dedent(
                f"""
                <div class="question-card">

                    <div style="
                        color:{COLORS["accent"]} !important;
                        font-weight:800;
                    ">
                        {L["question"]} {i + 1}
                    </div>

                    <h3>
                        {html.escape(get_question_text(q))}
                    </h3>

                </div>
                """
                ),
                unsafe_allow_html=True
            )

            previous = (
                st.session_state.reassessment_answers.get(
                    q["id"]
                )
            )

            answer = st.radio(
                L["choose"],
                q["options"],
                index=(
                    q["options"].index(previous)
                    if previous in q["options"]
                    else None
                ),
                key=f"reassessment_{q['id']}"
            )

            st.session_state.reassessment_answers[
                q["id"]
            ] = answer

        if st.button(
            f"✓ {L['finish']}",
            use_container_width=True,
            key="reassessment_finish_button"
        ):

            correct = sum(
                1
                for q in questions
                if (
                    st.session_state.reassessment_answers.get(
                        q["id"]
                    )
                    == q["answer"]
                )
            )

            score = round(
                correct / len(questions) * 100
            )

            st.session_state.after_score = score

            st.session_state.reassessment_submitted = True

            st.rerun()

    else:

        before = st.session_state.before_score

        after = st.session_state.after_score

        change = after - before

        st.markdown(
            textwrap.dedent(
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
            """
            ),
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

        if change > 0:

            st.success(
                f"🎉 {L['improved']}"
            )

        elif change == 0:

            st.info(
                f"🌱 {L['same_score']}"
            )

        else:

            st.info(
                f"💪 {L['review_again']}"
            )

        show_encouragement(after)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    textwrap.dedent(
    f"""
    <div class="footer">
        {L["footer"]}<br>
        © 2026 MINNA MOHAMMED — NABD Educational Platform.
        All rights reserved.
    </div>
    """
    ),
    unsafe_allow_html=True
)
