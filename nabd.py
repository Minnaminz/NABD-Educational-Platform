from pathlib import Path
import random
import json
import time

from google import genai
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.tree import DecisionTreeClassifier


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NABD | Educational Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. TRANSLATIONS (قاموس اللغات الكامل)
# =========================================================

T = {
    "English": {
        "title": "NABD Platform 🧠",
        "subtitle": "Smart Diagnostic Assessment & AI Learning Platform",
        "home": "Home",
        "assessment": "Diagnostic Assessment",
        "snapshot": "Learning Snapshot",
        "errors": "Error Analysis",
        "path": "Learning Path",
        "practice": "Smart Practice",
        "activities": "Games & Activities",
        "ai_questions": "AI Generator",
        "reassessment": "Reassessment",
        "language": "Language / اللغة",
        "theme": "Theme / المظهر",
        "dark": "Dark",
        "light": "Light",
        "streak": "Streak",
        "points": "Points",
        "days": "Days",
        "start_btn": "Start Assessment Now 🚀",
        "next": "Next",
        "previous": "Previous",
        "finish": "Finish & View Results 🏁",
        "choose": "Select the correct answer:",
        "score_msg_high": "🎉 Excellent Work! Outstanding performance!",
        "score_msg_low": "💪 Great Effort! Review your customized learning path below.",
        "view_snapshot": "View Learning Snapshot 📊",
        "speed_drill": "⚡ 60-Second Speed Drill",
        "speed_desc": "Answer as many questions as you can in one minute!",
        "start_speed": "Start Speed Challenge ⏱️",
        "time_left": "Time Remaining",
        "submit": "Submit Answer",
        "error_detective": "🕵️ Error Detective",
        "error_q": "Find the step containing the mistake in solving: 3x + 5 = 20",
        "daily_q": "📅 Question of the Day",
        "daily_title": "What is half of 2¹⁰?",
        "generate_btn": "Generate AI Questions 🪄",
        "footer": "© 2026 MINNA MOHAMMED — NABD Educational Platform. All rights reserved."
    },
    "العربية": {
        "title": "منصة نَبْض 🧠",
        "subtitle": "نظام التقييم التشخيصي الذكي والتعلم بالذكاء الاصطناعي",
        "home": "الرئيسية",
        "assessment": "التقييم التشخيصي",
        "snapshot": "ملخص التعلم",
        "errors": "تحليل الأخطاء",
        "path": "مسار التعلم",
        "practice": "التدريب الذكي",
        "activities": "الألعاب والفعاليات",
        "ai_questions": "مولّد الأسئلة (AI)",
        "reassessment": "إعادة التقييم",
        "language": "Language / اللغة",
        "theme": "Theme / المظهر",
        "dark": "داكن",
        "light": "فاتح",
        "streak": "السلسلة",
        "points": "النقاط",
        "days": "أيام",
        "start_btn": "ابدأ التقييم التشخيصي الآن 🚀",
        "next": "التالي",
        "previous": "السابق",
        "finish": "إنهاء وإظهار النتيجة 🏁",
        "choose": "اختر الإجابة الصحيحة:",
        "score_msg_high": "🎉 أداء أسطوري وممتاز جداً!",
        "score_msg_low": "💪 بداية رائعة! يمكنك مراجعة مسار التعلم المخصص لتحسين مستواك.",
        "view_snapshot": "عرض ملخص التعلم 📊",
        "speed_drill": "⚡ تحدي الـ 60 ثانية",
        "speed_desc": "أجب على أكبر عدد من الأسئلة في دقيقة واحدة!",
        "start_speed": "ابدأ التحدي الآن ⏱️",
        "time_left": "الوقت المتبقي",
        "submit": "تأكيد الإجابة",
        "error_detective": "🕵️ لعبة كاشف الأخطاء",
        "error_q": "حدد الخطوة التي تحتوي على الخطأ في حل: ٣س + ٥ = ٢٠",
        "daily_q": "📅 سؤال اليوم",
        "daily_title": "ما هو نصف العدد ٢¹⁰؟",
        "generate_btn": "توليد أسئلة بالذكاء الاصطناعي 🪄",
        "footer": "© 2026 MINNA MOHAMMED — NABD Educational Platform. All rights reserved."
    }
}


# =========================================================
# 3. SESSION STATE INITIALIZATION
# =========================================================

for key, default in [
    ("lang", "English"), ("theme", "Dark"), ("page", "Home"),
    ("assessment_questions", []), ("assessment_answers", {}),
    ("assessment_index", 0), ("assessment_submitted", False),
    ("before_score", None), ("after_score", None),
    ("reassess_questions", []), ("reassess_answers", {}),
    ("reassess_index", 0), ("reassess_submitted", False),
    ("streak", 5), ("user_points", 320),
    ("timer_start", None), ("speed_score", 0), ("speed_index", 0)
]:
    if key not in st.session_state:
        st.session_state[key] = default

L = T[st.session_state.lang]
is_arabic = (st.session_state.lang == "العربية")


# =========================================================
# 4. SOUND & EFFECTS
# =========================================================

def play_audio_and_effects(score):
    if score >= 75:
        st.balloons()
        audio_html = """
            <audio autoplay>
                <source src="https://assets.mixkit.co/active_storage/sfx/2013/2013-preview.mp3" type="audio/mpeg">
            </audio>
        """
        components.html(audio_html, height=0)


# =========================================================
# 5. GEMINI API SETUP
# =========================================================

try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    client = None


def generate_ai_questions(skill, difficulty, count, language):
    if client is None:
        raise RuntimeError("Gemini API key is missing.")

    prompt = f"""
Create exactly {count} multiple-choice math questions in {language}.
Skill: {skill}
Difficulty: {difficulty}

Return ONLY a valid JSON array of objects with keys: "question", "options" (array of 4), "answer", "solution".
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)


# =========================================================
# 6. DATA LOADING & ML MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "nabd_data.csv"

try:
    df = pd.read_csv(DATA_FILE)
    df["student_id"] = df["student_id"].fillna(0)
    df["concept"] = df["concept"].fillna("Unknown").astype(str)
    df["skill"] = df["skill"].fillna("Unknown").astype(str)
    df["error_type"] = df["error_type"].fillna("Unknown").astype(str)
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
except Exception:
    df = pd.DataFrame(columns=["student_id", "concept", "skill", "error_type", "correct"])

if not df.empty:
    X = pd.get_dummies(df[["concept", "skill", "correct"]], drop_first=True)
    y = df["error_type"]
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    feature_columns = X.columns
else:
    clf = None
    feature_columns = []


# =========================================================
# 7. MODERN STYLES & ELEGANT PALETTE
# =========================================================

if st.session_state.theme == "Dark":
    BG = "#0f172a"
    CARD_BG = "#1e293b"
    TEXT_COLOR = "#f8fafc"
    BORDER_COLOR = "#334155"
    ACCENT_GRAD = "linear-gradient(135deg, #6366f1, #8b5cf6)"
else:
    BG = "#f8fafc"
    CARD_BG = "#ffffff"
    TEXT_COLOR = "#0f172a"
    BORDER_COLOR = "#e2e8f0"
    ACCENT_GRAD = "linear-gradient(135deg, #4f46e5, #7c3aed)"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {BG}; color: {TEXT_COLOR}; font-family: 'Inter', system-ui, sans-serif; }}
    
    /* Elegant Hero Banner */
    .hero-card {{
        background: {ACCENT_GRAD};
        padding: 30px;
        border-radius: 18px;
        color: white !important;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.25);
        margin-bottom: 25px;
    }}
    .hero-card h1, .hero-card p {{ color: white !important; margin: 0; }}
    
    /* Modern Custom Cards */
    .custom-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }}
    
    /* Buttons Styling */
    div.stButton > button {{
        background: {ACCENT_GRAD} !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }}
    div.stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(99, 102, 241, 0.35);
    }}
    </style>
""", unsafe_allow_html=True)


# =========================================================
# 8. QUESTION BANK (Bilingual)
# =========================================================

QUESTIONS = [
    {
        "id": 1, "concept": "Algebra", "skill": "Linear Equations",
        "question_ar": "أوجد قيمة س: س + ٥ = ١٢", "question_en": "Solve for x: x + 5 = 12",
        "options": ["5", "6", "7", "8"], "answer": "7",
        "solution_ar": "س = ١٢ - ٥ = ٧", "solution_en": "x = 12 - 5 = 7"
    },
    {
        "id": 2, "concept": "Algebra", "skill": "Linear Equations",
        "question_ar": "أوجد قيمة س: ٢س = ١٠", "question_en": "Solve for x: 2x = 10",
        "options": ["2", "5", "8", "10"], "answer": "5",
        "solution_ar": "س = ١٠ ÷ ٢ = ٥", "solution_en": "x = 10 / 2 = 5"
    },
    {
        "id": 3, "concept": "Arithmetic", "skill": "Percentages",
        "question_ar": "ما قيمة ١٠٪ من ٥٠؟", "question_en": "What is 10% of 50?",
        "options": ["2", "5", "10", "15"], "answer": "5",
        "solution_ar": "٠٫١ × ٥٠ = ٥", "solution_en": "0.1 * 50 = 5"
    },
    {
        "id": 4, "concept": "Arithmetic", "skill": "Multiplication",
        "question_ar": "ما حاصل ضرب ١٢ × ٨؟", "question_en": "What is 12 x 8?",
        "options": ["86", "96", "106", "116"], "answer": "96",
        "solution_ar": "١٢ × ٨ = ٩٦", "solution_en": "12 * 8 = 96"
    }
]


# =========================================================
# 9. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🧠 NABD</h1>", unsafe_allow_html=True)
    
    # Gamification Stats
    st.markdown(f"""
        <div style="background:{CARD_BG}; border:1px solid {BORDER_COLOR}; border-radius:12px; padding:10px; text-align:center; font-weight:bold; margin-bottom:15px;">
            🔥 {L['streak']}: {st.session_state.streak} {L['days']} | ⭐ {L['points']}: {st.session_state.user_points}
        </div>
    """, unsafe_allow_html=True)

    # Language & Theme Options
    selected_lang = st.radio(L["language"], ["English", "العربية"], index=0 if st.session_state.lang=="English" else 1)
    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang
        st.rerun()

    selected_theme = st.radio(L["theme"], [L["dark"], L["light"]], index=0 if st.session_state.theme=="Dark" else 1)
    st.session_state.theme = "Dark" if selected_theme == L["dark"] else "Light"

    st.divider()

    # Nav List
    nav_items = [
        ("🏠", L["home"], "Home"),
        ("📝", L["assessment"], "Assessment"),
        ("📊", L["snapshot"], "Learning Snapshot"),
        ("🔎", L["errors"], "Error Analysis"),
        ("🛤️", L["path"], "Learning Path"),
        ("🎯", L["practice"], "Smart Practice"),
        ("🎮", L["activities"], "Activities"),
        ("🤖", L["ai_questions"], "AI Questions"),
        ("🔄", L["reassessment"], "Reassessment")
    ]

    for icon, label, key in nav_items:
        if st.button(f"{icon} {label}", use_container_width=True, key=f"nav_{key}"):
            st.session_state.page = key
            st.rerun()


# =========================================================
# PAGE 1: HOME
# =========================================================

if st.session_state.page == "Home":
    st.markdown(f"""
        <div class="hero-card">
            <h1 style="font-size: 32px; font-weight:700;">{L["title"]}</h1>
            <p style="font-size: 16px; opacity: 0.9; margin-top:8px;">{L["subtitle"]}</p>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"🔥 {L['streak']}", f"{st.session_state.streak} {L['days']}")
    c2.metric(f"⭐ {L['points']}", f"{st.session_state.user_points}")
    c3.metric("🎯 Activities", "3 Active Challenges")

    st.write("")
    if st.button(L["start_btn"], use_container_width=True):
        st.session_state.assessment_questions = random.sample(QUESTIONS, len(QUESTIONS))
        st.session_state.page = "Assessment"
        st.rerun()


# =========================================================
# PAGE 2: ASSESSMENT
# =========================================================

elif st.session_state.page == "Assessment":
    st.title(f"📝 {L['assessment']}")

    if not st.session_state.assessment_submitted:
        q_list = st.session_state.assessment_questions
        if not q_list:
            st.session_state.assessment_questions = random.sample(QUESTIONS, len(QUESTIONS))
            q_list = st.session_state.assessment_questions

        idx = st.session_state.assessment_index
        q = q_list[idx]

        st.progress((idx + 1) / len(q_list))
        st.subheader(f"Q{idx + 1}/{len(q_list)}: {q['question_en' if st.session_state.lang=='English' else 'question_ar']}")

        ans = st.radio(L["choose"], q["options"], key=f"q_ans_{q['id']}")
        st.session_state.assessment_answers[q["id"]] = ans

        col_p, col_n = st.columns(2)
        if idx > 0 and col_p.button(L["previous"]):
            st.session_state.assessment_index -= 1
            st.rerun()

        if idx < len(q_list) - 1:
            if col_n.button(L["next"]):
                st.session_state.assessment_index += 1
                st.rerun()
        else:
            if col_n.button(L["finish"]):
                correct = sum(1 for item in q_list if st.session_state.assessment_answers.get(item["id"]) == item["answer"])
                score = round((correct / len(q_list)) * 100)
                st.session_state.before_score = score
                st.session_state.assessment_submitted = True
                st.rerun()

    else:
        score = st.session_state.before_score
        play_audio_and_effects(score)

        st.success(f"Score: {score}%")
        st.markdown(f"### {L['score_msg_high'] if score >= 75 else L['score_msg_low']}")

        if st.button(L["view_snapshot"]):
            st.session_state.page = "Learning Snapshot"
            st.rerun()


# =========================================================
# PAGE 3: LEARNING SNAPSHOT
# =========================================================

elif st.session_state.page == "Learning Snapshot":
    st.title(f"📊 {L['snapshot']}")

    if st.session_state.before_score is not None:
        st.metric("Assessment Score", f"{st.session_state.before_score}%")
        st.progress(st.session_state.before_score / 100)

        st.subheader("Skills Overview")
        q_list = st.session_state.assessment_questions
        for item in q_list:
            u_ans = st.session_state.assessment_answers.get(item["id"])
            is_corr = (u_ans == item["answer"])
            status = "✅" if is_corr else "❌"

            with st.expander(f"{status} Skill: {item['skill']} ({item['concept']})"):
                st.write(f"**Question:** {item['question_en' if st.session_state.lang=='English' else 'question_ar']}")
                st.write(f"**Your Answer:** {u_ans}")
                st.write(f"**Correct Answer:** {item['answer']}")
                st.write(f"💡 **Solution:** {item['solution_en' if st.session_state.lang=='English' else 'solution_ar']}")
    else:
        st.warning("Please complete the assessment first.")


# =========================================================
# PAGE 4: ERROR ANALYSIS
# =========================================================

elif st.session_state.page == "Error Analysis":
    st.title(f"🔎 {L['errors']}")
    st.write("AI-powered Decision Tree model analyzing common misconceptions.")

    if clf is not None:
        sample_concept = st.selectbox("Select Concept:", df["concept"].unique())
        sample_skill = st.selectbox("Select Skill:", df[df["concept"] == sample_concept]["skill"].unique())
        sample_correct = st.radio("Attempt Status:", ["Correct (1)", "Incorrect (0)"])

        if st.button("Analyze Error Pattern 🧠"):
            input_data = pd.DataFrame([{
                "concept": sample_concept,
                "skill": sample_skill,
                "correct": 1 if "1" in sample_correct else 0
            }])
            input_encoded = pd.get_dummies(input_data).reindex(columns=feature_columns, fill_value=0)
            pred_error = clf.predict(input_encoded)[0]

            st.info(f"🎯 **Diagnostic Result:** Detected pattern -> **{pred_error}**")
    else:
        st.info("Loading training dataset...")


# =========================================================
# PAGE 5: LEARNING PATH
# =========================================================

elif st.session_state.page == "Learning Path":
    st.title(f"🛤️ {L['path']}")

    st.markdown("""
        <div class="custom-card">
            <h3>📍 Step 1: Linear Equations Mastery</h3>
            <p>Review transposition steps, sign inversion, and coefficient separation.</p>
        </div>
        <div class="custom-card">
            <h3>📍 Step 2: Mental Math Speed Drills</h3>
            <p>Practice percentage short-cuts and fast arithmetic multiplications.</p>
        </div>
        <div class="custom-card">
            <h3>📍 Step 3: Reassessment Challenge</h3>
            <p>Retake assessment to target 90%+ mastery rating.</p>
        </div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE 6: SMART PRACTICE
# =========================================================

elif st.session_state.page == "Smart Practice":
    st.title(f"🎯 {L['practice']}")

    p_q = QUESTIONS[0]
    st.markdown(f"### {p_q['question_en' if st.session_state.lang=='English' else 'question_ar']}")
    user_p = st.radio(L["choose"], p_q["options"], key="pract_1")

    if st.button(L["submit"]):
        if user_p == p_q["answer"]:
            st.success("Correct answer! Great job! ✨")
        else:
            st.error(f"Incorrect. The correct answer is {p_q['answer']}")


# =========================================================
# PAGE 7: ACTIVITIES & GAMES
# =========================================================

elif st.session_state.page == "Activities":
    st.title(f"🎮 {L['activities']}")

    tab1, tab2, tab3 = st.tabs([L["speed_drill"], L["error_detective"], L["daily_q"]])

    # 1. SPEED DRILL
    with tab1:
        st.subheader(L["speed_drill"])
        st.caption(L["speed_desc"])

        if st.button(L["start_speed"], key="start_speed"):
            st.session_state.timer_start = time.time()
            st.session_state.speed_score = 0
            st.session_state.speed_index = 0

        if st.session_state.timer_start:
            elapsed = time.time() - st.session_state.timer_start
            time_left = max(0, 60 - int(elapsed))

            st.metric(L["time_left"], f"⏱️ {time_left}s")

            if time_left > 0:
                sq = QUESTIONS[st.session_state.speed_index % len(QUESTIONS)]
                st.write(f"**Q:** {sq['question_en' if st.session_state.lang=='English' else 'question_ar']}")
                user_a = st.radio(L["choose"], sq["options"], key=f"sp_{st.session_state.speed_index}")

                if st.button(L["submit"], key=f"btn_sp_{st.session_state.speed_index}"):
                    if user_a == sq["answer"]:
                        st.session_state.speed_score += 10
                        st.toast("Correct! +10 pts", icon="🎉")
                    st.session_state.speed_index += 1
                    st.rerun()
            else:
                st.success(f"🏆 Time is up! Total score: {st.session_state.speed_score} pts!")

    # 2. ERROR DETECTIVE
    with tab2:
        st.subheader(L["error_detective"])
        st.info(L["error_q"])
        st.write("• **Step 1:** 3x = 20 + 5")
        st.write("• **Step 2:** 3x = 25")
        st.write("• **Step 3:** x = 25 / 3")

        choice = st.radio("Which step contains the mistake?", ["Step 1", "Step 2", "Step 3"])
        if st.button("Check Detective Answer 🔍"):
            if choice == "Step 1":
                st.balloons()
                st.success("🎉 Correct! Moving +5 to the other side requires subtraction (-5), not addition.")
            else:
                st.error("❌ Try again! Check the sign in Step 1.")

    # 3. QUESTION OF THE DAY
    with tab3:
        st.subheader(L["daily_q"])
        st.write(L["daily_title"])
        ans_day = st.radio(L["choose"], ["2^5", "2^9", "1^10", "2^8"])
        if st.button("Submit Daily Challenge 🌟"):
            if ans_day == "2^9":
                st.success("Correct! Daily streak maintained 🔥 (+1 Day)")
            else:
                st.error("Try again tomorrow!")


# =========================================================
# PAGE 8: AI GENERATOR
# =========================================================

elif st.session_state.page == "AI Questions":
    st.title("🤖 AI Question Generator (Gemini)")

    if client:
        selected_skill = st.selectbox("Skill:", ["Linear Equations", "Percentages", "Multiplication"])
        selected_diff = st.select_slider("Difficulty:", ["Easy", "Medium", "Hard"])
        q_count = st.number_input("Count:", min_value=1, max_value=5, value=2)

        if st.button(L["generate_btn"]):
            with st.spinner("Generating with Gemini AI..."):
                try:
                    generated = generate_ai_questions(selected_skill, selected_diff, q_count, st.session_state.lang)
                    st.success("Generated Successfully!")
                    for i, q_item in enumerate(generated):
                        st.markdown(f"**Q{i+1}: {q_item['question']}**")
                        st.write(f"Options: {', '.join(q_item['options'])}")
                        st.caption(f"💡 Solution: {q_item['solution']}")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("Please configure GEMINI_API_KEY in `.streamlit/secrets.toml`.")


# =========================================================
# PAGE 9: REASSESSMENT
# =========================================================

elif st.session_state.page == "Reassessment":
    st.title(f"🔄 {L['reassessment']}")

    if not st.session_state.reassess_submitted:
        q_list = st.session_state.reassess_questions
        if not q_list:
            st.session_state.reassess_questions = random.sample(QUESTIONS, len(QUESTIONS))
            q_list = st.session_state.reassess_questions

        idx = st.session_state.reassess_index
        q = q_list[idx]

        st.subheader(f"Reassessment Q{idx + 1}: {q['question_en' if st.session_state.lang=='English' else 'question_ar']}")
        ans = st.radio(L["choose"], q["options"], key=f"rq_ans_{q['id']}")
        st.session_state.reassess_answers[q["id"]] = ans

        if idx < len(q_list) - 1:
            if st.button(L["next"]):
                st.session_state.reassess_index += 1
                st.rerun()
        else:
            if st.button(L["finish"]):
                correct = sum(1 for item in q_list if st.session_state.reassess_answers.get(item["id"]) == item["answer"])
                st.session_state.after_score = round((correct / len(q_list)) * 100)
                st.session_state.reassess_submitted = True
                st.rerun()

    else:
        before = st.session_state.before_score or 0
        after = st.session_state.after_score

        play_audio_and_effects(after)
        st.success(f"Reassessment Score: {after}%")

        c1, c2 = st.columns(2)
        c1.metric("Previous Score", f"{before}%")
        c2.metric("New Score", f"{after}%", delta=f"{after - before}%")


# =========================================================
# FOOTER
# =========================================================

st.markdown(f"""
    <hr style="margin-top:40px; border-color:{BORDER_COLOR};">
    <div style="text-align: center; color: #94a3b8; font-size: 13px; padding: 10px 0;">
        {L['footer']}
    </div>
""", unsafe_allow_html=True)
