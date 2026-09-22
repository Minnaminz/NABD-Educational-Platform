from pathlib import Path
import random
import html
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
    page_title="NABD | منصة نبض التعليمية",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. SOUND & EFFECTS HELPER (تشجيع وحفاوة للنتائج)
# =========================================================

def play_audio_and_effects(score):
    if score >= 75:
        st.balloons()
        # صوت تصفيق وتشجيع احتفالي
        audio_html = """
            <audio autoplay>
                <source src="https://assets.mixkit.co/active_storage/sfx/2013/2013-preview.mp3" type="audio/mpeg">
            </audio>
        """
        components.html(audio_html, height=0)
    else:
        # صوت تحفيزي هادئ
        audio_html = """
            <audio autoplay>
                <source src="https://assets.mixkit.co/active_storage/sfx/1435/1435-preview.mp3" type="audio/mpeg">
            </audio>
        """
        components.html(audio_html, height=0)


# =========================================================
# 3. GEMINI CLIENT SETUP
# =========================================================

try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    client = None


def generate_ai_questions(skill, difficulty, count, language):
    if client is None:
        raise RuntimeError("Gemini API is not configured.")

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
# 4. PATHS + DATA LOADING & MODEL TRAINING
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

# تدريب نموذج شجرة القرار لتحليل الأخطاء (Decision Tree Classifier)
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
# 5. TRANSLATIONS (اللغات)
# =========================================================

T = {
    "English": {
        "home": "Home", "assessment": "Assessment", "snapshot": "Snapshot",
        "errors": "Error Analysis", "practice": "Smart Practice", "ai_questions": "AI Generator",
        "activities": "Activities & Games", "path": "Learning Path", "reassessment": "Reassessment",
        "language": "Language", "theme": "Theme", "dark": "Dark", "light": "Light",
        "start": "Start Assessment", "score": "Score", "level": "Level",
        "next": "Next", "previous": "Previous", "finish": "Finish", "choose": "Select Answer"
    },
    "العربية": {
        "home": "الرئيسية", "assessment": "التقييم التشخيصي", "snapshot": "ملخص التعلم",
        "errors": "تحليل الأخطاء", "practice": "التدريب الذكي", "ai_questions": "مولّد الأسئلة (AI)",
        "activities": "الفعاليات والألعاب", "path": "مسار التعلم", "reassessment": "إعادة التقييم",
        "language": "اللغة", "theme": "المظهر", "dark": "داكن", "light": "فاتح",
        "start": "ابدأ التقييم", "score": "النتيجة", "level": "المستوى",
        "next": "التالي", "previous": "السابق", "finish": "إنهاء الاختبار", "choose": "اختر الإجابة الصحيحة"
    }
}


# =========================================================
# 6. SESSION STATE INITIALIZATION
# =========================================================

for key, default in [
    ("lang", "العربية"), ("theme", "Dark"), ("page", "Home"),
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
# 7. STYLES & MODERN COLOR PALETTE
# =========================================================

if st.session_state.theme == "Dark":
    COLORS = {
        "page": "#0b0f19", "surface": "#1e293b", "surface_alt": "#334155",
        "text": "#f8fafc", "muted": "#94a3b8", "border": "#475569",
        "accent": "#6366f1", "accent_grad": "linear-gradient(135deg, #4f46e5, #7c3aed)",
        "card_bg": "rgba(30, 41, 59, 0.85)"
    }
else:
    COLORS = {
        "page": "#f1f5f9", "surface": "#ffffff", "surface_alt": "#e2e8f0",
        "text": "#0f172a", "muted": "#64748b", "border": "#cbd5e1",
        "accent": "#4f46e5", "accent_grad": "linear-gradient(135deg, #6366f1, #a855f7)",
        "card_bg": "#ffffff"
    }

st.markdown(f"""
    <style>
    body {{ background-color: {COLORS["page"]}; color: {COLORS["text"]}; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
    .stApp {{ background-color: {COLORS["page"]}; }}
    
    /* Hero Banner */
    .hero-banner {{
        background: {COLORS["accent_grad"]};
        padding: 35px;
        border-radius: 20px;
        color: white !important;
        box-shadow: 0 15px 30px rgba(99, 102, 241, 0.3);
        margin-bottom: 25px;
    }}
    .hero-banner * {{ color: white !important; }}
    
    /* Gamification Status Bar */
    .badge-card {{
        background: {COLORS["card_bg"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 14px;
        padding: 12px 18px;
        display: flex;
        justify-content: space-around;
        align-items: center;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }}

    /* Card Box Design */
    .custom-card {{
        background: {COLORS["card_bg"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    }}
    
    /* Custom Modern Buttons */
    div.stButton > button {{
        background: {COLORS["accent_grad"]} !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 22px !important;
        font-weight: bold !important;
        transition: all 0.25s ease !important;
    }}
    div.stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(99, 102, 241, 0.4);
    }}
    </style>
""", unsafe_allow_html=True)


# =========================================================
# 8. QUESTION BANK
# =========================================================

QUESTIONS = [
    {"id": 1, "concept": "Algebra", "skill": "Linear Equations", "question_ar": "أوجد قيمة س: س + ٥ = ١٢", "question_en": "Solve: x + 5 = 12", "options": ["5", "6", "7", "8"], "answer": "7", "solution_ar": "س = ١٢ - ٥ = ٧", "solution_en": "x = 12 - 5 = 7"},
    {"id": 2, "concept": "Algebra", "skill": "Linear Equations", "question_ar": "أوجد قيمة س: ٢س = ١٠", "question_en": "Solve: 2x = 10", "options": ["2", "5", "8", "10"], "answer": "5", "solution_ar": "س = ١٠ ÷ ٢ = ٥", "solution_en": "x = 10 / 2 = 5"},
    {"id": 3, "concept": "Arithmetic", "skill": "Percentages", "question_ar": "ما قيمة ١٠٪ من ٥٠؟", "question_en": "What is 10% of 50?", "options": ["2", "5", "10", "15"], "answer": "5", "solution_ar": "٠٫١ × ٥٠ = ٥", "solution_en": "0.1 * 50 = 5"},
    {"id": 4, "concept": "Arithmetic", "skill": "Multiplication", "question_ar": "ما حاصل ضرب ١٢ × ٨؟", "question_en": "What is 12 x 8?", "options": ["86", "96", "106", "116"], "answer": "96", "solution_ar": "١٢ × ٨ = ٩٦", "solution_en": "12 * 8 = 96"},
]


# =========================================================
# 9. SIDEBAR & NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 32px;'>🧠 NABD</h1>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="badge-card">
            <div>🔥 {st.session_state.streak} أيام</div>
            <div>⭐ {st.session_state.user_points} نقطة</div>
        </div>
    """, unsafe_allow_html=True)
    st.write("")

    st.session_state.lang = st.radio(L["language"], ["العربية", "English"], index=0 if is_arabic else 1)
    st.session_state.theme = st.radio(L["theme"], [L["dark"], L["light"]], index=0 if st.session_state.theme=="Dark" else 1)
    
    st.divider()

    nav_options = [
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

    for icon, label, page_key in nav_options:
        if st.button(f"{icon} {label}", use_container_width=True, key=f"nav_{page_key}"):
            st.session_state.page = page_key
            st.rerun()


# =========================================================
# PAGE 1: HOME
# =========================================================

if st.session_state.page == "Home":
    st.markdown(f"""
        <div class="hero-banner">
            <h1 style="font-size: 38px; margin-bottom: 8px;">مرحباً بك في منصة نَبْض (NABD) 🧠</h1>
            <p style="font-size: 17px;">نظام التقييم الذكي وتشخيص المفاهيم بالألعاب والذكاء الاصطناعي.</p>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("🔥 حماسك اليومي", f"{st.session_state.streak} أيام متتالية")
    c2.metric("⭐ مجموع نقاطك", f"{st.session_state.user_points} نقطة")
    c3.metric("🎯 التحديات المكتملة", "5 ألعاب تفاعلية")

    st.write("")
    if st.button("🚀 ابدأ التقييم التشخيصي الآن", use_container_width=True):
        st.session_state.assessment_questions = random.sample(QUESTIONS, len(QUESTIONS))
        st.session_state.page = "Assessment"
        st.rerun()


# =========================================================
# PAGE 2: DIAGNOSTIC ASSESSMENT
# =========================================================

elif st.session_state.page == "Assessment":
    st.title("📝 التقييم التشخيصي")
    
    if not st.session_state.assessment_submitted:
        q_list = st.session_state.assessment_questions
        if not q_list:
            st.session_state.assessment_questions = random.sample(QUESTIONS, len(QUESTIONS))
            q_list = st.session_state.assessment_questions

        idx = st.session_state.assessment_index
        q = q_list[idx]

        st.progress((idx + 1) / len(q_list))
        st.subheader(f"السؤال {idx + 1} من {len(q_list)}: {q['question_ar' if is_arabic else 'question_en']}")
        
        ans = st.radio(L["choose"], q["options"], key=f"q_ans_{q['id']}")
        st.session_state.assessment_answers[q["id"]] = ans

        col_p, col_n = st.columns(2)
        if idx > 0 and col_p.button("السابق"):
            st.session_state.assessment_index -= 1
            st.rerun()
        if idx < len(q_list) - 1:
            if col_n.button("التالي"):
                st.session_state.assessment_index += 1
                st.rerun()
        else:
            if col_n.button("إنهاء واظهار النتيجة 🏁"):
                correct = sum(1 for item in q_list if st.session_state.assessment_answers.get(item["id"]) == item["answer"])
                score = round((correct / len(q_list)) * 100)
                st.session_state.before_score = score
                st.session_state.assessment_submitted = True
                st.rerun()

    else:
        score = st.session_state.before_score
        play_audio_and_effects(score)
        
        if score >= 75:
            st.success(f"🎉 أسطوري! درجتك: {score}%")
            st.markdown("### 👏 ممتاز جداً! أداؤك عالي وتفكيرك الرياضي سريع وقوي.")
        else:
            st.info(f"💪 بداية رائعة! درجتك: {score}%")
            st.markdown("### 🌟 لا تقلق! الأخطاء هي بداية التعلم. المنصة جهزت لك مساراً مخصصاً لتقوية مهاراتك!")

        if st.button("عرض ملخص التعلم ومسار التدريب 📊"):
            st.session_state.page = "Learning Snapshot"
            st.rerun()


# =========================================================
# PAGE 3: LEARNING SNAPSHOT
# =========================================================

elif st.session_state.page == "Learning Snapshot":
    st.title("📊 ملخص التعلم (Learning Snapshot)")
    
    if st.session_state.before_score is not None:
        c1, c2 = st.columns(2)
        c1.metric("درجة التقييم الحالي", f"{st.session_state.before_score}%")
        c2.metric("الحالة التعليمية", "يحتاج دعم خفيف" if st.session_state.before_score < 75 else "متقن للمفاهيم")
        
        st.progress(st.session_state.before_score / 100)
        
        st.subheader("تفاصيل أداء المهارات")
        q_list = st.session_state.assessment_questions
        for item in q_list:
            u_ans = st.session_state.assessment_answers.get(item["id"])
            is_corr = (u_ans == item["answer"])
            status_icon = "✅" if is_corr else "❌"
            
            with st.expander(f"{status_icon} المهارة: {item['skill']} ({item['concept']})"):
                st.write(f"**السؤال:** {item['question_ar' if is_arabic else 'question_en']}")
                st.write(f"**إجابتك:** {u_ans}")
                st.write(f"**الإجابة الصحيحة:** {item['answer']}")
                st.write(f"💡 **خطوات الحل:** {item['solution_ar' if is_arabic else 'solution_en']}")
    else:
        st.warning("يرجى إكمال التقييم التشخيصي أولاً لارضاء ملخص نتائجك!")


# =========================================================
# PAGE 4: ERROR ANALYSIS
# =========================================================

elif st.session_state.page == "Error Analysis":
    st.title("🔎 تحليل الأخطاء الذكي (Decision Tree Classifier)")
    st.write("تقوم خوارزميات الذكاء الاصطناعي بتصنيف نوع خطئك لتوجيهك بشكل أفضل.")

    if clf is not None:
        sample_concept = st.selectbox("اختر المفهوم الرياضي للتحليل:", df["concept"].unique())
        sample_skill = st.selectbox("اختر المهارة الفرعية:", df[df["concept"] == sample_concept]["skill"].unique())
        sample_correct = st.radio("نتيجة المحاولة:", ["صحيحة (1)", "خاطئة (0)"])
        
        if st.button("تحليل نوع الخطأ المتوقع 🧠"):
            input_data = pd.DataFrame([{
                "concept": sample_concept,
                "skill": sample_skill,
                "correct": 1 if "1" in sample_correct else 0
            }])
            input_encoded = pd.get_dummies(input_data).reindex(columns=feature_columns, fill_value=0)
            pred_error = clf.predict(input_encoded)[0]
            
            st.info(f"🎯 **التشخيص:** نمط الخطأ المتوقع هو: **{pred_error}**")
            st.markdown("💡 **توجيه معلم نبض:** يُفضل مراجعة خطوات النقل وتغيير الإشارات الرياضية بدقة.")
    else:
        st.info("جارٍ تحميل البيانات لتدريب نموذج الأخطاء...")


# =========================================================
# PAGE 5: LEARNING PATH
# =========================================================

elif st.session_state.page == "Learning Path":
    st.title("🛤️ مسار التعلم المخصص")
    st.write("بناءً على نتائجك، إليك الخطة المقترحة لتحسين أدائك:")

    st.markdown("""
        <div class="custom-card">
            <h3>📍 الخطوة الأولى: التأسيس في المعادلة الجبرية</h3>
            <p>مراجعة مفهوم النقل بتبديل الإشارة وكيفية التخلص من المعاملات.</p>
        </div>
        <div class="custom-card">
            <h3>📍 الخطوة الثانية: تمارين السرعة في الحساب الذهني</h3>
            <p>التدرب على النسب المئوية السريعة ومضاعفات الأرقام.</p>
        </div>
        <div class="custom-card">
            <h3>📍 الخطوة الثالثة: خوض تحديات الفعاليات والإعادة</h3>
            <p>التأكد من رفع النسبة لـ 90%+ عند إعادة التقييم.</p>
        </div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE 6: SMART PRACTICE
# =========================================================

elif st.session_state.page == "Smart Practice":
    st.title("🎯 التدريب الذكي الموجه")
    st.write("تدرب برتم هادئ ومريح دون توقيت زمني:")

    p_q = QUESTIONS[0]
    st.markdown(f"### {p_q['question_ar']}")
    user_p = st.radio("اختر الإجابة:", p_q["options"], key="pract_1")
    
    if st.button("تحقق من الحل"):
        if user_p == p_q["answer"]:
            st.success("إجابة صحيحة وممتازة! ✨")
        else:
            st.error(f"حاول ثانية! الإجابة الصحيحة هي: {p_q['answer']}")


# =========================================================
# PAGE 7: ACTIVITIES & GAMES
# =========================================================

elif st.session_state.page == "Activities":
    st.title("🎮 الفعاليات والألعاب التفاعلية")
    st.write("اختر نوع الفعالية وابدأ التحدي لجمع النقاط والحفاظ على الـ Streak!")

    tab1, tab2, tab3 = st.tabs(["⚡ تحدي الـ 60 ثانية", "🕵️ كاشف الأخطاء", "📅 سؤال اليوم"])

    # 1. تحدي الـ 60 ثانية
    with tab1:
        st.subheader("⚡ تحدي السرعة (Speed Drill)")
        st.caption("أجب على أكبر عدد من الأسئلة في دقيقة واحدة!")
        
        if st.button("ابدأ التحدي الان ⏱️", key="start_speed"):
            st.session_state.timer_start = time.time()
            st.session_state.speed_score = 0
            st.session_state.speed_index = 0

        if st.session_state.timer_start:
            elapsed = time.time() - st.session_state.timer_start
            time_left = max(0, 60 - int(elapsed))
            
            st.metric("الوقت المتبقي", f"⏱️ {time_left} ثانية")
            
            if time_left > 0:
                sq = QUESTIONS[st.session_state.speed_index % len(QUESTIONS)]
                st.write(f"**السؤال:** {sq['question_ar']}")
                user_a = st.radio("إجابتك:", sq["options"], key=f"sp_{st.session_state.speed_index}")
                
                if st.button("تأكيد الإجابة 🚀", key=f"btn_sp_{st.session_state.speed_index}"):
                    if user_a == sq["answer"]:
                        st.session_state.speed_score += 10
                        st.toast("إجابة صحيحة! +10 نقاط", icon="🎉")
                    st.session_state.speed_index += 1
                    st.rerun()
            else:
                st.success(f"🏆 انتهى الوقت! جمعت {st.session_state.speed_score} نقطة!")
                play_audio_and_effects(st.session_state.speed_score)

    # 2. كاشف الأخطاء
    with tab2:
        st.subheader("🕵️ لعبة كاشف الأخطاء (Error Detective)")
        st.info("**المعادلة:** ٣س + ٥ = ٢٠")
        st.write("• **الخطوة ١:** ٣س = ٢٠ + ٥")
        st.write("• **الخطوة ٢:** ٣س = ٢٥")
        st.write("• **الخطوة ٣:** س = ٢٥ ÷ ٣")

        choice = st.radio("ما هي الخطوة التي تحتوي على الخطأ؟", ["الخطوة ١", "الخطوة ٢", "الخطوة ٣"])
        if st.button("تحقق من الإجابة المحققة 🔍"):
            if choice == "الخطوة ١":
                st.balloons()
                st.success("🎉 أحسنت يا محقق! نقل ٥ للطرف الآخر يجب أن يكون بالطرح وليس الجمع.")
            else:
                st.error("❌ حاول ثانية! ركز في علامة النقل في الخطوة الأولى.")

    # 3. سؤال اليوم
    with tab3:
        st.subheader("📅 سؤال اليوم للـ Streak")
        st.write("ما هو نصف العدد $2^{10}$ ؟")
        ans_day = st.radio("اختر إجابتك الصحيحة:", ["2^5", "2^9", "1^10", "2^8"])
        if st.button("إرسال إجابة اليوم 🌟"):
            if ans_day == "2^9":
                st.success("إجابة صحيحة! حافظت على سلسلة الاستمرار 🔥 (+1 يوم)")
            else:
                st.error("حاول مرة أخرى بكره!")


# =========================================================
# PAGE 8: AI QUESTIONS GENERATOR
# =========================================================

elif st.session_state.page == "AI Questions":
    st.title("🤖 مولّد الأسئلة التكيّفي (Gemini AI)")
    
    if client:
        selected_skill = st.selectbox("اختر المهارة:", ["Linear Equations", "Percentages", "Multiplication"])
        selected_diff = st.select_slider("مستوى الصعوبة:", ["Easy", "Medium", "Hard"])
        q_count = st.number_input("عدد الأسئلة:", min_value=1, max_value=5, value=2)

        if st.button("توليد أسئلة جديدة بالذكاء الاصطناعي 🪄"):
            with st.spinner("جارٍ التواصل مع Gemini لتوليد الأسئلة..."):
                try:
                    generated = generate_ai_questions(selected_skill, selected_diff, q_count, st.session_state.lang)
                    st.success("تم توليد الأسئلة بنجاح!")
                    for i, q_item in enumerate(generated):
                        st.markdown(f"**س {i+1}: {q_item['question']}**")
                        st.write(f"الخيارات: {', '.join(q_item['options'])}")
                        st.caption(f"💡 الحل: {q_item['solution']}")
                except Exception as e:
                    st.error(f"حدث خطأ أثناء التوليد: {e}")
    else:
        st.warning("يرجى إضافة مفتاح GEMINI_API_KEY في ملف `.streamlit/secrets.toml` لتفعيل هذه الميزة.")


# =========================================================
# PAGE 9: REASSESSMENT
# =========================================================

elif st.session_state.page == "Reassessment":
    st.title("🔄 إعادة التقييم (قياس الأثر والتقدم)")
    
    if not st.session_state.reassess_submitted:
        q_list = st.session_state.reassess_questions
        if not q_list:
            st.session_state.reassess_questions = random.sample(QUESTIONS, len(QUESTIONS))
            q_list = st.session_state.reassess_questions

        idx = st.session_state.reassess_index
        q = q_list[idx]

        st.subheader(f"سؤال إعادة التقييم {idx + 1}: {q['question_ar' if is_arabic else 'question_en']}")
        ans = st.radio("اختر إجابتك:", q["options"], key=f"rq_ans_{q['id']}")
        st.session_state.reassess_answers[q["id"]] = ans

        if idx < len(q_list) - 1:
            if st.button("السؤال التالي"):
                st.session_state.reassess_index += 1
                st.rerun()
        else:
            if st.button("إنهاء إعادة التقييم 🏆"):
                correct = sum(1 for item in q_list if st.session_state.reassess_answers.get(item["id"]) == item["answer"])
                st.session_state.after_score = round((correct / len(q_list)) * 100)
                st.session_state.reassess_submitted = True
                st.rerun()

    else:
        before = st.session_state.before_score or 0
        after = st.session_state.after_score
        
        play_audio_and_effects(after)
        st.balloons()
        
        st.success(f"🎉 أكملت إعادة التقييم بنجاح! درجتك الجديدة: {after}%")
        
        c1, c2 = st.columns(2)
        c1.metric("الدرجة السابقة", f"{before}%")
        c2.metric("الدرجة الجديدة", f"{after}%", delta=f"{after - before}%")


# =========================================================
# FOOTER (حقوق الملكية المحدثة)
# =========================================================

st.markdown("""
    <hr>
    <div style="text-align: center; color: #94a3b8; font-size: 14px; padding: 10px 0;">
        © 2026 MINNA MOHAMMED — NABD Educational Platform. All rights reserved.
    </div>
""", unsafe_allow_html=True)
