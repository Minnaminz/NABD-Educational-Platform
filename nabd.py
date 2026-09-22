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

        st.html(
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
            ))

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

st.html(
    textwrap.dedent(
    f"""
    <div class="footer">
        {L["footer"]}<br>
        © 2026 MINNA MOHAMMED — NABD Educational Platform.
        All rights reserved.
    </div>
    """
    ))
