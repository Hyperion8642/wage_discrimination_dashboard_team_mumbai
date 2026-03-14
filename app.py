import streamlit as st

# Set page config for a wide layout
st.set_page_config(layout="wide")

st.title("Wage Discrimination Analysis")

# Main Top-Level Tabs
tab_eda, tab_q1, tab_q2, tab_q3, tab_q4 = st.tabs(["EDA", "Q1", "Q2", "Q3", "Q4"])

# --- EDA TAB ---
with tab_eda:
    st.subheader("EDA")
    st.write("EDA desc: This section contains the exploratory data analysis for the wage dataset.")
    
    # Create three columns for visualizations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Viz 1")
        # Insert Plot 1 here
        
    with col2:
        st.subheader("Viz 2")
        # Insert Plot 2 here
        
    with col3:
        st.subheader("Viz 3")
        # Insert Plot 3 here

# --- HELPER FUNCTIONs - put results to be displayed here
def render_question_tab1(label):
    st.subheader(label)
    st.write(f"**Results Summary for {label}** ...")
    
    # Nested Tabs: Uncontrolled vs Controlled
    sub_tab1, sub_tab2 = st.tabs(["Uncontrolled", "Controlled"])
    
    with sub_tab1:
        st.markdown("## T-Test")
        st.markdown(
            '<div style="min-height: 280px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa;"></div>',
            unsafe_allow_html=True,
        )

        st.markdown("## Permutation Test")
        st.markdown(
            '<div style="min-height: 280px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa;"></div>',
            unsafe_allow_html=True,
        )

    with sub_tab2:
        st.subheader("ANOVA and Wald Test")
        anova_col, wald_col = st.columns(2)
        with anova_col:
            st.markdown("## ANOVA")
            st.caption("Description")
            st.markdown(
                '<div style="min-height: 80px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa; margin-bottom: 1rem;"></div>',
                unsafe_allow_html=True,
            )
            st.caption("Results")
            st.markdown(
                '<div style="min-height: 280px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa;"></div>',
                unsafe_allow_html=True,
            )
        with wald_col:
            st.markdown("## Wald Test")
            st.caption("Description")
            st.markdown(
                '<div style="min-height: 80px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa; margin-bottom: 1rem;"></div>',
                unsafe_allow_html=True,
            )
            st.caption("Results")
            st.markdown(
                '<div style="min-height: 280px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa;"></div>',
                unsafe_allow_html=True,
            )

        st.subheader("MLR Test")
        mlr_base_tab, mlr_interaction_tab = st.tabs(["Base", "Interaction"])

        with mlr_base_tab:
            st.caption("Description")
            st.markdown(
                '<div style="min-height: 80px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa; margin-bottom: 1rem;"></div>',
                unsafe_allow_html=True,
            )
            st.caption("Results")
            st.markdown(
                '<div style="min-height: 280px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa;"></div>',
                unsafe_allow_html=True,
            )

        with mlr_interaction_tab:
            st.caption("Description")
            st.markdown(
                '<div style="min-height: 80px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa; margin-bottom: 1rem;"></div>',
                unsafe_allow_html=True,
            )
            st.caption("Results")
            st.markdown(
                '<div style="min-height: 280px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa;"></div>',
                unsafe_allow_html=True,
            )

def render_question_tab2(label):
    st.subheader(label)
    st.write(f"**Results Summary for {label}** ...")
    # Nested Tabs: Uncontrolled vs Controlled
    sub_tab1, sub_tab2 = st.tabs(["Uncontrolled", "Controlled"])
    
    with sub_tab1:
        st.subheader("T-Test")
        st.markdown(
            '<p style="font-size: 1.15rem;">Is there a difference in average salary jump between males and females?</p>',
            unsafe_allow_html=True,
        )

        with st.expander("**Step 1: Prepare**", expanded=True):
            st.markdown("""
            In order to answer our question, we need to prepare the data first. 
            We filtered our dataframe to all staff who were promoted to full time from associate level. 
            \n You can see an example of the filtered dataframe below.""")
            img_col, _ = st.columns([1, 1])
            with img_col:
                st.image("assets/step1_prepared_data.png", use_container_width=True)
            st.markdown("""
            We then calculated the salary difference between the first year they were a full professor, 
            and the last year associate professor. \n You can see an example of the calculated dataframe below.
            """)
            diff_col, _ = st.columns([1, 4])
            with diff_col:
                st.image("assets/step1_salary_diff.png", use_container_width=True)
            st.markdown("""
            In total we have 545 professors that were promoted to full time from associate level.
            """)
        with st.expander("**Step 2: Hypotheses & Test**"):
            st.markdown("""
            Now we can test the hypothesis. We will use a two sample T-Test to test the hypothesis.
            The null hypothesis is that there is no difference in the average salary jump between males and females.
            The alternative hypothesis is that there is a difference in the average salary jump between males and females.
            
            **Two sample T-Test**
            - H₀: μ(males) = μ(females)
            - Hₐ: μ(males) ≠ μ(females)
            - alpha = 0.05
            t.test(salary_diff ~ sex, data = promoted, var.equal = TRUE) 
            \n We are using the var.equal = TRUE argument because we are assuming that the variance is equal between the two groups.
            And since our sample size is large, we can use the t-distribution to approximate the normal distribution.
            """)

        with st.expander("**Step 3: Results**"):
            st.markdown("""
            After running the T-Test, we get the following results:
            | Statistic | Value |
            |-----------|-------|
            | T Statistic | 1.3291 |
            | P-Value | 0.1858 |
            | 95% CI | [-26.21, 133.88] |
            """)
        st.markdown("## Results")
        st.markdown("""
        **Conclusion:** We fail to reject the null hypothesis

        Since our p-value is 0.1858, which is greater than 0.05, we fail to reject the null hypothesis.
        
        So we say that we do not have enough evidence to say that there is a difference in the average salary jump between males and females.

        However, the T-test doesn't account for other factors which is why we need to control for other factors.
        """)

    with sub_tab2:
        st.subheader(f"Controlled Models")
        m1_tab, m2_tab = st.tabs(["Base", "Interaction"])

        with m1_tab:
            st.markdown("## Method 1")
            st.caption("Description")
            st.markdown(
                '<div style="min-height: 80px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa; margin-bottom: 1rem;"></div>',
                unsafe_allow_html=True,
            )
            st.caption("Results")
            st.markdown(
                '<div style="min-height: 280px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa;"></div>',
                unsafe_allow_html=True,
            )

        with m2_tab:
            st.markdown("## Method 2")
            st.caption("Description")
            st.markdown(
                '<div style="min-height: 80px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa; margin-bottom: 1rem;"></div>',
                unsafe_allow_html=True,
            )
            st.caption("Results")
            st.markdown(
                '<div style="min-height: 280px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa;"></div>',
                unsafe_allow_html=True,
            )
        # Add regression results or stats here

def render_question_tab3(label):
    st.subheader(label)
    st.write(f"**Results Summary for {label}** ...")
    
    # Nested Tabs: Uncontrolled vs Controlled
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Propensity Score Matching", "MLR Approach", "ANOVA and Wald Test"])
    
    with sub_tab1:
        st.subheader("Propensity Score Matching")
        st.caption("Description")
        st.markdown(
         '<div style="min-height: 80px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa; margin-bottom: 1rem;"></div>',
            unsafe_allow_html=True,
        )
        st.caption("Results")
        st.markdown(
            '<div style="min-height: 280px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa;"></div>',
            unsafe_allow_html=True,
        )
        # Add regression results or stats here
        
    with sub_tab2:
        st.subheader("MLR Approach")
        st.caption("Description")
        st.markdown(
            '<div style="min-height: 80px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa; margin-bottom: 1rem;"></div>',
            unsafe_allow_html=True,
        )
        st.caption("Results")
        st.markdown(
        '<div style="min-height: 280px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa;"></div>',
                unsafe_allow_html=True,
        )
    with sub_tab3:
        st.subheader("ANOVA and Wald Test")
        anova_col, wald_col = st.columns(2)
        with anova_col:
            st.caption("Description")
            st.markdown(
                '<div style="min-height: 80px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa; margin-bottom: 1rem;"></div>',
                unsafe_allow_html=True,
            )
            st.caption("Results")
            st.markdown(
                '<div style="min-height: 280px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa;"></div>',
                unsafe_allow_html=True,
            )
        with wald_col:
            st.caption("Description")
            st.markdown(
                '<div style="min-height: 80px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa; margin-bottom: 1rem;"></div>',
                unsafe_allow_html=True,
            )
            st.caption("Results")
            st.markdown(
                '<div style="min-height: 280px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa;"></div>',
                unsafe_allow_html=True,
            )
with tab_q1:
    render_question_tab1("Q1")

with tab_q2:
    render_question_tab2("Q2")

with tab_q3:
    render_question_tab3("Q3")

# --- Q4 TAB ---
with tab_q4:
    st.subheader("Q4")
    st.markdown("""
    * **Issue 1**: [Insert details here]
    * **Issue 2**: [Insert details here]
    * **Issue 3**: [Insert details here]
    """)