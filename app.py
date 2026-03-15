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

        st.markdown("**Step 1a: Prepare**")
        st.markdown("""
        In order to answer our question, we need to prepare the data first. 
        We filtered our dataframe to all staff who were promoted to full time from associate level. 
        We then calculated the salary difference between the first year they were a full professor, 
        and the last year associate professor. You can see an example of the data below, 
        where we first filtered the data to only include promoted professors, and then calculated the salary difference for each professor.
        In total we had 545 professors that were promoted to full time from associate level.
        """)
        if "q2_step1_image" not in st.session_state:
            st.session_state["q2_step1_image"] = "promoted"
        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("Promoted Professors Data", key="q2_btn_promoted", use_container_width=True):
                st.session_state["q2_step1_image"] = "promoted"
                st.rerun()
        with btn2:
            if st.button("Salary Difference Data", key="q2_btn_salary_diff", use_container_width=True):
                st.session_state["q2_step1_image"] = "salary_diff"
                st.rerun()
        diff_col, _ = st.columns([3, 1])
        with diff_col:
            if st.session_state["q2_step1_image"] == "promoted":
                st.image("assets/step1_prepared_data.png", use_container_width=True)
            else:
                st.image("assets/step1_salary_diff.png", use_container_width=True)
        st.markdown("**Step 1b: Salary Difference EDA**")
        st.markdown("To get a glance at our data, we plotted the density and boxplot of the salary difference to see the distribution of the salary difference between males and females.")
        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            st.image("assets/salary_diff_density.png", use_container_width=True)
        with plot_col2:
            st.image("assets/salary_diff_boxplot.png", use_container_width=True)
        st.markdown("""At a glance, we can see that the density curves of salary jump between the two genders are relatively similar, 
        and the boxplots show a similar median salary jump, with females having a slightly higher median. However males seem to have a larger range and outliers in the data.""")
        st.markdown("**Step 2: Hypotheses & Test**")
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

        st.markdown("**Step 3: Results**")
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
    st.subheader("Limitations & Considerations")
    st.markdown("Key limitations and caveats when generalizing the wage discrimination analysis:")
    st.divider()

    lim1, lim2 = st.columns(2)
    with lim1:
        with st.expander("**Limited Dataset**", expanded=False):
            st.markdown("The data may not support extrapolation or predictions beyond the observed time range and population. The data is from more than 20 years ago, and may not be representative of the current labor market.")
        with st.expander("**Potential Sampling Bias**", expanded=False):
            st.markdown("The data is only from one institution, and may not be representative of all institutions. This is not be representative of all faculty or institutions.")
        with st.expander("**Gender imbalance**", expanded=False):
            st.markdown("The dataset has far more male than female faculty; estimates for women may be less precise or more sensitive to outliers.")
    with lim2:
        with st.expander("**Correlation doesn’t mean causation**", expanded=False):
            st.markdown("Observed associations (e.g., sex and salary) may be driven by unmeasured factors rather than a direct causal effect.")
        with st.expander("**Potential confounding variables**", expanded=False):
            st.markdown("Variables such as teaching ratings, grants, or publications were not controlled for and could explain part of the observed differences.")
        with st.expander("**Generalizing inflation**", expanded=False):
            st.markdown("Salary levels and gaps may not translate directly to other eras or settings due to inflation and labor market differences.")