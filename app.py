import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf

# Set page config for a wide layout
st.set_page_config(layout="wide")

st.title("Wage Discrimination Analysis")

# Main Top-Level Tabs
tab_eda, tab_q1, tab_q2, tab_q3, tab_q4 = st.tabs(["EDA", "Q1", "Q2", "Q3", "Q4"])

# --- EDA TAB ---
with tab_eda:
    st.markdown("""
    <style>
    [data-testid="stTable"] {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
    }
    [data-testid="stTable"] table {
        color: black;
        border: 1px solid black;
    }
    [data-testid="stTable"] th, [data-testid="stTable"] td {
        color: black !important;
        border: 1px solid black !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("EDA")
    st.write("In this dataset, there were a total of 19,792 faculty members, with most (15,866) of them being male and 3926 of them being female. When looking at the breakdown of academic fields across genders in the dataset, men make up a larger proportion of faculty members across all three different field types (Arts, Other, and Professional). The most drastic difference in proportions is in the Professional category (Table 1), where men make up 89.8% percent of all members compared to women making up 10.2%, while the least drastic difference is the Arts category, where men make up 71.8% of the members and women make up 28.2%. These are explained by segmenting the academic fields by gender.")
    st.write("Looking at Figure 1a and 1b, the proportion of men who are Professional faculty members more than doubles that of the proportion of women (21.6% for men, 9.9% for women), while a greater proportion of women are Arts faculty members (20.4%) relative to the proportion of men (12.8%). For both genders, faculty members belonging to the Other field made up the largest proportion, with the women having a slightly higher proportion than the men (69.7% relative to 65.6%).")
    
    # Pie chart settings
    pie_colors = ['#ADD8E6', '#FF6B6B', '#90EE90']  # light blue, red, green
    labels = ['Arts', 'Prof', 'Other']
    
    col_table, col_pie1, col_pie2 = st.columns(3)
    
    with col_table:
        table1_data = {
            "": ["Male", "Female"],
            "Arts": ["2,038 (71.8%)", "802 (28.2%)"],
            "Other": ["10,408 (79.2%)", "2,735 (20.8%)"],
            "Professional": ["3,420 (89.8%)", "389 (10.2%)"]
        }
        table1_df = pd.DataFrame(table1_data)
        st.table(table1_df.set_index(""))
        st.caption("Table 1: Gender distribution across academic fields")
    
    with col_pie1:
        fig1, ax1 = plt.subplots()
        sizes_male = [12.8, 21.6, 65.6]
        ax1.pie(sizes_male, labels=[f'{l}({s}%)' for l, s in zip(labels, sizes_male)], 
                colors=pie_colors, startangle=90)
        ax1.set_title('Male')
        st.pyplot(fig1)
        st.caption("Figure 1a: Male faculty by field")
    
    with col_pie2:
        fig2, ax2 = plt.subplots()
        sizes_female = [20.4, 9.9, 69.7]
        ax2.pie(sizes_female, labels=[f'{l}({s}%)' for l, s in zip(labels, sizes_female)], 
                colors=pie_colors, startangle=90)
        ax2.set_title('Female')
        st.pyplot(fig2)
        st.caption("Figure 1b: Female faculty by field")

    st.write("When looking at the distribution of years for when the degree was earned (where values take on the year a degree was earned in the 20th century), female faculty members tended to earn their degrees later than male faculty members, evidenced by the summary statistics between genders. Table 2 shows that the minimum and interquartile range values representing the year a degree was earned were all lower for men relative to women. The minimum, 25th percentile, median, and 75th percentile values were 1948, 1966, 1970, and 1976 for men, and 1954, 1971, 1976 and 1983 for women. Only the maximum value when a degree was obtained was larger for men compared to women, likely due to there being many more male faculty members.")

    col_table2, col_fig2 = st.columns(2)
    
    with col_table2:
        table2_data = {
            "": ["All", "Male", "Female"],
            "Min": [48, 48, 54],
            "Max": [96, 96, 95],
            "1st Quar.": [67, 66, 71],
            "3rd Quar.": [78, 76, 83],
            "Median": [72, 70, 76],
            "Mean": [72.1, 71, 76.6]
        }
        table2_df = pd.DataFrame(table2_data)
        st.table(table2_df.set_index(""))
        st.caption("Table 2: Summary statistics of degree year by gender")
    
    with col_fig2:
        st.image("eda_figure2.png")
        st.caption("Figure 2: Distribution of Year Degree by Gender")

    st.write("To build on this, there is a strong pattern suggesting that there may be a correlation between the year of highest degree attained and academic rank. The data indicates that the female faculty members mostly achieved their degrees later in the 20th century as compared to their male counterparts. The median year for degree attainment for women was 1976 compared to 1970 for men (Table 2). This difference suggests that the female faculty generally have fewer years of post degree professional experience than the male faculty. This difference in experience aligns with the distribution in academic ranks.")
    
    st.write("Men hold a significant majority in the highest academic rank, representing 85.1% of Full Professors, while women make up only 14.9% (Table 3). This disparity is less severe at the Associate Professor level (68.4% male vs. 31.6% female) but women are more represented at the entry-level Assistant Professor rank, accounting for 46%. This can also be seen visually in the bar graph in Figure 3.")

    col_table3, col_fig3 = st.columns(2)
    
    with col_table3:
        table3_data = {
            "": ["Male", "Female"],
            "Assist": ["170 (54%)", "145 (46%)"],
            "Associate": ["299 (68.4%)", "138 (31.6%)"],
            "Full": ["719 (85.1%)", "126 (14.9%)"]
        }
        table3_df = pd.DataFrame(table3_data)
        st.table(table3_df.set_index(""))
        st.caption("Table 3: Academic rank distribution by gender")

    with col_fig3:
        fig3, ax3 = plt.subplots()
        ranks = ['Assist', 'Assoc', 'Full']
        male_counts = [170, 299, 719]
        female_counts = [145, 138, 126]
        
        ax3.bar(ranks, male_counts, label='Male', color='#ADD8E6')
        ax3.bar(ranks, female_counts, bottom=male_counts, label='Female', color='#E57373')
        ax3.set_ylabel('Count')
        ax3.set_title('rank Distribution: All vs Male')
        ax3.legend(loc='upper left')
        ax3.set_ylim(0, 850)
        st.pyplot(fig3)
        st.caption("Figure 3: Academic rank distribution by gender")
    
    st.write("The data shows that female faculty members generally have a later year of highest degree attainment and are significantly underrepresented in the higher academic ranks (e.g., Full Professor) compared to their male counterparts. This pattern suggests an inverse correlation where a later degree year (and implicitly, less experience) is associated with a lower academic rank, which contributes to the observed lower representation of women in senior faculty positions in 1995.")

# --- HELPER FUNCTIONs - put results to be displayed here
def render_question_tab1(label):
    st.subheader(label)
    st.write(f"**Results Summary for {label}** ...")
    
    # Nested Tabs: Uncontrolled vs Controlled
    sub_tab1, sub_tab2 = st.tabs(["Uncontrolled", "Controlled"])
    
    with sub_tab1:
        st.markdown("## T-Test")
        
        # Initialize session state for T-test block
        if 'ttest_state' not in st.session_state:
            st.session_state.ttest_state = 0
        
        # Assumptions data
        assumptions = [
            ("Independence", "After matching the start year on the current year, all observations are independent of each other."),
            ("Normality", "With a sample size of over 19,000 faculty members, the distribution of the data is approximately normal, and the t-test is robust to violations of normality."),
            ("Homogeneity of Variance", "A Welch's t-test is used to account for the difference in variances between the two groups so homogeneity of variance is not a concern."),
            ("Random Sampling", "The data is all faculty so random sampling is not a concern.")
        ]
        
        # T-test interactive block
        if st.session_state.ttest_state == 0:
            # Initial state - Click to explore
            st.markdown('''
            <div style="background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 6px; padding: 1rem; min-height: 280px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            </div>
            ''', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Click here to explore T-test", key="ttest_explore", use_container_width=True):
                    st.session_state.ttest_state = 1
                    st.rerun()
        elif st.session_state.ttest_state == 1:
            # Goal state
            st.markdown('''
            <div style="background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 6px; padding: 1rem; min-height: 280px;">
                <h4 style="color: black; margin-top: 0;">Goal</h4>
                <p style="color: black; font-size: 1.25em;">Shuffle the ‘sex’ label and observe how extreme our observed difference will be to random labeling. If the difference is extreme, we can reject the null hypothesis that the average salaries are the same between males and females.</p>
                <div style="height: 50px;"></div>
            </div>
            ''', unsafe_allow_html=True)
            col_spacer, col_btn = st.columns([8, 1])
            with col_btn:
                if st.button("Next", key="ttest_goal_next"):
                    st.session_state.ttest_state = 2
                    st.rerun()
        elif 2 <= st.session_state.ttest_state <= 5:
            # Assumptions states (2-5)
            current_assumption_idx = st.session_state.ttest_state - 2
            assumptions_html = '<h4 style="color: black; margin-top: 0;">Check Assumptions</h4>'
            for i in range(current_assumption_idx + 1):
                name, desc = assumptions[i]
                assumptions_html += f'<p style="color: black;"><span style="font-weight: bold; font-size: 1.1em;">{name}:</span> {desc}</p>'
            
            st.markdown(f'''
            <div style="background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 6px; padding: 1rem; min-height: 280px; position: relative;">
                {assumptions_html}
                <div style="height: 50px;"></div>
            </div>
            ''', unsafe_allow_html=True)
            col_spacer, col_btn = st.columns([8, 1])
            with col_btn:
                if st.button("Next", key=f"ttest_next_{st.session_state.ttest_state}"):
                    st.session_state.ttest_state += 1
                    st.rerun()
        else:
            # Results state
            st.markdown('''
            <div style="background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 6px; padding: 1rem; min-height: 280px; display: flex; justify-content: space-around; align-items: center;">
                <div>
                    <h4 style="color: black; margin-top: 0;">Results</h4>
                    <ul style="color: black; font-size: 1.1em;">
                        <li>Test Statistic = -12.00</li>
                        <li>p-value = 0.00</li>
                        <li>95% CI: [218.0, 612.4]</li>
                    </ul>
                </div>
                <div style="display: flex; align-items: center; max-width: 350px; text-align: center;">
                    <p style="color: black; font-weight: bold; font-size: 1.3em;">Difference in average salaries are significant without controlling other variables.</p>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            col_spacer, col_btn = st.columns([6, 2])
            with col_btn:
                st.markdown('''
                <a href="#permutation-test-section" onclick="document.getElementById('permutation-test-section').scrollIntoView({behavior: 'smooth'}); return false;" style="text-decoration: none;">
                    <button style="padding: 0.5rem 1rem; cursor: pointer; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px; color: black;">Explore Permutation Test</button>
                </a>
                ''', unsafe_allow_html=True)

        st.markdown('<div id="permutation-test-section" style="padding-top: 10px;"></div>', unsafe_allow_html=True)
        st.markdown("## Permutation Test")
        
        # Initialize session state for Permutation Test block
        if 'perm_state' not in st.session_state:
            st.session_state.perm_state = 0
        
        # Permutation Test interactive block
        if st.session_state.perm_state == 0:
            # Initial state - Click to explore
            st.markdown('''
            <div style="background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 6px; padding: 1rem; min-height: 280px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            </div>
            ''', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Click here to explore Permutation Test", key="perm_explore", use_container_width=True):
                    st.session_state.perm_state = 1
                    st.rerun()
        elif st.session_state.perm_state == 1:
            # Goal state
            st.markdown('''
            <div style="background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 6px; padding: 1rem; min-height: 280px;">
                <h4 style="color: black; margin-top: 0;">Goal</h4>
                <p style="color: black; font-size: 1.25em;">Compare average starting salaries between males and females using a non-parametric approach.</p>
                <h4 style="color: black;">Hypotheses</h4>
                <p style="color: black;">H₀: μ<sub>males</sub> - μ<sub>females</sub> = 0</p>
                <p style="color: black;">Hₐ: μ<sub>males</sub> - μ<sub>females</sub> ≠ 0</p>
                <p style="color: black; font-style: italic;">(If null is true, we should expect our p-value to be large.)</p>
                <div style="height: 20px;"></div>
            </div>
            ''', unsafe_allow_html=True)
            col_spacer, col_btn = st.columns([8, 1])
            with col_btn:
                if st.button("Next", key="perm_goal_next"):
                    st.session_state.perm_state = 2
                    st.rerun()
        else:
            # Results state
            perm_col_left, perm_col_right = st.columns(2)
            with perm_col_left:
                st.markdown('''
                <div style="background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 6px; padding: 1rem; min-height: 280px;">
                    <h4 style="color: black; margin-top: 0;">Results</h4>
                    <ul style="color: black; font-size: 1.1em;">
                        <li>p-value = 0.002</li>
                    </ul>
                    <p style="color: black; font-weight: bold; font-size: 1.1em;">At the 5% significance level, we can reject the null hypothesis that the average salaries are the same between males and females.</p>
                </div>
                ''', unsafe_allow_html=True)
            with perm_col_right:
                st.image("perm_test.png")

    with sub_tab2:
        # Load same-year model once for Wald column and MLR Base / Interaction tabs
        salary_path = Path(__file__).parent / "salary.txt"
        mls_model_same_year = None
        mls_model_same_year_interact = None
        robust_wald_same_year = None
        if salary_path.exists():
            try:
                df = pd.read_csv(salary_path, sep=r"\s+")
                df_same_salary = df[df["startyr"] == df["year"]]
                mls_model_same_year = smf.ols(
                    formula="salary ~ sex + deg + yrdeg + field + startyr + year + rank + admin",
                    data=df_same_salary,
                ).fit(cov_type="HC3")
                robust_wald_same_year = mls_model_same_year.wald_test_terms()
                mls_model_same_year_interact = smf.ols(
                    formula="salary ~ sex + sex * deg + sex * yrdeg + sex * field + sex * year + sex * rank + sex * admin",
                    data=df_same_salary,
                ).fit(cov_type="HC3")
            except Exception:
                pass

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
                "Same-year salary model: OLS of **salary** on sex, deg, yrdeg, field, startyr, year, rank, admin "
                "restricted to rows where `startyr == year` (first year at institution). "
                "Wald test of all terms uses **HC3** robust standard errors."
            )
            st.caption("Results")
            if not salary_path.exists():
                st.warning("Data file `salary.txt` not found. Add it to the project directory to run the Wald test.")
                st.code(
                    "df = pd.read_csv('salary.txt', sep=r'\\s+')\n"
                    "df_same_salary = df[df['startyr'] == df['year']]\n"
                    "mls_model_same_year = smf.ols(formula='salary ~ sex + deg + yrdeg + field + startyr + '\n"
                    "    'year + rank + admin', data=df_same_salary).fit(cov_type='HC3')\n"
                    "robust_wald_same_year = mls_model_same_year.wald_test_terms()",
                    language="python",
                )
            elif robust_wald_same_year is None:
                st.error("Error fitting model. Check data and formula.")
            else:
                try:
                    wald_df = getattr(robust_wald_same_year, "result_frame", None) or getattr(
                        robust_wald_same_year, "table", None
                    )
                    if wald_df is not None and hasattr(wald_df, "to_html"):
                        wald_df_display = wald_df.astype(str)
                        wald_tab_left, wald_tab_right = st.columns([1, 1])
                        with wald_tab_left:
                            st.dataframe(wald_df_display, use_container_width=True)
                            st.markdown("**Download results**")
                            csv_bytes = wald_df_display.to_csv(index=True).encode("utf-8")
                            st.download_button(
                                "Download Wald test table (CSV)",
                                data=csv_bytes,
                                file_name="wald_test_results.csv",
                                mime="text/csv",
                                key="wald_download",
                            )
                        with wald_tab_right:
                            # Infer statistic and pvalue columns (statsmodels naming varies)
                            stat_col = None
                            pval_col = None
                            for c in wald_df.columns:
                                c_lower = str(c).lower()
                                if stat_col is None and ("statistic" in c_lower or c_lower in ("f", "chi2", "wald")):
                                    stat_col = c
                                if pval_col is None and ("pvalue" in c_lower or "p>" in c_lower or "pr>" in c_lower):
                                    pval_col = c
                            if stat_col is None and len(wald_df.columns) >= 1:
                                stat_col = wald_df.select_dtypes(include=[np.number]).columns[0]
                            if pval_col is None and len(wald_df.columns) >= 2:
                                num_cols = wald_df.select_dtypes(include=[np.number]).columns.tolist()
                                pval_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]

                            def _bar_colors(labels, navy="#000080", red="#b22222"):
                                return [red if "sex" in str(lb).lower() else navy for lb in labels]

                            if stat_col is not None:
                                s = wald_df[stat_col].copy()
                                if hasattr(s, "values") and s.dtype == object:
                                    s = pd.to_numeric(s.astype(str).str.replace(r"[^\d.e\-]", "", regex=True), errors="coerce")
                                s = s.dropna().sort_values(ascending=False)
                                if len(s) > 0:
                                    fig_stat = go.Figure(go.Bar(
                                        x=s.values, y=s.index.astype(str),
                                        orientation="h",
                                        marker_color=_bar_colors(s.index),
                                    ))
                                    fig_stat.update_layout(
                                        title="Statistic by factor",
                                        xaxis_title=stat_col,
                                        yaxis_title="",
                                        margin=dict(l=10, r=10, t=40, b=40),
                                        height=280,
                                        showlegend=False,
                                    )
                                    fig_stat.update_yaxes(autorange="reversed")
                                    st.plotly_chart(fig_stat, use_container_width=True, key="wald_stat_bar")
                            if pval_col is not None:
                                p = wald_df[pval_col].copy()
                                if hasattr(p, "values") and p.dtype == object:
                                    p = pd.to_numeric(p.astype(str).str.replace(r"[^\d.e\-]", "", regex=True), errors="coerce")
                                p = p.dropna().sort_values(ascending=False)
                                if len(p) > 0:
                                    fig_pval = go.Figure(go.Bar(
                                        x=p.values, y=p.index.astype(str),
                                        orientation="h",
                                        marker_color=_bar_colors(p.index),
                                    ))
                                    fig_pval.update_layout(
                                        title="P-value by factor",
                                        xaxis_title=pval_col,
                                        yaxis_title="",
                                        margin=dict(l=10, r=10, t=40, b=40),
                                        height=280,
                                        showlegend=False,
                                    )
                                    fig_pval.update_yaxes(autorange="reversed")
                                    st.plotly_chart(fig_pval, use_container_width=True, key="wald_pval_bar")
                    else:
                        st.text(str(robust_wald_same_year))
                except Exception:
                    st.text(str(robust_wald_same_year))

        st.subheader("MLR Test")
        mlr_base_tab, mlr_interaction_tab = st.tabs(["Base", "Interaction"])

        with mlr_base_tab:
            st.caption("Description")
            st.markdown(
                '<div style="min-height: 80px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa; margin-bottom: 1rem;"></div>',
                unsafe_allow_html=True,
            )
            st.caption("Results")
            if "mlr_base_run" not in st.session_state:
                st.session_state.mlr_base_run = False
            st.latex(
                r"\widehat{\text{salary}} = \beta_0 + \beta_1 \text{sex} + \beta_2 \text{deg} + \beta_3 \text{yrdeg} "
                r"+ \beta_4 \text{field} + \beta_5 \text{startyr} + \beta_6 \text{year} + \beta_7 \text{rank} "
                r"+ \beta_8 \text{admin} + \epsilon"
            )
            mlr_left, mlr_right = st.columns([1, 1])
            with mlr_left:
                if mls_model_same_year is not None:
                    run_clicked = st.button("Run MLR", key="mlr_base_btn")
                    if not st.session_state.mlr_base_run:
                        st.info("Click **Run MLR** (left) to run the regression and see results here.")
                    hide_clicked = st.button("Hide Regression Results", key="mlr_base_hide_btn")
                    if run_clicked:
                        st.session_state.mlr_base_run = True
                    if hide_clicked:
                        st.session_state.mlr_base_run = False
                    st.caption("Same-year OLS (HC3). Show or hide results on the right.")
                else:
                    st.caption("Add `salary.txt` to run the model.")
            with mlr_right:
                if st.session_state.mlr_base_run and mls_model_same_year is not None:
                    st.text(str(mls_model_same_year.summary()))

        with mlr_interaction_tab:
            st.caption("Description")
            st.markdown(
                '<div style="min-height: 80px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa; margin-bottom: 1rem;"></div>',
                unsafe_allow_html=True,
            )
            st.caption("Results")
            if "mlr_interaction_run" not in st.session_state:
                st.session_state.mlr_interaction_run = False
            st.latex(
                r"\widehat{\text{salary}} = \beta_0 + \beta_1 \text{sex} + \beta_2 \text{deg} + \beta_3 \text{yrdeg} "
                r"+ \beta_4 \text{field} + \beta_5 \text{year} + \beta_6 \text{rank} + \beta_7 \text{admin}"
            )
            st.latex(
                r"\quad {} + \beta_8 (\text{sex} \times \text{deg}) + \beta_9 (\text{sex} \times \text{yrdeg}) "
                r"+ \beta_{10} (\text{sex} \times \text{field}) + \beta_{11} (\text{sex} \times \text{year}) "
                r"+ \beta_{12} (\text{sex} \times \text{rank}) + \beta_{13} (\text{sex} \times \text{admin}) + \epsilon"
            )
            mlr_int_left, mlr_int_right = st.columns([1, 1])
            with mlr_int_left:
                if mls_model_same_year_interact is not None:
                    run_int_clicked = st.button("Run MLR", key="mlr_interaction_btn")
                    if not st.session_state.mlr_interaction_run:
                        st.info("Click **Run MLR** (left) to run the regression and see results here.")
                    hide_int_clicked = st.button("Hide Regression Results", key="mlr_interaction_hide_btn")
                    if run_int_clicked:
                        st.session_state.mlr_interaction_run = True
                    if hide_int_clicked:
                        st.session_state.mlr_interaction_run = False
                    st.caption("Same-year OLS with sex interactions (HC3). Show or hide results on the right.")
                else:
                    st.caption("Add `salary.txt` to run the model.")
            with mlr_int_right:
                if st.session_state.mlr_interaction_run and mls_model_same_year_interact is not None:
                    st.text(str(mls_model_same_year_interact.summary()))

# Variable tooltips for Q2 MLR (salary jump analysis)
Q2_VAR_TOOLTIPS = {
    "Intercept": "Baseline expected salary jump when all predictors are at reference level.",
    "sexM": "Sex: Male vs. Female (reference). Interpretation: Difference in salary jump for males vs. females, holding other covariates constant.",
    "degPhD": "Degree: PhD vs. reference. Interpretation: Difference in salary jump for PhD holders vs. reference degree.",
    "degProf": "Degree: Professional vs. reference. Interpretation: Difference in salary jump for professional degree holders.",
    "yrdeg": "Year the professor received their degree. Interpretation: Change in salary jump per additional year of degree.",
    "field0other": "Field: Other vs. reference. Interpretation: Field-specific effect on salary jump.",
    "fieldProf": "Field: Prof vs. reference. Interpretation: Field-specific effect on salary jump.",
    "startyr": "Year the professor started at the institution. Interpretation: Change in salary jump per year later start date.",
    "salary": "Associate professor salary in the last year before promotion. Interpretation: Effect of baseline salary on promotion jump.",
    "admin": "Administrative appointment (yes/no). Interpretation: Difference in salary jump for professors with vs. without admin duties.",
}

# Variable tooltips for Full Salary model (first-year Full professor salary)
Q2_FULL_VAR_TOOLTIPS = {
    "Intercept": "Baseline expected first-year Full salary when all predictors are at reference level.",
    "sexM": "Sex: Male vs. Female (reference). Interpretation: Difference in first-year Full salary for males vs. females.",
    "degPhD": "Degree: PhD vs. reference.",
    "degProf": "Degree: Professional vs. reference.",
    "yrdeg": "Year the professor received their degree.",
    "startyr": "Year the professor started at the institution.",
    "year": "Year of observation (first year as Full professor).",
    "field0other": "Field: Other vs. reference.",
    "fieldOther": "Field: Other vs. reference.",
    "fieldProf": "Field: Prof vs. reference.",
    "admin": "Administrative appointment (yes/no).",
}


def _q2_coef_row_robust(var_name: str, est: str, se: str, t: str, p: str, ci: str, highlight: bool = False, tooltips: dict = None) -> None:
    """Render a robust SE coefficient row with CI (hover for variable tooltip)."""
    tips = tooltips if tooltips is not None else Q2_VAR_TOOLTIPS
    tip = tips.get(var_name, "").replace('"', "&quot;")
    var_html = f'<span title="{tip}" style="cursor: help; border-bottom: 1px dotted #666;">{var_name} ℹ️</span>' if tip else var_name
    if highlight:
        def _esc(s): return str(s).replace("<", "&lt;").replace(">", "&gt;")
        row_html = (
            f'<div style="background-color: #ffff99; padding: 6px 8px; margin: 2px 0; border-radius: 4px; '
            f'display: flex; align-items: center; gap: 8px;">'
            f'<span style="flex: 1.8;">{var_html}</span>'
            f'<span style="flex: 1;">{_esc(est)}</span><span style="flex: 1;">{_esc(se)}</span>'
            f'<span style="flex: 0.8;">{_esc(t)}</span><span style="flex: 0.8;">{_esc(p)}</span>'
            f'<span style="flex: 1.5;">{_esc(ci)}</span></div>'
        )
        st.markdown(row_html, unsafe_allow_html=True)
        return
    cols = st.columns([1.8, 1, 1, 0.8, 0.8, 1.5])
    cols[0].markdown(var_html, unsafe_allow_html=True)
    cols[1].text(est)
    cols[2].text(se)
    cols[3].text(t)
    cols[4].text(p)
    cols[5].text(ci)


# Map assumption name to plot filename(s) - Salary Jump model
Q2_PLOT_FILES = {
    "Normality": ["normality.png"],
    "Linearity + Homoskedasticity": ["linearity.png"],
    "Outliers": ["outliers.png"],
}

# Map assumption name to plot filename(s) - Full Salary model
Q2_FULL_PLOT_FILES = {
    "Normality": ["normality_full.png"],
    "Linearity + Homoskedasticity": ["linearity_full.png"],
}


def render_question_tab2(label):
    st.subheader(label)
    st.write(f"**Results Summary for {label}** ...")
    sub_tab1, sub_tab2 = st.tabs(["Uncontrolled", "Controlled"])
    
    with sub_tab1:
        st.subheader("T-Test")
        st.caption("Description")
        st.markdown(
            '<div style="min-height: 80px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa; margin-bottom: 1rem;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown("## Results")
        st.markdown(
            '<div style="min-height: 280px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa;"></div>',
            unsafe_allow_html=True,
        )
        
    with sub_tab2:
        st.subheader("Multiple Linear Regression")
        st.markdown("Multiple linear regression helps control for other influential variables (degree, field, experience, salary, etc.), allowing us to measure the significance of sex on salary jump while holding these factors constant. There are two models, one for the salary jump and one for the first-year Full professor salary. Outcomes: salary jump = first-year Full professor salary − last-year Associate professor salary and first-year Full professor salary. Sample: professors promoted from Associate to Full.")
        
        m1_tab, m2_tab = st.tabs(["Salary Jump Model", "Full Salary Model"])

        with m1_tab:
            st.markdown("### Data")
            st.markdown("""
            Sample: Professors promoted from Associate to Full. For each professor, we kept the row corresponding to their last year as Associate, and defined 
            salary jump = (first-year Full salary) − (last-year Associate salary). Covariates are measured at that last Associate year.
            """)
            
            st.markdown("### Final Model")
            st.latex(r"\widehat{\text{salary\_jump}} = \beta_0 + \beta_1 \text{sex} + \beta_2 \text{deg} + \beta_3 \text{yrdeg} + \beta_4 \text{field} + \beta_5 \text{startyr} + \beta_6 \text{salary} + \beta_7 \text{admin} + \epsilon")
            st.caption("Note: `year` was dropped due to high VIF (multicollinearity).")
            
            st.markdown("### Assumption Checks")
            assump_choices = ["Normality", "Linearity + Homoskedasticity", "Outliers"]
            left_col, right_col = st.columns([1, 1])
            with left_col:
                st.markdown("""
                | Check | Method | Result |
                |-------|--------|--------|
                | Multicollinearity | VIF (car) | `year` had highest VIF → removed. Remaining predictors acceptable. |
                | Linearity | Residuals vs. fitted | Residuals centered near zero; linearity mostly reasonable. |
                | Homoskedasticity | Residuals vs. fitted | Spread increased with fitted values → heteroskedasticity. |
                | Normality | Q-Q plot | Right tail deviated; large sample supports approximate normality. |
                | Outliers | Cook's distance (4/n) | Top 6 identified; kept full sample (R² worse without). |
                """)
                selected = st.radio("Select assumption to view diagnostic:", assump_choices, key="salary_jump_assump", horizontal=True)
                st.caption("Diagnostic plots are shown for the base model, which includes the year variable. The outlier plot was created after fitting the final model.")
            with right_col:
                plot_files = Q2_PLOT_FILES[selected]
                plots_dir = Path(__file__).parent / "q2_plots"
                shown = 0
                for f in plot_files:
                    if (plots_dir / f).exists():
                        st.image(str(plots_dir / f), use_column_width=True)
                        shown += 1
                if shown == 0:
                    st.info("Add diagnostic plots to `q2_plots/` (linearity.png, homoskedasticity.png, normality.png, outliers.png).")
            
            st.markdown("### Results (Robust SE, HC1)")
            st.caption("Standard errors adjusted for heteroskedasticity. Significance: *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05, . p&lt;0.1")
            cols = st.columns([1.8, 1, 1, 0.8, 0.8, 1.5])
            for i, c in enumerate(["Variable", "Estimate", "Robust SE", "t", "p", "95% CI"]):
                cols[i].markdown(c)
            st.divider()
            _q2_coef_row_robust("Intercept", "-741.76", "267.11", "-2.78", "0.006**", "(-1265, -218)")
            _q2_coef_row_robust("sexM", "-31.04", "40.17", "-0.77", "0.440", "(-110, 48)", highlight=True)
            _q2_coef_row_robust("degPhD", "6.75", "47.20", "0.14", "0.886", "(-86, 99)")
            _q2_coef_row_robust("degProf", "-17.18", "72.57", "-0.24", "0.813", "(-159, 125)")
            _q2_coef_row_robust("yrdeg", "13.92", "4.28", "3.25", "0.001**", "(5.5, 22.3)")
            _q2_coef_row_robust("field0other", "-95.94", "52.04", "-1.84", "0.066.", "(-198, 6.1)")
            _q2_coef_row_robust("fieldProf", "6.70", "66.41", "0.10", "0.920", "(-123, 137)")
            _q2_coef_row_robust("startyr", "3.72", "3.77", "0.99", "0.324", "(-3.7, 11.1)")
            _q2_coef_row_robust("salary", "-0.0017", "0.0165", "-0.10", "0.917", "(-0.034, 0.031)")
            _q2_coef_row_robust("admin", "39.89", "51.60", "0.77", "0.440", "(-61, 141)")
            st.caption("Hover over ℹ️ for variable definitions.")

        with m2_tab:
            st.markdown("### Data")
            st.markdown("""
            The goal is to consider the difference in salary range between males and females within the promoted professors. 
			By modeling the first-year salary as a full professor, we can evaulate the salary jump in a fair manner and determine if there is a wage discrimination between males and females who were promoted.
			This aspect focuses less on the salary jump and more on the final salary of full professors, which addresses Question 3 more than Question 2.
			Sample: Professors promoted from Associate to Full. For each professor, we kept the first year they held the Full rank.
            Outcome: first-year Full professor salary.
            """)
            
            st.markdown("### Final Model")
            st.latex(r"\widehat{\text{promoted\_full\_salary}} = \beta_0 + \beta_1 \text{sex} + \beta_2 \text{deg} + \beta_3 \text{yrdeg} + \beta_4 \text{startyr} + \beta_5 \text{year} + \beta_6 \text{field} + \beta_7 \text{admin} + \epsilon")
            
            st.markdown("### Assumption Checks")
            full_assump_choices = ["Normality", "Linearity + Homoskedasticity"]
            full_left_col, full_right_col = st.columns([1, 1])
            with full_left_col:
                st.markdown("""
                | Check | Method | Result |
                |-------|--------|--------|
                | Multicollinearity | VIF (car) | VIF values were low and no variables needed to be removed. |
                | Linearity | Residuals vs. fitted | Residuals centered near zero; linearity mostly reasonable. |
                | Homoskedasticity | Residuals vs. fitted | There is a funnel shape and robust SE's need to be applied. |
                | Normality | Q-Q plot | Right tail deviated; large sample supports approximate normality. |
                """)
                full_selected = st.radio("Select assumption to view diagnostic:", full_assump_choices, key="full_salary_assump", horizontal=True)
            with full_right_col:
                full_plot_files = Q2_FULL_PLOT_FILES[full_selected]
                full_plots_dir = Path(__file__).parent / "q2_plots"
                full_shown = 0
                for f in full_plot_files:
                    if (full_plots_dir / f).exists():
                        st.image(str(full_plots_dir / f), use_column_width=True)
                        full_shown += 1
                if full_shown == 0:
                    st.info("Add Full Salary diagnostic plots to q2_plots/ (linearity_full.png, normality_full.png).")
            
            st.markdown("### Results (Robust SE, HC1)")
            st.caption("Standard errors adjusted for heteroskedasticity. Significance: *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05, . p&lt;0.1")
            full_cols = st.columns([1.8, 1, 1, 0.8, 0.8, 1.5])
            for i, c in enumerate(["Variable", "Estimate", "Robust SE", "t", "p", "95% CI"]):
                full_cols[i].markdown(c)
            st.divider()
            _q2_coef_row_robust("Intercept", "-15602.89", "539.22", "-28.94", "<0.001***", "(-16660, -14546)", tooltips=Q2_FULL_VAR_TOOLTIPS)
            _q2_coef_row_robust("sexM", "212.72", "77.33", "2.75", "0.006**", "(61, 364)", highlight=True, tooltips=Q2_FULL_VAR_TOOLTIPS)
            _q2_coef_row_robust("degPhD", "-65.89", "103.63", "-0.64", "0.525", "(-269, 137)", tooltips=Q2_FULL_VAR_TOOLTIPS)
            _q2_coef_row_robust("degProf", "299.89", "167.04", "1.80", "0.073.", "(-28, 627)", tooltips=Q2_FULL_VAR_TOOLTIPS)
            _q2_coef_row_robust("yrdeg", "29.40", "9.78", "3.01", "0.003**", "(10, 49)", tooltips=Q2_FULL_VAR_TOOLTIPS)
            _q2_coef_row_robust("startyr", "30.47", "8.56", "3.56", "<0.001***", "(14, 47)", tooltips=Q2_FULL_VAR_TOOLTIPS)
            _q2_coef_row_robust("year", "173.97", "7.69", "22.63", "<0.001***", "(159, 189)", tooltips=Q2_FULL_VAR_TOOLTIPS)
            _q2_coef_row_robust("fieldOther", "292.12", "77.29", "3.78", "<0.001***", "(141, 444)", tooltips=Q2_FULL_VAR_TOOLTIPS)
            _q2_coef_row_robust("fieldProf", "788.42", "96.45", "8.17", "<0.001***", "(599, 977)", tooltips=Q2_FULL_VAR_TOOLTIPS)
            _q2_coef_row_robust("admin", "490.82", "103.42", "4.75", "<0.001***", "(288, 694)", tooltips=Q2_FULL_VAR_TOOLTIPS)
            st.caption("Hover over ℹ️ for variable definitions.")

def render_question_tab3(label):
    st.subheader(label)
    st.write(f"**Results Summary for {label}**:")
    st.markdown(
        """
        - Our models reveal a statistically significant gender-based wage disparity in baseline starting salaries.
        - When isolating the promotion process, we found no evidence of discrimination in the specific salary increases granted when transitioning to Full Professor.
        - However, because the initial wage gap is never corrected, significant salary disparities remain present during a professor's first year at the Full Professor rank.
        """
    )
    
    # Nested Tabs: Uncontrolled vs Controlled
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Propensity Score Matching", "MLR Approach", "ANOVA and Wald Test"])
    
    with sub_tab1:
        st.subheader("Propensity Score Matching")
        psm_text_col, psm_graph_col = st.columns([1.15, 1])
        with psm_text_col:
            st.markdown("#### Methodological Rationale")
            st.markdown(
                "To definitively test for structural wage discrimination and address potential confounding variables, we implemented Propensity Score Matching (PSM). "
                "Standard regression models control for covariates mathematically, but PSM allows us to construct a highly balanced, \"apples-to-apples\" comparison. "
                "By matching female faculty members with male peers who have nearly identical professional profiles, we aimed to isolate the specific effect of gender on salary."
            )
            st.markdown("#### Determining the Propensity Score")
            st.markdown(
                "The propensity score for each observation was calculated using a Logistic Regression model. Rather than predicting salary, this model predicted the probability of a "
                "faculty member being female based on their structural and background characteristics. The predictor variables included years since degree (yrdeg), starting year (startyr), "
                "administrative duties (admin), degree type (deg), academic field (field), and current rank (rank). This single probability score compressed a multi-dimensional professional "
                "profile into one comparable metric."
            )
            st.markdown("#### The Matching Process and Post-Match Analysis")
            st.markdown(
                "Once scores were assigned, female faculty were paired with male faculty who possessed the same (or nearly identical) propensity scores. This matching algorithm essentially "
                "created \"twins\" within the dataset, ensuring that we were comparing individuals with the exact same qualifications and career trajectories. Following the matching process, "
                "we applied a post-matching Ordinary Least Squares (OLS) Multiple Linear Regression on this balanced subset for each year to extract the exact wage penalty associated with gender."
            )
            st.markdown("#### Key Findings")
            st.markdown(
                "The PSM analysis revealed a clear and persistent wage penalty for female faculty. When calculating the wage deficit as a percentage of the matched male peer's salary, "
                "the data showed that identically qualified women consistently earned roughly 5% to 15% less than their direct male counterparts. Notably, across the entire two-decade span "
                "analyzed, this estimated wage gap never dropped to 0%. This demonstrates that the observed salary differences are not merely an artifact of differing qualifications or career "
                "choices (such as field or rank), but rather represent a persistent, structural wage gap."
            )
        with psm_graph_col:
            st.markdown("#### Results")
            st.image("q3_plots/psm_graph.png", use_container_width=True)
            st.markdown("#### Yearly Wage Penalty Results For Females(%)")
            psm_yearly_results = [
                {"year": 76, "gap_pct": 2.556195},
                {"year": 78, "gap_pct": 2.369941},
                {"year": 80, "gap_pct": 4.412997},
                {"year": 82, "gap_pct": 4.820340},
                {"year": 84, "gap_pct": 4.121635},
                {"year": 86, "gap_pct": 8.743591},
                {"year": 88, "gap_pct": 7.668708},
                {"year": 90, "gap_pct": 6.421193},
                {"year": 92, "gap_pct": 7.556835},
                {"year": 94, "gap_pct": 6.439020},
                {"year": 95, "gap_pct": 5.350346}
            ]
            st.dataframe(psm_yearly_results, hide_index=True, use_container_width=True)
        
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