import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


def _format_pval_5digits(v):
    """Format a p-value to 5 significant figures (handles decimal and exponential form)."""
    try:
        a = np.asarray(v).flatten()
        if len(a) == 0:
            return v
        x = float(a[0])
        return f"{x:.5g}"
    except (TypeError, ValueError, IndexError):
        return v


# Factor tooltips for Wald test tables (row labels)
WALD_FACTOR_TOOLTIPS = {
    "Intercept": "Baseline (reference) term.",
    "sex": "Sex: Male vs. Female. Tests whether the effect of sex on the outcome is significant.",
    "deg": "Degree type (e.g. PhD, Professional).",
    "yrdeg": "Years since degree. Experience measure.",
    "field": "Academic field (e.g. Arts, Other, Professional).",
    "startyr": "Year the professor started at the institution.",
    "year": "Year of observation.",
    "rank": "Academic rank.",
    "admin": "Administrative appointment (yes/no).",
    "id": "Institution or identifier.",
}


def _render_wald_table_with_tooltips(wald_df_display, tooltips_dict=None):
    """Render Wald result dataframe as a styled table with tooltips on the factor (index) column."""
    tips = tooltips_dict if tooltips_dict is not None else WALD_FACTOR_TOOLTIPS
    cols = list(wald_df_display.columns)
    n_cols = len(cols) + 1  # +1 for index
    # Header: index name (e.g. "term") + column names
    widths = [1.5] + [1.0] * len(cols)
    header_cols = st.columns(widths)
    header_cols[0].markdown("**Term**")
    for i, c in enumerate(cols):
        header_cols[i + 1].markdown(f"**{c}**")
    st.divider()
    for idx in wald_df_display.index:
        key = str(idx).split("[")[0] if "[" in str(idx) else idx
        tip = (tips.get(idx, "") or tips.get(key, "")).replace('"', "&quot;")
        var_html = f'<span title="{tip}" style="cursor: help; border-bottom: 1px dotted #666;">{idx} ℹ️</span>' if tip else str(idx)
        row_cols = st.columns(widths)
        row_cols[0].markdown(var_html, unsafe_allow_html=True)
        for i, c in enumerate(cols):
            val = wald_df_display.loc[idx, c]
            row_cols[i + 1].text(str(val) if not isinstance(val, str) else val)
    st.caption("Hover over ℹ️ for term definitions.")

import statsmodels.api as sm
import seaborn as sns
import scipy.stats as stats

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
    
    with st.expander("**Gender Distribution by Academic Field**", expanded=False):
        st.write("In this dataset, there were a total of 19,792 faculty members, with most (15,866) of them being male and 3926 of them being female. When looking at the breakdown of academic fields across genders in the dataset, men make up a larger proportion of faculty members across all three different field types (Arts, Other, and Professional). The most drastic difference in proportions is in the Professional category (Table 1), where men make up 89.8% percent of all members compared to women making up 10.2%, while the least drastic difference is the Arts category, where men make up 71.8% of the members and women make up 28.2%. These are explained by segmenting the academic fields by gender.")
        st.write("Looking at Figure 1a and 1b, the proportion of men who are Professional faculty members more than doubles that of the proportion of women (21.6% for men, 9.9% for women), while a greater proportion of women are Arts faculty members (20.4%) relative to the proportion of men (12.8%). For both genders, faculty members belonging to the Other field made up the largest proportion, with the women having a slightly higher proportion than the men (69.7% relative to 65.6%).")
        
        pie_colors = ['#ADD8E6', '#FF6B6B', '#90EE90']
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

    with st.expander("**Degree Year Distribution by Gender**", expanded=False):
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

    with st.expander("**Academic Rank Distribution by Gender**", expanded=False):
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
    st.subheader("Question 1: Starting Salary Analysis")
    
    st.markdown("""
    **Results Summary for Q1:**
    - Without controlling for any variables, there is a statistically significant difference in starting salaries between male and female faculty members.
    - Using multiple linear regression (MLR) and ANOVA to control for confounding variables such as degree, year of degree, field, and rank, we still find that gender plays a significant role in starting salaries.
    - Thus, we conclude that there is a significant discrepancy in starting salaries by gender that cannot be fully explained by the available experiential and professional factors.
    """)
    
    salary_path = Path(__file__).parent / "salary.txt"
    if salary_path.exists():
        df = pd.read_csv(salary_path, sep=r"\s+")
        starting_df = df[df['year'] == df['startyr']].copy()
        
        col_plot, col_stats = st.columns([1, 1])
        with col_plot:
            st.markdown("### Starting Salaries by Gender")
            fig_box, ax_box = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='sex', y='salary', data=starting_df, ax=ax_box)
            ax_box.set_title("Starting Salaries by Gender (Uncontrolled)")
            ax_box.set_xlabel("Gender")
            ax_box.set_ylabel("Starting Salary")
            st.pyplot(fig_box)
            st.caption("Figure: Boxplot displaying the distribution of starting salaries globally between male and female faculty.")
            
        with col_stats:
            men_start = starting_df[starting_df['sex'] == 'M']['salary']
            women_start = starting_df[starting_df['sex'] == 'F']['salary']
            mean_diff = np.mean(men_start) - np.mean(women_start)
            t_stat, p_val_t = stats.ttest_ind(men_start, women_start, equal_var=False)
            
            st.markdown("### Preliminary Statistics")
            st.write(f"**Mean Salary (Men):** ${np.mean(men_start):.2f}")
            st.write(f"**Mean Salary (Women):** ${np.mean(women_start):.2f}")
            st.write(f"**Mean Difference (Men - Women):** ${mean_diff:.2f}")
            st.write(f"**Welch t-test:** t-statistic = {t_stat:.4f}, p-value = {p_val_t:.4e}")
            
    st.divider()
    
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
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Click here to explore T-test", key="ttest_explore", use_container_width=True):
                    st.session_state.ttest_state = 1
                    st.rerun()
        elif st.session_state.ttest_state == 1:
            # Goal state
            st.markdown("#### Goal")
            st.write("Shuffle the ‘sex’ label and observe how extreme our observed difference will be to random labeling. If the difference is extreme, we can reject the null hypothesis that the average salaries are the same between males and females.")
            col_spacer, col_btn = st.columns([8, 1])
            with col_btn:
                if st.button("Next", key="ttest_goal_next"):
                    st.session_state.ttest_state = 2
                    st.rerun()
        elif 2 <= st.session_state.ttest_state <= 5:
            # Assumptions states (2-5)
            current_assumption_idx = st.session_state.ttest_state - 2
            st.markdown("#### Check Assumptions")
            for i in range(current_assumption_idx + 1):
                name, desc = assumptions[i]
                st.write(f"**{name}:** {desc}")
            
            col_spacer, col_btn = st.columns([8, 1])
            with col_btn:
                if st.button("Next", key=f"ttest_next_{st.session_state.ttest_state}"):
                    st.session_state.ttest_state += 1
                    st.rerun()
        else:
            # Results state
            st.markdown("#### Results")
            res_c1, res_c2 = st.columns(2)
            with res_c1:
                st.write("- **Test Statistic:** -12.00")
                st.write("- **p-value:** 0.00")
                st.write("- **95% CI:** [218.0, 612.4]")
            with res_c2:
                st.info("Difference in average salaries are significant without controlling other variables.")
            
            col_spacer, col_btn = st.columns([6, 2])
            with col_btn:
                st.markdown('''
                <a href="#permutation-test-section" onclick="document.getElementById('permutation-test-section').scrollIntoView({behavior: 'smooth'}); return false;" style="text-decoration: none;">
                    <button style="padding: 0.5rem 1rem; cursor: pointer; background-color: transparent; border: 1px solid #ccc; border-radius: 4px; color: inherit;">Explore Permutation Test</button>
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
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Click here to explore Permutation Test", key="perm_explore", use_container_width=True):
                    st.session_state.perm_state = 1
                    st.rerun()
        elif st.session_state.perm_state == 1:
            # Goal state
            st.markdown("#### Goal")
            st.write("Compare average starting salaries between males and females using a non-parametric approach.")
            st.markdown("##### Hypotheses")
            st.write("- **H₀:** μ(males) - μ(females) = 0")
            st.write("- **Hₐ:** μ(males) - μ(females) ≠ 0")
            st.write("*(If null is true, we should expect our p-value to be large.)*")
            col_spacer, col_btn = st.columns([8, 1])
            with col_btn:
                if st.button("Next", key="perm_goal_next"):
                    st.session_state.perm_state = 2
                    st.rerun()
        else:
            # Results state
            perm_col_left, perm_col_right = st.columns(2)
            with perm_col_left:
                st.markdown("#### Results")
                st.write("- **p-value:** 0.002")
                st.success("At the 5% significance level, we can reject the null hypothesis that the average salaries are the same between males and females.")
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
            
            # Use a session state similar to T-test for Goal/Description
            if 'q1_anova_state' not in st.session_state:
                st.session_state.q1_anova_state = 0 # 0: Goal, 1: Running/Results

            if st.session_state.q1_anova_state == 0:
                st.markdown("#### Goal")
                st.markdown(
                    "Is gender a significant predictor of starting salaries after controlling for other experiential and professional factors? "
                    "We compare a model predicting starting salaries **with** the sex variable against a model **without** it."
                )
                st.markdown(
                    "**Predictors:** degree, year of degree, field, rank, admin duties, and year. "
                    "The comparison is restricted to observations where `year == startyr`."
                )
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("Run ANOVA", key="q1_anova_run_trigger", use_container_width=True):
                        st.session_state.q1_anova_state = 1
                        st.rerun()
            else:
                if not salary_path.exists():
                    st.warning("Data file `salary.txt` not found. Add it to the project directory to run the ANOVA test.")
                else:
                    try:
                        df = pd.read_csv(salary_path, sep=r"\s+")
                        starting_df = df[df['year'] == df['startyr']].copy()
                        starting_df = starting_df.dropna(subset=['salary', 'deg', 'yrdeg', 'field', 'rank', 'admin', 'year'])
                        
                        model_no_sex = smf.ols('salary ~ deg + yrdeg + C(field) + C(rank) + admin + year', data=starting_df).fit()
                        model_with_sex = smf.ols('salary ~ deg + yrdeg + C(field) + C(rank) + admin + year + C(sex)', data=starting_df).fit()
                        
                        anova_results = sm.stats.anova_lm(model_no_sex, model_with_sex)
                        f_stat = anova_results['F'].iloc[1]
                        p_val_anova = anova_results['Pr(>F)'].iloc[1]
                        
                        st.markdown("#### ANOVA Results")
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.metric("F-statistic", f"{f_stat:.4f}")
                        with res_col2:
                            st.metric("p-value", f"{p_val_anova:.4e}", delta="Significant" if p_val_anova < 0.05 else "Not Significant", delta_color="normal" if p_val_anova < 0.05 else "inverse")
                        
                        st.markdown(
                            f"**Conclusion:** {'As the p-value is extremely small (< 0.05), we can reject the null hypothesis and conclude that gender plays a significant role in determining starting salaries even after controlling for other factors.' if p_val_anova < 0.05 else 'At the 5% significance level, we fail to reject the null hypothesis, suggesting that gender does not have a significant effect on starting salaries when controlling for other variables.'}"
                        )
                        
                        # Visualization like Q3
                        st.divider()
                        st.markdown("#### Impact of Gender on Other Coefficients")
                        params_no_sex = model_no_sex.params
                        params_with_sex = model_with_sex.params
                        common = [i for i in params_no_sex.index if i in params_with_sex.index]
                        
                        plot_vars = []
                        for v in common:
                            c_no = float(params_no_sex[v])
                            c_with = float(params_with_sex[v])
                            if c_no != 0:
                                pct = abs((c_with - c_no) / c_no * 100)
                                if pct > 3: # Lower threshold for Q1 to show something interesting
                                    plot_vars.append(v)
                        
                        if plot_vars:
                            n_var = len(plot_vars)
                            fig_q1_anova = make_subplots(
                                rows=n_var,
                                cols=1,
                                subplot_titles=[v.replace("C(", "").replace(")", "").replace("[T.", ": ") for v in plot_vars],
                                vertical_spacing=0.1,
                            )
                            for i, var in enumerate(plot_vars):
                                fig_q1_anova.add_trace(
                                    go.Bar(x=["Without sex"], y=[params_no_sex[var]], marker_color="#000080", name="Without sex", showlegend=(i == 0)),
                                    row=i + 1, col=1
                                )
                                fig_q1_anova.add_trace(
                                    go.Bar(x=["With sex"], y=[params_with_sex[var]], marker_color="#b22222", name="With sex", showlegend=(i == 0)),
                                    row=i + 1, col=1
                                )
                            fig_q1_anova.update_layout(height=max(300, 150 * n_var), showlegend=True, barmode="group", margin=dict(l=20, r=20, t=40, b=20))
                            st.plotly_chart(fig_q1_anova, use_container_width=True, key="q1_anova_coef_bar")
                        
                        with st.expander("Show ANOVA Table"):
                            st.dataframe(anova_results.fillna(0.0).astype(str), use_container_width=True)

                    except Exception as e:
                        st.error(f"Error computing ANOVA: {e}")
                
                if st.button("Hide ANOVA Results", key="q1_anova_hide_trigger"):
                    st.session_state.q1_anova_state = 0
                    st.rerun()
        with wald_col:
            st.markdown("## Wald Test")
            st.caption("Description")
            st.markdown(
                "Same-year salary model: OLS of **salary** on sex, deg, yrdeg, field, startyr, year, rank, admin "
                "restricted to rows where `startyr == year` (first year at institution). "
                "Wald test of all terms uses **HC3** robust standard errors."
            )
            st.caption("Results")
            if "q1_wald_run" not in st.session_state:
                st.session_state.q1_wald_run = False
                
            q1_wald_left, q1_wald_right = st.columns([1, 1])
            with q1_wald_left:
                if st.button("Run Wald Test", key="q1_wald_run_btn"):
                    st.session_state.q1_wald_run = True
            with q1_wald_right:
                if st.button("Hide Wald Results", key="q1_wald_hide_btn"):
                    st.session_state.q1_wald_run = False
                    
            if st.session_state.q1_wald_run:
                if not salary_path.exists():
                    st.warning("Data file `salary.txt` not found. Add it to the project directory to run the Wald test.")
                elif robust_wald_same_year is not None:
                    wald_df = getattr(robust_wald_same_year, "result_frame", None) or getattr(robust_wald_same_year, "table", None)
                    if wald_df is not None:
                        wald_df_display = wald_df.copy()
                        for c in wald_df_display.columns:
                            if "statistic" in str(c).lower() or str(c).lower() in ("f", "chi2", "wald"):
                                def _round_stat(v):
                                    try:
                                        a = np.asarray(v).flatten()
                                        return round(float(a[0]), 5) if len(a) else v
                                    except (TypeError, ValueError, IndexError):
                                        return v
                                wald_df_display[c] = wald_df_display[c].apply(_round_stat)
                        pval_col_display = None
                        for c in wald_df_display.columns:
                            cl = str(c).lower()
                            if "pvalue" in cl or "p>" in cl or "pr>" in cl:
                                pval_col_display = c
                                break
                        if pval_col_display is not None:
                            wald_df_display[pval_col_display] = wald_df_display[pval_col_display].apply(_format_pval_5digits)
                        
                        wald_df_display = wald_df_display.astype(str).replace(r"\[\[|\]\]", "", regex=True)
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
                            
                            st.divider()
    
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
                                    x_vals = np.round(s.values.astype(float), 5)
                                    fig_stat = go.Figure(go.Bar(
                                        x=x_vals, y=s.index.astype(str),
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
                        with wald_tab_right:
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
            else:
                st.info("Click 'Run Wald Test' to process and view robustness checks.")

        st.subheader("MLR Test")
        mlr_base_tab, mlr_interaction_tab = st.tabs(["Base", "Interaction"])

        with mlr_base_tab:
            st.caption("Description")
            st.markdown(
                "This **base MLR model** predicts starting salaries using predictors such as degree, year of degree, field, rank, and administrative duties. "
                "The `sex` variable is included to see if gender significantly impacts the starting salary baseline after controlling for these other main effects. "
                "Because this is evaluated only on starting years, it analyzes the baseline pay entry wage gap."
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
                    st.caption("Standard errors: HC3 robust. Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
                    _render_ols_coef_table(mls_model_same_year, Q1_MLR_TOOLTIPS, highlight_var="sex")

        with mlr_interaction_tab:
            st.caption("Description")
            st.markdown(
                "This **interaction MLR model** takes the base model further by multiplying the `sex` variable with every other covariate. "
                "This allows us to see not just if there is a flat wage penalty for females, but if the *effect* of other factors (like getting a PhD or being in a certain field) "
                "differs by gender. For example, do men get a higher starting salary bump for a PhD than women do?"
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
                    st.caption("Standard errors: HC3 robust. Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
                    _render_ols_coef_table(mls_model_same_year_interact, Q1_MLR_TOOLTIPS, highlight_var="sex")

# Variable tooltips for Q1 MLR (same-year salary, base and interaction)
Q1_MLR_TOOLTIPS = {
    "Intercept": "Baseline expected salary when all predictors are at reference level.",
    "sex": "Sex: Male vs. Female (reference). Effect of sex on same-year salary.",
    "deg": "Degree type (PhD, Professional, etc.).",
    "yrdeg": "Years since degree.",
    "field": "Academic field (Arts, Other, Professional).",
    "startyr": "Year the professor started at the institution.",
    "year": "Year of observation.",
    "rank": "Academic rank.",
    "admin": "Administrative appointment (yes/no).",
    "sex:deg": "Interaction: sex × degree. Difference in the effect of degree by sex.",
    "sex:yrdeg": "Interaction: sex × years since degree.",
    "sex:field": "Interaction: sex × field.",
    "sex:year": "Interaction: sex × year.",
    "sex:startyr": "Interaction: sex × start year.",
    "sex:rank": "Interaction: sex × rank.",
    "sex:admin": "Interaction: sex × administrative appointment.",
    "id": "Institution or identifier.",
    "sex:id": "Interaction: sex × institution/identifier.",
}

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


def _tip_for_param(param_name: str, tips: dict) -> str:
    """Get tooltip for a param name; try exact key, then interaction key (e.g. sex:deg), then prefix before '['."""
    s = str(param_name)
    if param_name in tips:
        return tips[param_name]
    if ":" in s:
        key_inter = ":".join(p.split("[")[0] for p in s.split(":"))
        if key_inter in tips:
            return tips[key_inter]
    key = s.split("[")[0].split(":")[0]
    return tips.get(key, "")


def _format_p_display(p: float) -> str:
    """Format p-value for display with significance stars."""
    if p < 0.001:
        return "<0.001***"
    if p < 0.01:
        return f"{p:.3f}**"
    if p < 0.05:
        return f"{p:.3f}*"
    if p < 0.1:
        return f"{p:.3f}."
    return f"{p:.3f}"


def _render_ols_coef_table(model, tooltips: dict, highlight_var: str = "sex"):
    """Render OLS coefficient table in m2_tab style (Variable, Estimate, Robust SE, t, p, 95% CI) with tooltips."""
    params = model.params
    bse = model.bse
    tvalues = model.tvalues
    pvalues = model.pvalues
    conf_int = model.conf_int(alpha=0.05)
    header_cols = st.columns([1.8, 1, 1, 0.8, 0.8, 1.5])
    for i, label in enumerate(["Variable", "Estimate", "Robust SE", "t", "p", "95% CI"]):
        header_cols[i].markdown(label)
    st.divider()
    for name in params.index:
        est = float(params[name])
        se = float(bse[name])
        t = float(tvalues[name])
        p = float(pvalues[name])
        ci_lo, ci_hi = float(conf_int.loc[name, 0]), float(conf_int.loc[name, 1])
        est_str = f"{est:.4g}" if abs(est) < 1e-4 or abs(est) >= 1e4 else f"{est:.4f}"
        se_str = f"{se:.4g}" if abs(se) < 1e-4 or abs(se) >= 1e4 else f"{se:.4f}"
        t_str = f"{t:.3f}"
        p_str = _format_p_display(p)
        ci_str = f"({ci_lo:.2f}, {ci_hi:.2f})"
        tip = _tip_for_param(name, tooltips)
        custom_tips = {name: tip} if tip else {}
        for k, v in tooltips.items():
            custom_tips.setdefault(k, v)
        highlight = highlight_var and highlight_var in str(name).lower()
        _q2_coef_row_robust(name, est_str, se_str, t_str, p_str, ci_str, highlight=highlight, tooltips=custom_tips)
    st.caption("Hover over ℹ️ for variable definitions. Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")


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
                st.image("q2_plots/step1_prepared_data.png", use_container_width=True)
            else:
                st.image("q2_plots/step1_salary_diff.png", use_container_width=True)
        st.markdown("**Step 1b: Salary Difference EDA**")
        st.markdown("To get a glance at our data, we plotted the density and boxplot of the salary difference to see the distribution of the salary difference between males and females.")
        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            st.image("q2_plots/salary_diff_density.png", use_container_width=True)
        with plot_col2:
            st.image("q2_plots/salary_diff_boxplot.png", use_container_width=True)
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
            st.image("q3_plots/psm_graph.png", use_column_width=True)
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
        # Load same-year models for Q3 Base / Interaction (same as Q1)
        salary_path_q3 = Path(__file__).parent / "salary.txt"
        df_q3 = None
        q3_mls_base = None
        q3_mls_interact = None
        if salary_path_q3.exists():
            try:
                df_q3 = pd.read_csv(salary_path_q3, sep=r"\s+")
                q3_mls_base = smf.ols(
                    formula="salary ~ sex + id + deg + yrdeg + field + startyr + year + rank + admin",
                    data=df_q3,
                ).fit(cov_type="HC3")
                q3_mls_interact = smf.ols(
                    formula="salary ~ sex + id + deg + yrdeg + field + startyr + year + rank + admin + "
                    "sex:deg + sex:yrdeg + sex:field + sex:year + sex:startyr + sex:rank + sex:admin + sex:id",
                    data=df_q3,
                ).fit(cov_type="HC3")
            except Exception:
                pass
        q3_base_tab, q3_interaction_tab, q3_confounders_tab = st.tabs(["Base", "Interaction", "Attempting to Find Confounders"])
        with q3_base_tab:
            st.caption("Results")
            if "q3_mlr_base_run" not in st.session_state:
                st.session_state.q3_mlr_base_run = False
            st.latex(
                r"\widehat{\text{salary}} = \beta_0 + \beta_1 \text{sex} + \beta_2 \text{deg} + \beta_3 \text{yrdeg} "
                r"+ \beta_4 \text{field} + \beta_5 \text{startyr} + \beta_6 \text{year} + \beta_7 \text{rank} "
                r"+ \beta_8 \text{admin} + \epsilon"
            )
            q3_mlr_left, q3_mlr_right = st.columns([1, 1])
            with q3_mlr_left:
                if q3_mls_base is not None:
                    run_q3_base = st.button("Run MLR", key="q3_mlr_base_btn")
                    if not st.session_state.q3_mlr_base_run:
                        st.info("Click **Run MLR** (left) to run the regression and see results here.")
                    hide_q3_base = st.button("Hide Regression Results", key="q3_mlr_base_hide_btn")
                    if run_q3_base:
                        st.session_state.q3_mlr_base_run = True
                    if hide_q3_base:
                        st.session_state.q3_mlr_base_run = False
                    st.caption("Same-year OLS (HC3). Show or hide results on the right.")
                else:
                    st.caption("Add `salary.txt` to run the model.")
            with q3_mlr_right:
                if st.session_state.q3_mlr_base_run and q3_mls_base is not None:
                    st.caption("Standard errors: HC3 robust. Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
                    _render_ols_coef_table(q3_mls_base, Q1_MLR_TOOLTIPS, highlight_var="sex")
        with q3_interaction_tab:
            st.caption("Results")
            if "q3_mlr_interaction_run" not in st.session_state:
                st.session_state.q3_mlr_interaction_run = False
            st.latex(
                r"\widehat{\text{salary}} = \beta_0 + \beta_1 \text{sex} + \beta_2 \text{deg} + \beta_3 \text{yrdeg} "
                r"+ \beta_4 \text{field} + \beta_5 \text{year} + \beta_6 \text{rank} + \beta_7 \text{admin}"
            )
            st.latex(
                r"\quad {} + \beta_8 (\text{sex} \times \text{deg}) + \beta_9 (\text{sex} \times \text{yrdeg}) "
                r"+ \beta_{10} (\text{sex} \times \text{field}) + \beta_{11} (\text{sex} \times \text{year}) "
                r"+ \beta_{12} (\text{sex} \times \text{rank}) + \beta_{13} (\text{sex} \times \text{admin}) + \epsilon"
            )
            q3_int_left, q3_int_right = st.columns([1, 1])
            with q3_int_left:
                if q3_mls_interact is not None:
                    run_q3_int = st.button("Run MLR", key="q3_mlr_interaction_btn")
                    if not st.session_state.q3_mlr_interaction_run:
                        st.info("Click **Run MLR** (left) to run the regression and see results here.")
                    hide_q3_int = st.button("Hide Regression Results", key="q3_mlr_interaction_hide_btn")
                    if run_q3_int:
                        st.session_state.q3_mlr_interaction_run = True
                    if hide_q3_int:
                        st.session_state.q3_mlr_interaction_run = False
                    st.caption("Same-year OLS with sex interactions (HC3). Show or hide results on the right.")
                else:
                    st.caption("Add `salary.txt` to run the model.")
            with q3_int_right:
                if st.session_state.q3_mlr_interaction_run and q3_mls_interact is not None:
                    st.caption("Standard errors: HC3 robust. Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
                    _render_ols_coef_table(q3_mls_interact, Q1_MLR_TOOLTIPS, highlight_var="sex")
        with q3_confounders_tab:
            st.caption("Description")
            st.markdown(
                '<div style="min-height: 80px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa; margin-bottom: 1rem;"></div>',
                unsafe_allow_html=True,
            )
            st.caption("Results")
            if "q3_confounders_run" not in st.session_state:
                st.session_state.q3_confounders_run = False
            if "q3_ovb_df" not in st.session_state:
                st.session_state.q3_ovb_df = None
            if df_q3 is not None:
                btn_left, btn_right = st.columns([1, 1])
                with btn_left:
                    if st.button("Run: Find impact of other variables on Sex", key="q3_confounders_btn"):
                        try:
                            controls = ["id", "deg", "yrdeg", "field", "startyr", "year", "rank", "admin"]
                            full_formula = "salary ~ sex + " + " + ".join(controls)
                            m_full = smf.ols(full_formula, data=df_q3).fit(cov_type="HC3")
                            sex_param = [c for c in m_full.params.index if "sex" in c][0]
                            full_beta = m_full.params[sex_param]
                            ovb_results = []
                            for var in controls:
                                reduced_controls = [c for c in controls if c != var]
                                reduced_formula = "salary ~ sex + " + " + ".join(reduced_controls)
                                m_reduced = smf.ols(reduced_formula, data=df_q3).fit(cov_type="HC3")
                                reduced_beta = m_reduced.params[sex_param]
                                delta = full_beta - reduced_beta
                                percent_impact = (delta / reduced_beta) * 100 if reduced_beta != 0 else 0
                                ovb_results.append({
                                    "Omitted Variable": var,
                                    "Sex Beta (Without it)": reduced_beta,
                                    "Shift (%)": percent_impact,
                                })
                            ovb_df = pd.DataFrame(ovb_results).sort_values(by="Shift (%)", ascending=False)
                            st.session_state.q3_ovb_df = ovb_df
                            st.session_state.q3_confounders_run = True
                        except Exception as e:
                            st.error(f"Error running analysis: {e}")
                with btn_right:
                    if st.button("Hide Analysis", key="q3_confounders_hide_btn", disabled=not st.session_state.q3_confounders_run):
                        st.session_state.q3_confounders_run = False
            if st.session_state.q3_confounders_run and st.session_state.q3_ovb_df is not None:
                ovb_df = st.session_state.q3_ovb_df
                st.dataframe(ovb_df, use_container_width=True)
                shift_col = "Shift (%)"
                ovb_sorted = ovb_df.sort_values(by=shift_col, ascending=True)
                colors = ["#b22222" if abs(p) >= 10 else "#000080" for p in ovb_sorted[shift_col]]
                fig_ovb = go.Figure(
                    go.Bar(
                        x=ovb_sorted[shift_col],
                        y=ovb_sorted["Omitted Variable"],
                        orientation="h",
                        marker_color=colors,
                    )
                )
                fig_ovb.add_vline(x=-10, line_dash="dash", line_color="gray")
                fig_ovb.add_vline(x=10, line_dash="dash", line_color="gray")
                fig_ovb.update_layout(
                    title="Shift (%) by Omitted Variable",
                    xaxis_title="Shift (%)",
                    yaxis_title="Omitted Variable",
                    margin=dict(l=10, r=10, t=40, b=40),
                    height=320,
                    showlegend=False,
                )
                fig_ovb.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_ovb, use_container_width=True, key="q3_ovb_bar")
            else:
                if df_q3 is None:
                    st.caption("Add `salary.txt` to run the analysis.")
                else:
                    st.info("Click the button above to run the analysis.")
    with sub_tab3:
        st.subheader("ANOVA and Wald Test")
        st.markdown(
            "ANOVA is used to compare the fit of the baseline MLR model strictly against a model omitting gender. "
            "A Wald test robustly checks if the coefficients for gender significantly differ from zero under heteroskedasticity."
        )
        anova_col, wald_col = st.columns(2)
        with anova_col:
            st.subheader("ANOVA")
            st.caption("Description")
            st.markdown(
                "The ANOVA model we used controlled for all the variables and tested the model with \"sex\" against a model without sex."
            )
            st.markdown("**Model without sex:**")
            st.latex(
                r"\begin{aligned} \widehat{\text{salary}} &= \beta_0 + \beta_1 \text{id} + \beta_2 \text{deg} + \beta_3 \text{yrdeg} \\ "
                r"&\quad + \beta_4 \text{field} + \beta_5 \text{startyr} + \beta_6 \text{year} + \beta_7 \text{rank} \\ "
                r"&\quad + \beta_8 \text{admin} + \epsilon \end{aligned}"
            )
            st.markdown("**Model with sex:**")
            st.latex(
                r"\begin{aligned} \widehat{\text{salary}} &= \beta_0 + \beta_1 \text{sex} + \beta_2 \text{id} + \beta_3 \text{deg} \\ "
                r"&\quad + \beta_4 \text{yrdeg} + \beta_5 \text{field} + \beta_6 \text{startyr} + \beta_7 \text{year} \\ "
                r"&\quad + \beta_8 \text{rank} + \beta_9 \text{admin} + \epsilon \end{aligned}"
            )
            anova_f_stat = None
            anova_p_val = None
            mls_model_whole_no_sex = None
            mls_model_whole_anova = None
            if df_q3 is not None:
                try:
                    mls_model_whole_no_sex = smf.ols(
                        formula="salary ~ id + deg + yrdeg + field + startyr + year + rank + admin",
                        data=df_q3,
                    ).fit()
                    mls_model_whole_anova = smf.ols(
                        formula="salary ~ sex + id + deg + yrdeg + field + startyr + year + rank + admin",
                        data=df_q3,
                    ).fit()
                    anova_results_whole = anova_lm(mls_model_whole_no_sex, mls_model_whole_anova)
                    anova_f_stat = float(anova_results_whole["F"].iloc[1])
                    pval_col_anova = "Pr(>F)" if "Pr(>F)" in anova_results_whole.columns else "PR(>F)"
                    anova_p_val = float(anova_results_whole[pval_col_anova].iloc[1])
                except Exception:
                    pass
            if anova_f_stat is not None and anova_p_val is not None:
                st.markdown(
                    f"**F-statistic (sex):** {anova_f_stat:.5f} — **P-value (ANOVA):** {anova_p_val:.6f}"
                )
            else:
                st.markdown(
                    '<div style="min-height: 80px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa; margin-bottom: 1rem;"></div>',
                    unsafe_allow_html=True,
                )
            st.caption("Results")
            if df_q3 is not None and mls_model_whole_no_sex is not None and mls_model_whole_anova is not None:
                try:
                    r2_no_sex = mls_model_whole_no_sex.rsquared
                    r2_with_sex = mls_model_whole_anova.rsquared
                    st.markdown("**R² comparison**")
                    st.write(f"Model without sex: R² = {r2_no_sex:.5f}")
                    st.write(f"Model with sex: R² = {r2_with_sex:.5f}")
                    st.write(f"ΔR² = {(r2_with_sex - r2_no_sex):.5f}")
                    params_no_sex = mls_model_whole_no_sex.params
                    params_with_sex = mls_model_whole_anova.params
                    sex_like = [i for i in params_with_sex.index if "sex" in str(i).lower()]
                    common = [i for i in params_no_sex.index if i in params_with_sex.index and i not in sex_like]
                    # Only plot variables where coefficient change exceeds 10%
                    common_large_change = []
                    for v in common:
                        c_no = float(params_no_sex[v])
                        c_with = float(params_with_sex[v])
                        if c_no != 0:
                            pct = abs((c_with - c_no) / c_no * 100)
                            if pct > 10:
                                common_large_change.append(v)
                        else:
                            if abs(c_with - c_no) > 0:
                                common_large_change.append(v)
                    plot_vars = common_large_change
                    if plot_vars:
                        n_var = len(plot_vars)
                        fig_anova = make_subplots(
                            rows=n_var,
                            cols=1,
                            subplot_titles=plot_vars,
                            vertical_spacing=0.06,
                            shared_xaxes=False,
                        )
                        for i, var in enumerate(plot_vars):
                            c_no = float(params_no_sex[var])
                            c_with = float(params_with_sex[var])
                            fig_anova.add_trace(
                                go.Bar(x=["Without sex"], y=[c_no], marker_color="#000080", name="Without sex", showlegend=(i == 0)),
                                row=i + 1,
                                col=1,
                            )
                            fig_anova.add_trace(
                                go.Bar(x=["With sex"], y=[c_with], marker_color="#b22222", name="With sex", showlegend=(i == 0)),
                                row=i + 1,
                                col=1,
                            )
                        fig_anova.update_layout(
                            title="Significant Coefficient Value Changes (>10% change) (excluding sex)",
                            height=max(320, 120 * n_var),
                            showlegend=True,
                            barmode="group",
                        )
                        for i in range(1, n_var + 1):
                            fig_anova.update_xaxes(title_text="", row=i, col=1)
                            fig_anova.update_yaxes(title_text="Coefficient", row=i, col=1)
                        st.plotly_chart(fig_anova, use_container_width=True, key="q3_anova_coef_bar")
                    else:
                        st.caption("No variables with coefficient change &gt; 10%.")
                except Exception as e:
                    st.error(f"Error running ANOVA: {e}")
            else:
                st.markdown(
                    '<div style="min-height: 280px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa;"></div>',
                    unsafe_allow_html=True,
                )
        with wald_col:
            st.subheader("Wald Test")
            st.caption("Description")
            st.markdown(
                "Full sample OLS of **salary** on sex, id, deg, yrdeg, field, startyr, year, rank, admin. "
                "Wald test of all terms uses **HC3** robust standard errors."
            )
            st.caption("Results")
            if df_q3 is None:
                st.warning("Data file `salary.txt` not found. Add it to the project directory to run the Wald test.")
            else:
                try:
                    mls_model_whole = smf.ols(
                        formula="salary ~ sex + id + deg + yrdeg + field + startyr + year + rank + admin",
                        data=df_q3,
                    ).fit(cov_type="HC3")
                    robust_anova_whole = mls_model_whole.wald_test_terms()
                    wald_df_q3 = getattr(robust_anova_whole, "result_frame", None) or getattr(robust_anova_whole, "table", None)
                    if wald_df_q3 is not None and hasattr(wald_df_q3, "to_html"):
                        wald_df_q3_display = wald_df_q3.copy()
                        for c in wald_df_q3_display.columns:
                            if "statistic" in str(c).lower() or str(c).lower() in ("f", "chi2", "wald"):
                                def _round_stat_q3(v):
                                    try:
                                        a = np.asarray(v).flatten()
                                        return round(float(a[0]), 5) if len(a) else v
                                    except (TypeError, ValueError, IndexError):
                                        return v
                                wald_df_q3_display[c] = wald_df_q3_display[c].apply(_round_stat_q3)
                        pval_col_q3_display = None
                        for c in wald_df_q3_display.columns:
                            cl = str(c).lower()
                            if "pvalue" in cl or "p>" in cl or "pr>" in cl:
                                pval_col_q3_display = c
                                break
                        if pval_col_q3_display is not None:
                            wald_df_q3_display[pval_col_q3_display] = wald_df_q3_display[pval_col_q3_display].apply(_format_pval_5digits)
                        wald_df_q3_display = wald_df_q3_display.astype(str).replace(r"\[\[|\]\]", "", regex=True)
                        wald_q3_left, wald_q3_right = st.columns([1, 1])
                        with wald_q3_left:
                            st.dataframe(wald_df_q3_display, use_container_width=True)
                            st.markdown("**Download results**")
                            csv_bytes_q3 = wald_df_q3_display.to_csv(index=True).encode("utf-8")
                            st.download_button(
                                "Download Wald test table (CSV)",
                                data=csv_bytes_q3,
                                file_name="wald_test_q3_results.csv",
                                mime="text/csv",
                                key="wald_q3_download",
                            )
                        with wald_q3_right:
                            stat_col_q3 = None
                            pval_col_q3 = None
                            for c in wald_df_q3.columns:
                                c_lower = str(c).lower()
                                if stat_col_q3 is None and ("statistic" in c_lower or c_lower in ("f", "chi2", "wald")):
                                    stat_col_q3 = c
                                if pval_col_q3 is None and ("pvalue" in c_lower or "p>" in c_lower or "pr>" in c_lower):
                                    pval_col_q3 = c
                            if stat_col_q3 is None and len(wald_df_q3.columns) >= 1:
                                stat_col_q3 = wald_df_q3.select_dtypes(include=[np.number]).columns[0]
                            if pval_col_q3 is None and len(wald_df_q3.columns) >= 2:
                                num_cols_q3 = wald_df_q3.select_dtypes(include=[np.number]).columns.tolist()
                                pval_col_q3 = num_cols_q3[1] if len(num_cols_q3) > 1 else num_cols_q3[0]

                            def _bar_colors_q3(labels, navy="#000080", red="#b22222"):
                                return [red if "sex" in str(lb).lower() else navy for lb in labels]

                            if stat_col_q3 is not None:
                                s_q3 = wald_df_q3[stat_col_q3].copy()
                                if hasattr(s_q3, "values") and s_q3.dtype == object:
                                    s_q3 = pd.to_numeric(s_q3.astype(str).str.replace(r"[^\d.e\-]", "", regex=True), errors="coerce")
                                s_q3 = s_q3.dropna().sort_values(ascending=False)
                                if len(s_q3) > 0:
                                    x_vals_q3 = np.round(s_q3.values.astype(float), 5)
                                    fig_stat_q3 = go.Figure(go.Bar(
                                        x=x_vals_q3, y=s_q3.index.astype(str),
                                        orientation="h",
                                        marker_color=_bar_colors_q3(s_q3.index),
                                    ))
                                    fig_stat_q3.update_layout(
                                        title="Statistic by factor",
                                        xaxis_title=stat_col_q3,
                                        yaxis_title="",
                                        margin=dict(l=10, r=10, t=40, b=40),
                                        height=280,
                                        showlegend=False,
                                    )
                                    fig_stat_q3.update_yaxes(autorange="reversed")
                                    st.plotly_chart(fig_stat_q3, use_container_width=True, key="wald_q3_stat_bar")
                    else:
                        st.text(str(robust_anova_whole))
                except Exception as e:
                    st.error(f"Error running Wald test: {e}")
                    st.text(str(e))
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
