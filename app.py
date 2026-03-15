import streamlit as st
from pathlib import Path

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
                        st.image(str(plots_dir / f), use_container_width=True)
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
                        st.image(str(full_plots_dir / f), use_container_width=True)
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