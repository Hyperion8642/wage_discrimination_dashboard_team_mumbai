import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
                <p style="color: black; font-size: 1.25em;">Compare average starting salaries between male and female faculty members before controlling other variables.</p>
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
        # Add regression results or stats here
        
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