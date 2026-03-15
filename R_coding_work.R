
#------------------------Q3: Propensity Score Matching-------------------------#

# install.packages(c("dplyr", "MatchIt", "ggplot2"))
library(dplyr)
library(MatchIt)
library(ggplot2)

# 2. Load and Preprocess the Data
df <- read.table("/Users/sreeraj/Downloads/salary.txt", header = TRUE) %>%
  mutate(
    # MatchIt requires the "treatment" variable to be binary (0/1) or logical
    # We set Female = 1 (the group we want to match) and Male = 0 (the pool of peers)
    is_female = ifelse(sex == "F", 1, 0),
    deg = as.factor(deg),
    field = as.factor(field),
    rank = as.factor(rank),
    admin = as.factor(admin)
  )

# Initialize an empty dataframe to store the wage gap percentage for each year
yearly_gaps <- data.frame(year = integer(), gap_pct = numeric())

# Get a sorted list of all the unique years in the dataset (1976 - 1995)
years <- sort(unique(df$year))

for (y in years) {
  
  # 1. Filter for the year AND remove any rows with missing data (NAs)
  # This fixes the "Missing and non-finite values" error
  df_year <- df %>% 
    filter(year == y) %>%
    na.omit() 
  
  # 2. Safety Check: We need both genders AND at least two ranks to match
  # Sometimes in a specific year, all women might be the same rank, which breaks the model
  if(length(unique(df_year$is_female)) < 2 | length(unique(df_year$rank)) < 2) {
    next
  }
  
  # 3. The Match (Wrapped in tryCatch to prevent the whole loop from crashing)
  match_model <- tryCatch({
    matchit(is_female ~ yrdeg + startyr + admin + deg + field + rank, 
            data = df_year, 
            method = "nearest", 
            distance = "glm", 
            ratio = 1, 
            replace = FALSE)
  }, error = function(e) { return(NULL) })
  
  if (is.null(match_model)) next
  
  # 4. Extract Matched Data
  matched_data <- match.data(match_model)
  
  # 5. Post-Matching Regression
  ols_model <- lm(salary ~ is_female + yrdeg + startyr + admin + deg + field + rank, 
                  data = matched_data)
  
  # 6. Extract the Penalty %
  if ("is_female" %in% names(coef(ols_model))) {
    female_penalty_dollars <- coef(ols_model)["is_female"]
    avg_male_salary <- mean(matched_data$salary[matched_data$is_female == 0])
    
    # We use the absolute value to show the 'size' of the gap on the graph
    penalty_pct <- abs(female_penalty_dollars / avg_male_salary) * 100
    
    yearly_gaps <- rbind(yearly_gaps, data.frame(year = y, gap_pct = penalty_pct))
  }
}
cat("\n--- Yearly Wage Penalty Results (%) ---\n")
print(yearly_gaps)

# 4. Generate the Final Line Graph
ggplot(yearly_gaps, aes(x = year, y = gap_pct)) +
  geom_line(color = "blue", linewidth = 1.2) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) + # Format y-axis as percentages
  scale_x_continuous(breaks = seq(min(years), max(years), by = 2)) + # Clean x-axis ticks
  labs(
    title = "Female Wage Penalty Over Time (Matched Peers)",
    subtitle = "Calculated as a % of the matched male peer's salary using PSM",
    x = "Year",
    y = "Wage Penalty (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.title = element_text(size = 12)
  )
#------------------------------------------------------------------------------#