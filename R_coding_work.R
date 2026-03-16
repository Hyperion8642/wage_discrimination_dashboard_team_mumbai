#------------------------Q1: T Test-------------------------#
df = read.table("salary.txt", header = TRUE, sep="")
df_filter = df %>% filter(startyr == year)
df_male = df %>% filter(sex == "M") %>% filter(startyr == year)
df_female = df %>% filter(sex == "F") %>% filter(startyr == year)

# Perform two sample T test
t.test(df_male$salary, df_female$salary)
# Fit MLR model
summary(lm(salary ~ sex+ deg + yrdeg + year+  field + rank + admin, data=df_filter))
#-----------------------------------------------------------#
#------------------------Q2: T Test-------------------------#
salary <- read.table("salary.txt", header = TRUE)

# Create new dataframe of only staff members that were promoted from 
# associate to full professor, grouping by id and keeping their salary data.
# Create new column salary_diff by taking difference in monthly salary between
# first year as full professor and last year as associate professor.
promoted <- salary %>% 
  group_by(id) %>%
  arrange(startyr) %>%
  summarise(
    sex = sex[1],
    last_assoc_year = if (any(rank == "Assoc")) max(year[rank == "Assoc"]) else NA,
    first_full_year = if (any(rank == "Full")) min(year[rank == "Full"]) else NA,
    promoted_to_full = as.integer(!is.na(first_full_year) & !is.na(last_assoc_year) &
                                    first_full_year > last_assoc_year),
    last_assoc_salary = max(salary[year == last_assoc_year]),
    first_full_salary = max(salary[year == first_full_year]),
    salary_diff = first_full_salary-last_assoc_salary
  ) %>% 
  filter(promoted_to_full == 1) %>% 
  select(id, sex, salary_diff)

# Perform two sample T test
t.test(salary_diff ~ sex, data = promoted, var.equal = TRUE)
#------------------------------------------------------------------------------#
#-------------------------------- Q2: MLR -------------------------#
# code for multiple linear regression
salary
library(dplyr)
library(car)
library(sandwich)
library(lmtest)

# calculates which professors were promoted by checking 
# 1) if they were associate
# 2) if they were full 
# 3) if they were both associate and full and full year is after associate 
df <- salary %>% 
  group_by(id) %>%
  arrange(startyr) %>%
  summarise(
    first_assoc_year = if (any(rank == "Assoc")) min(year[rank == "Assoc"]) else NA,
    first_full_year = if (any(rank == "Full")) min(year[rank == "Full"]) else NA,
    promoted_to_full = as.integer(!is.na(first_full_year) & !is.na(first_assoc_year) &
                                    first_full_year > first_assoc_year)
  )

promoted <- salary %>% 
  left_join(df %>% select(id, promoted_to_full), by = "id") %>% 
  filter(promoted_to_full == 1)

# create new column for salary difference (full first year - associate last year)
salary_jump <- promoted %>%
  group_by(id) %>%
  summarise(
    last_assoc_salary = if(any(rank == "Assoc"))
      salary[max(which(rank == "Assoc"))] else NA,
    
    first_full_salary = if(any(rank == "Full"))
      salary[min(which(rank == "Full"))] else NA,
    
    salary_jump = first_full_salary - last_assoc_salary
  )
salary_jump_final <- promoted %>% 
  left_join(salary_jump %>% select(id, salary_jump), by = "id")
salary_jump_final
df_selected <- salary_jump_final %>%
  filter(rank == "Assoc") %>%                     # keep only associate professors
  group_by(id) %>%
  slice_max(year, n = 1, with_ties = FALSE) %>%   # last year they were associate
  ungroup()
df_selected <- df_selected %>%
  select(-c(rank, promoted_to_full))
# remove rank column


# check for multiple linear regression assumptions
df_selected
# multicollinearity - check numeric columns
numeric_salary_jump <- df_selected %>%
  select(where(is.numeric)) 
cor(numeric_salary_jump)


# linearity: check residuals
model <- lm(salary_jump ~ sex + deg + yrdeg + field + startyr + salary + year + admin, data = df_selected)
model
plot(model)

summary(model)
vif(model)
# we don't want multicollinearity, year has the highest VIF score, which states which covariates
# increase cofficient variance the most. 
# r^2 value is 9.24%

# year has the highest value, remove year
model2 <- lm(salary_jump ~ sex + deg + yrdeg + field + startyr + salary + admin, data = df_selected)
summary(model2)
plot(model2)
vif(model2)
# mutliple R squared is 0.09238, vif scores improve 


# The residuals vs fitted plot shows residuals centered around zero with a slight downward trend, suggesting the linearity assumption is mostly reasonable. However, the spread of residuals increases with fitted values, indicating heteroskedasticity. A few large residuals suggest potential influential observations.
# check for outliers
cooks.distance(model2)
which(cooks.distance(model2) > 4/nrow(df_selected))
plot(cooks.distance(model2), type="h", main = "Cooks Distance for Final Model")
abline(h=4/nrow(df_selected), col="red")
?plot
# remove the top 6 outlier values 
cooks <- cooks.distance(model2)
top6 <- order(cooks, decreasing = TRUE)[1:6]
top6
df_no_outliers <- df_selected[-top6, ]

# fit the same model again without top 6 outliers
model3 <- lm(salary_jump ~ sex + deg + yrdeg + field + startyr + salary + admin, data = df_no_outliers)
summary(model3)
plot(model3)
vif(model3)
# removing outliers did not improve the R^2 0.08449 value so we go back to original model 2


# go back to model 2, but we adjust for heterosked.
# some signs of heterosked. and qq plot shows some deviation from
# normal at the right tail end. this is typically ok since we have a large
# sample size and coefficients can be estiamted with a t distribution 
# r^2 value is relatively low, this could be because there are a lot of other 
# factors that influence salary jump 

coeftest(model2, vcov = vcovHC(model2, type = "HC1"))



# does bootstrapping work? nope gave similar ci - [(-112.99,   47.86 )  ]
library(boot)
boot_fn <- function(data, index){
  model <- lm(salary_jump ~ sex + deg + yrdeg + field + startyr + salary + admin, data = df_selected[index, ])
  return(coef(model)["sexM"])
}

boot_results <- boot(df_selected, boot_fn, R = 5000)
boot_results

boot.ci(boot_results, type="perc")


# model the first year salary of being a full professor 
salary_jump_final
library(dplyr)

first_full_salary <- salary_jump_final %>%
  filter(rank == "Full") %>%
  arrange(`id`, year) %>%
  group_by(`id`) %>%
  slice(1) %>%
  ungroup() %>%
  select(-promoted_to_full)
first_full_salary

# check for multicollinearity again with new outcome varible 
numeric_salary <- first_full_salary %>%
  select(where(is.numeric)) 
cor(numeric_salary)
model3 <- lm(salary ~ sex + deg + yrdeg + startyr + year + field + admin, data = first_full_salary)
plot(model3)

summary(model3)
vif(model3)

# there is potential heterscedasticity 
coeftest(model3, vcov = vcovHC(model3, type = "HC1")) 
coefci(model3, vcov. = vcovHC(model3, type = "HC1"))

#------------------------------------------------------------------------------#
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