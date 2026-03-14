salary <- read.table("/Users/ianchang/Downloads/salary.txt", header = TRUE)
head(salary)
full_ids <- salary %>%
  group_by(id) %>%
  filter(any(rank == "Full")) %>%
  pull(id) %>% 
  unique()
promoted <- salary %>% 
  group_by(id) %>%
  arrange(startyr) %>%
  summarise(
    sex = sex[1],
    last_assoc_year = ifelse((any(rank == "Assoc")), max(year[rank == "Assoc"]), NA),
    first_full_year = if (any(rank == "Full")) min(year[rank == "Full"]) else NA,
    promoted_to_full = as.integer(!is.na(first_full_year) & !is.na(last_assoc_year) &
                                    first_full_year > last_assoc_year),
    last_assoc_salary = max(salary[year == last_assoc_year]),
    first_full_salary = max(salary[year == first_full_year]),
    salary_diff = first_full_salary-last_assoc_salary
  ) %>% 
  filter(promoted_to_full == 1) %>% 
  select(id, sex, salary_diff)

males <- promoted %>% filter(sex == 'M')
females <- promoted %>% filter(sex == 'F')

t.test(salary_diff ~ sex, data = promoted, var.equal = TRUE)

summary(males$salary_diff)
summary(females$salary_diff)