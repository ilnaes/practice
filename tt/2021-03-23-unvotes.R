library(tidyverse)
library(tidytuesdayR)
library(tidytext)
library(skimr)

tt <- tt_load("2021-03-23")

issues <- tt$issues
rcall <- tt$roll_calls
votes <- tt$unvotes %>% mutate(vote = (vote == "yes"))

votes %>%
  mutate(vote = (vote == "yes")) %>%
  skim()

votes %>%
  left_join(rcall, by = "rcid") %>%
  group_by(country) %>%
  summarize(vote = mean(vote))

votes %>%
  left_join(rcall, by = "rcid") %>%
  left_join(issues, by = "rcid") %>%
  mutate(issue = replace_na(issue, "None")) %>%
  group_by(country, issue) %>%
  summarize(mean = mean(vote), count = n()) %>%
  filter(country == "United States")

votes %>%
  filter(country == "Afghanistan") %>%
  left_join(issues, by = "rcid")
