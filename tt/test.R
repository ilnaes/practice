library(recipes)

recipe(~., data = iris) %>%
  step_indicate_na(Species) %>% 
  step_dummy(Species) %>%
  prep() %>% 
  bake(new_data=NULL)
