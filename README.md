# Random-Forests-Regression-Model

## Libraries Used 

```
library(tidymodels)
library(tidymodels)
library(ISLR)
library(rpart)
library(rpart.plot)
library(ranger)
library(vip)
```

```{r}
Carseats
```

1. Fit a single decision tree to the entire dataset. Report the cross-validated metrics.

```{r}
Carseats_cvs <- vfold_cv(Carseats, v = 5)
Carseats_recipe <- recipe(Sales ~ ., 
                     data = Carseats)
```

```{r}
tree_mod <- decision_tree() %>%
  set_engine("rpart") %>%
  set_mode("regression")

tree_wflow <- workflow() %>%
  add_recipe(Carseats_recipe) %>%
  add_model(tree_mod)

tree_fit <- tree_wflow %>%
  fit_resamples(Carseats_cvs)
                

tree_fit %>% collect_metrics()
```
2. Now, tune your decision tree according to cost_complexity, tree_depth, and min_n to identify the best decision tree model. Report the cross-validated metrics. Plot the final tree and interpret the results.

```{r}
tree_grid <- grid_regular(cost_complexity(),
                          tree_depth(),
                          min_n(), 
                          levels = 2)
tree_grid
```

```{r}
tree_mod <- decision_tree(cost_complexity = tune(),
                          tree_depth = tune(),
                          min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

tree_wflow <- workflow() %>%
  add_recipe(Carseats_recipe) %>%
  add_model(tree_mod)

tree_grid_search <-
  tune_grid(
    tree_wflow,
    resamples = Carseats_cvs,
    grid = tree_grid
  )
tuning_metrics <- tree_grid_search %>% collect_metrics()
```

```{r}
tuning_metrics
```

```{r}
tuning_metrics %>%
  filter(.metric == "rmse") %>%
  slice_max(mean)

tuning_metrics %>%
  filter(.metric == "rsq") %>%
  slice_max(mean)
```

```{r}
tree_mod_2 <- decision_tree(cost_complexity = 0.0000000001,
                          tree_depth = 15,
                          min_n = 40 )%>%
                          set_engine("rpart") %>%
                          set_mode("regression")
  
tree_wflow_2 <- workflow() %>%
  add_recipe(Carseats_recipe) %>%
  add_model(tree_mod_2)

tree_fit_2 <- tree_wflow_2 %>%
  fit_resamples(Carseats_cvs)
                

tree_fit_2 %>% collect_metrics()
```
```{r}
tree_fit_or <- tree_wflow_2 %>% 
  fit(Carseats)
```

```{r}
tree_fitted <- tree_fit_or %>% 
  pull_workflow_fit()

rpart.plot(tree_fitted$fit)
```

3. Determine the best random forest model for these data and report the cross-validated metrics. Is this model better or worse then the single decision tree?

```{r}
splits <- initial_split(Carseats)

car_train <- training(splits)
car_test  <- testing(splits)

val_set <- validation_split(car_train, 
                            prop = 0.80)

Carseats_cvs2 <- vfold_cv(Carseats, v = 5)

rf_mod <- rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("ranger") %>%
  set_mode("regression") 

my_recipe <- recipe(Sales ~., data = car_train, importance = TRUE)

rf_wflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

grid_search <- 
  tune_grid(
    rf_wflow,
    resamples = Carseats_cvs2,
    grid = 25,
    control = control_grid(save_pred = TRUE)
  )

tuning_metrics <- grid_search %>% collect_metrics

tuning_metrics %>%
  filter(.metric == "rsq") %>% 
  arrange(desc(mean))
```

- The best random forest has has a mtry of 8, 517 trees and a min_n of 4.

4. nstall the vip package and checkout its usage here: https://www.tidymodels.org/start/case-study/#second-model. Even though random forests can be harder to interpret, we can still get variable importance scores out of the model results. Use the vip package to display variable importance scores for your final random forest model from (3). Do these scores align with your interpretations from (2) of the single decision tree.

```{r}
last_rf_mod <- 
  rand_forest(mtry = 8, min_n = 4, trees = 517) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("regression")

# the last workflow
last_rf_workflow <- 
  rf_wflow %>% 
  update_model(last_rf_mod)

# the last fit
set.seed(345)
last_rf_fit <- 
  last_rf_workflow %>% 
  last_fit(splits)

last_rf_fit
```
```{r}
last_rf_fit %>% 
  collect_metrics()
```
```{r}
last_rf_fit %>% 
  extract_fit_parsnip() %>% 
  vip(num_features = 20)
```

- These scores do align with our interpretations from the decision tree.

5. Explain what these variable importance scores represent as if you’re describing them to someone who is new to random forests.

- These variable importance scores represent the most important predictors for sales of Carseats, with price, shelveloc, advertising and age being the best predictors 
