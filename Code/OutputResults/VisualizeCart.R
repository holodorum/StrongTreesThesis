library("palmerpenguins") ## For 'penguins' dataset
library("rpart")          ## For fitting decisions trees
library("parttree")       ## This package (will automatically load ggplot2 too)
#> Loading required package: ggplot2

## First construct a scatterplot of the raw penguin data
p = ggplot(data = penguins, aes(x = flipper_length_mm, y = bill_length_mm)) +
  geom_point(aes(col = species)) +
  theme_minimal()

## Fit a decision tree using the same variables as the above plot
tree = rpart(species ~ flipper_length_mm + bill_length_mm, data = penguins)

## Visualise the tree partitions by adding it via geom_parttree()
p +  
  geom_parttree(data = tree, aes(fill=species), alpha = 0.1) +
  labs(caption = "Note: Points denote observed data. Shaded regions denote tree predictions.")
#> Warning: Removed 2 rows containing missing values (geom_point).

rpart.plot(tree)
