library(ggplot2)
library("scatterplot3d")
library(colorspace)

df <- read.csv("C:/Users/bartd/Documents/Erasmus_/Jaar 4/Master Econometrie/Thesis/Optimal Trees/StrongTree/Results/aresults_df_time_experiment_synthetic.csv")
#ADD Ratio OBS / Features
df$ratio_obs_feat = df$obs_count / df$enc_features
df$approach <- factor(df$approach)

gg<- ggplot(df, aes(x = depth, y = mean_gap, fill = dataset))

gg + 
  geom_bar(stat='identity', colour='black', position = position_dodge()) +
  geom_errorbar(aes(ymin = mean_gap - std_gap, ymax = mean_gap + std_gap), width = 0.2, position=position_dodge(0.9)) + 
  facet_wrap(~approach, scales="free")

##CREATE PLOT TEST ACC
png("C:/Users/bartd/Documents/Erasmus_/Jaar 4/Master Econometrie/Thesis/Optimal Trees/StrongTree/Results/stupidplots/Results/mean_test_acc_synthetic.png", 
    width=619, height=469)
gg_acc_test<- ggplot(df, aes(x = depth, y = mean_test_acc, fill = approach))

gg_acc_test + 
  geom_bar(stat='idegntity', colour='black', position = position_dodge()) +
  geom_errorbar(aes(ymin = mean_test_acc - std_test_acc, ymax = mean_test_acc + std_test_acc), width = 0.2, position=position_dodge(0.9)) + 
  facet_wrap(~dataset, scales="free")
dev.off()

##CREATE PLOT TRAIN ACC
png("C:/Users/bartd/Documents/Erasmus_/Jaar 4/Master Econometrie/Thesis/Optimal Trees/StrongTree/Results/stupidplots/Results/mean_training_acc_synthetic.png", 
    width=619, height=469)
gg_acc_train <- ggplot(df, aes(x = depth, y = mean_train_acc, fill = approach))

gg_acc_train + 
  geom_bar(stat='identity', colour='black', position = position_dodge()) +
  geom_errorbar(aes(ymin = mean_train_acc - std_train_acc, ymax = mean_train_acc + std_train_acc), width = 0.2, position=position_dodge(0.9)) + 
  facet_wrap(~dataset, scales="free")
dev.off()

##CREATE PLOT TRAIN SOLVING TIME
png("C:/Users/bartd/Documents/Erasmus_/Jaar 4/Master Econometrie/Thesis/Optimal Trees/StrongTree/Results/stupidplots/Results/training_time_synthetic.png",
    width=619, height=469)
gg_time_train <- ggplot(df, aes(x = depth, y = mean_solv_time, fill = approach))

gg_time_train + 
  geom_bar(stat='identity', colour='black', position = position_dodge()) +
  geom_errorbar(aes(ymin = mean_solv_time - std_solve_time, ymax = mean_solv_time + std_solve_time), width = 0.2, position=position_dodge(0.9)) + 
  facet_wrap(~dataset, scales="free")
dev.off()

##CREAT RATIO SCATTER PLOT
gg_scatter_ratio <- ggplot(df, aes(x = mean_fdr, y = enc_features/depth, shape = approach, color = approach, fill = approach))
gg_scatter_ratio +   geom_point(stroke=1, alpha = 0.8, size = 4) +
  scale_shape_manual(values = c(16,16,16,16,16))+
  scale_color_manual(values = c('blue1', 'violetred1', 'gray1', 'magenta4', 'steelblue')) + 
  scale_fill_manual(values=c('black', 'black', 'black', 'black','black')) + 
  geom_jitter(width = 0.05, height = 0.01)

##3d Scatter Plot TDR
colors <- c("#E495A5", "#BDAB66", "#65BC8C", "#55B8D0", "#C29DDE")
colors <- colors[as.numeric(df$approach)]
shapes <- c(16, 17, 18, 19, 20)
shapes <- shapes[as.numeric(df$approach)]
#Add Grid
source('http://www.sthda.com/sthda/RDoc/functions/addgrids3d.r')

s3d <- scatterplot3d(df[, c('enc_features', 'depth', 'mean_fdr')], angle=55, pch= "", color = colors,box = FALSE, grid = FALSE)
#Add grids
addgrids3d(df[, c('enc_features', 'depth', 'mean_fdr')], angle = 55, grid = c("xy", "xz", "yz"))
s3d$points3d(df[, c('enc_features', 'depth', 'mean_fdr')], pch = shapes, , col = colors, alpha = 0.5)

legend("bottomright", legend = levels(df$approach),
       col =   c("#E495A5", "#BDAB66", "#65BC8C", "#55B8D0", "#C29DDE"), 
       pch =  c(16, 17, 18, 19, 20),
       inset=-.05, xpd = TRUE, horiz = FALSE)

#METHOD FOR TWO APPROACHES
png("C:/Users/bartd/Documents/Erasmus_/Jaar 4/Master Econometrie/Thesis/Optimal Trees/StrongTree/Results/stupidplots/Results/3dRateCartFlow.png", 
    width=619, height=469)
methods <- c('FlowOCT', 'Cart')
method_selection <- df$approach %in% methods
colors <- c("blue", "green", 'pink', 'yellow', 'steelblue')
colors <- colors[as.numeric(df[method_selection,'approach'])]
shapes <- c(16, 17,18,19,20)
shapes <- shapes[as.numeric(df[method_selection, 'approach'])]

s3d <- scatterplot3d(df[method_selection, c('enc_features', 'depth', 'mean_fdr')], angle=55, pch= "", color = colors,box = FALSE, grid = FALSE, , zlab = 'mean_tdr', cex.axis=1.15)
addgrids3d(df[method_selection, c('enc_features', 'depth', 'mean_fdr')], angle = 55, grid = c("xy", "xz", "yz"))
s3d$points3d(df[method_selection, c('enc_features', 'depth', 'mean_fdr')], pch = shapes, col = colors, type ='h', cex = 1.7, cex.axis = 2)

legend("bottomright", legend = methods,
       col =   c("blue", "pink"), 
       pch =  shapes,
       inset=-.05, xpd = TRUE, horiz = FALSE)
dev.off()

#Method for heatmap
methods <- c('FlowOCT', 'Cart')
method_selection_cart <- df$approach %in% 'Cart'
method_selection_other <-df$approach %in% 'FlowOCT'

df[method_selection_other, 'subtract'] <- df[method_selection_cart, 'mean_tdr'] - df[method_selection_other, 'mean_tdr']
ggplot(df_test)

