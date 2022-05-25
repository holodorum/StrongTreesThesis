library(ggplot2)

df <- read.csv("C:/Users/bartd/Documents/Erasmus_/Jaar 4/Master Econometrie/Thesis/Optimal Trees/StrongTree/Results/aresults_df_time_experiment_real.csv")
#ADD Ratio OBS / Features
df$ratio_obs_feat = df$obs_count / df$enc_features

gg<- ggplot(df, aes(x = depth, y = mean_gap, fill = dataset))

gg + 
  geom_bar(stat='identity', colour='black', position = position_dodge()) +
  geom_errorbar(aes(ymin = mean_gap - std_gap, ymax = mean_gap + std_gap), width = 0.2, position=position_dodge(0.9)) + 
  facet_wrap(~approach, scales="free")

##CREATE PLOT TEST ACC
png("C:/Users/bartd/Documents/Erasmus_/Jaar 4/Master Econometrie/Thesis/Optimal Trees/StrongTree/Results/stupidplots/Results/mean_test_acc_real.png", 
    width=619, height=469)
gg_acc_test<- ggplot(df, aes(x = depth, y = mean_test_acc, fill = approach))

gg_acc_test + 
  geom_bar(stat='idegntity', colour='black', position = position_dodge()) +
  geom_errorbar(aes(ymin = mean_test_acc - std_test_acc, ymax = mean_test_acc + std_test_acc), width = 0.2, position=position_dodge(0.9)) + 
  facet_wrap(~dataset, scales="free")
dev.off()

##CREATE PLOT TRAIN ACC
png("C:/Users/bartd/Documents/Erasmus_/Jaar 4/Master Econometrie/Thesis/Optimal Trees/StrongTree/Results/stupidplots/Results/mean_training_acc_real.png", 
    width=619, height=469)
gg_acc_train <- ggplot(df, aes(x = depth, y = mean_train_acc, fill = approach))

gg_acc_train + 
  geom_bar(stat='identity', colour='black', position = position_dodge()) +
  geom_errorbar(aes(ymin = mean_train_acc - std_train_acc, ymax = mean_train_acc + std_train_acc), width = 0.2, position=position_dodge(0.9)) + 
  facet_wrap(~dataset, scales="free")
dev.off()

##CREATE PLOT TRAIN SOLVING TIME
png("C:/Users/bartd/Documents/Erasmus_/Jaar 4/Master Econometrie/Thesis/Optimal Trees/StrongTree/Results/stupidplots/Results/training_time_real.png",
    width=619, height=469)
gg_time_train <- ggplot(df, aes(x = depth, y = mean_solv_time, fill = approach))

gg_time_train + 
  geom_bar(stat='identity', colour='black', position = position_dodge()) +
  geom_errorbar(aes(ymin = mean_solv_time - std_solve_time, ymax = mean_solv_time + std_solve_time), width = 0.2, position=position_dodge(0.9)) + 
  facet_wrap(~dataset, scales="free")
dev.off()

##CREAT RATIO SCATTER PLOT
gg_scatter_ratio <- ggplot(df[df$depth==5,], aes(x = mean_train_acc, y = ratio_obs_feat, shape = approach, color = approach, fill = approach))
gg_scatter_ratio +   geom_point(stroke=1, alpha = 0.8, size = 4,  position=position_jitter(h=0.001,w=0.001)) +
  scale_shape_manual(values = c(16,16,16,16,16))+
  scale_color_manual(values = c('blue1', 'violetred1', 'gray1', 'magenta4', 'steelblue')) + 
  scale_fill_manual(values=c('black', 'black', 'black', 'black','black'))
  

