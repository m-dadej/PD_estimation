library(ggplot2)
library(tidyverse)

df <- data.frame(orignial_default = rep(1000, 4),
                 resampled_default = c(0,0,8000, 0),
                 synth_default = c(0,0,0,1000),
                 non_default = c(9000, 1000, 9000, 3000),
                 dataset = c('original dataset', 'undersampling', 'oversampling', 'SMOTE + undersampling'))

resampling_plot <- mutate(df, dataset = factor(dataset, levels = dataset)) %>%
                    pivot_longer(cols = -dataset) %>%
                      ggplot(aes(fill = name, y = value, x = dataset)) +
                      geom_bar(position = 'stack', stat='identity') +
                      labs(x = '', y = 'N observations', title = 'Resampling strategies', 
                           subtitle = 'General demonstration of resampling strategies') +
                      theme(axis.text.x = element_text(angle = 45, vjust = 0.5), 
                            legend.title = element_blank(),
                            axis.text.y=element_blank(),
                            axis.ticks.y=element_blank()) +
                      scale_fill_discrete(labels = c('non-defaults', 
                                                     'original defaults', 
                                                     'resampled defaults', 
                                                     'synthetic defaults'))

ggsave("py/PD_estimation/latex/img/resampling_strat.png", resampling_plot, width = 7, height = 6)
