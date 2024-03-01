# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import pandas as pd
from utils import human_format, percentage


data_stat = pd.read_csv('plasticity/results/xnli_adaptation_corpus_stat.csv')
tot_tokens = dict(zip(data_stat.lang, data_stat['#tokens_cc100']))
df_tot_tokens = pd.DataFrame(tot_tokens.items(), columns=['lang', '#tokens'])
df_tot_tokens['readable_#tokens'] = df_tot_tokens['#tokens'].apply(human_format)
df_tot_tokens.sort_values('#tokens', inplace=True)
data_stat['ratio_5M'] = data_stat['#tokens_cc100'].apply(lambda x: 5e6 / x) 

tot_de = data_stat[data_stat['lang'] == 'th']['#tokens_cc100'].values[0] 
for num_tokens in [1000, 10000, 100000, 1000000, 5000000, 
                   10000000, 100000000, 1000000000, 5000000000]:
    print("preprocess_vary th {} {}".format(num_tokens/tot_de, human_format((num_tokens))))
    


# for i, row in data_stat[['lang', 'ratio_5M']].iterrows():
#     print('preprocess_5M {} {};'.format(row['lang'], row['ratio_5M']))

# data_stat['ratio_10M'] = data_stat['#tokens_cc100'].apply(lambda x: 10e6 / x) 
# for i, row in data_stat[['lang', 'ratio_10M']].iterrows():
#     print('preprocess_10M {} {};'.format(row['lang'], row['ratio_10M']))

