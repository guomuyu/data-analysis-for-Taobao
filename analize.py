import pandas as pd
import jieba
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.rc('font', family='SimHei', size=7)


data = pd.read_excel(r'D：\文档\学习\商务数据分析\项目\淘宝列表采集（无人机）.xlsx')
data = data[{'产品名称', '价格', '月销量'}]
# print(data.head())


add_words = pd.read_excel(r'D：\文档\学习\商务数据分析\项目\add_words.xlsx')
for w in add_words:
    jieba.add_word(w, freq=1000)

title = data['产品名称']
title_s = []
for line in title:
    title_cut = jieba.lcut(line)
    title_s.append(title_cut)
# 输出title_s

# 停用词表剔除
stopwords = pd.read_excel(r'D：\文档\学习\商务数据分析\项目\stopwords.xlsx')
title_clean = []
for line in title_s:
    line_clean = []
    for word in line:
        if word not in stopwords:
            line_clean.append(word)
    title_clean.append(line_clean)
#输出stopwords

# 去重复
title_clean_dist = []
for line in title_clean:
    line_dist = []
    for word in line:
        if word not in line_dist:
            line_dist.append(word)
    title_clean_dist.append(line_dist)
allwords_clean_dist = []
for line in title_clean_dist:
    for word in line:
        allwords_clean_dist.append(word)
df_allwords_clean_dist = pd.DataFrame({'allwords': allwords_clean_dist})
word_count = df_allwords_clean_dist.allwords.value_counts().reset_index()
word_count.columns = ['word', 'count']
sales = data['月销量']
clean_sales = []
for line in sales:
    clean_sale = int(line.replace(',', ''))
    clean_sales.append(clean_sale)
w_s_sum = []
for w in word_count.word:
    i = 0
    s_list = []
    for t in title_clean_dist:
        if w in t:
            s_list.append(clean_sales[i])
        i += 1
    w_s_sum.append(sum(s_list))
df_w_s_sum = pd.DataFrame({'w_s_sum': w_s_sum})
df_w_s_sum = pd.concat([word_count, df_w_s_sum], axis=1, ignore_index=True)
df_w_s_sum.columns = ['word', 'count', 'w_s_sum']
#输出df_w_s_sum
values = data['价格']
clean_values = []
for line in values:
    clean_value = int(line)
    clean_values.append(clean_value)
# 输出clean_values
w_val_mean = []
for w in word_count.word:
    i = 0
    s_list = []
    for t in title_clean_dist:
        if w in t:
            s_list.append(clean_values[i])
        i += 1
    w_val_mean.append(np.mean(s_list))
df_w_val_mean = pd.DataFrame({'w_val_mean': w_val_mean})
df_w_val_mean = pd.concat([df_w_s_sum, df_w_val_mean], axis=1, ignore_index=True)
df_w_val_mean.columns = ['word', 'count', 'w_s_sum', 'w_val_mean']

# 不同关键词对应的销量统计分析
df_w_val_mean.sort_values('w_s_sum',inplace = True, ascending= True)
df_w_drop = df_w_val_mean.drop([12,13,16,24,44,])
df_w_s = df_w_drop.tail(30)
# print(df_w_s)
font = {'family':'SimHei'}
matplotlib.rc('font',**font)
index = np.arange(df_w_s['word'].size)
plt.figure(figsize=(6,12))
plt.barh(index,df_w_s['w_s_sum'],color = 'purple',align = 'center', alpha = 0.8)
plt.yticks(index,df_w_s['word'],fontsize = 11)
for y,x in zip(index,df_w_s['w_s_sum']):
    plt.text(x,y,'%.0f' %x,ha='left',va = 'center',fontsize = 11)
plt.show()

# 不同关键词对应的价格统计分析
df_w_val_mean.sort_values('count',inplace = True, ascending= True)
df_w_count_drop = df_w_val_mean.drop([12,13,16,24,44,])
df_w_values = df_w_count_drop.tail(30)
df_w_values.sort_values('w_val_mean',inplace = True, ascending= True)
# print(df_w_values)
font = {'family':'SimHei'}
matplotlib.rc('font',**font)
index = np.arange(df_w_values['word'].size)
plt.figure(figsize=(6,12))
plt.barh(index,df_w_values['w_val_mean'],color = 'purple',align = 'center', alpha = 0.8)
plt.yticks(index,df_w_values['word'],fontsize = 11)
for y,x in zip(index,df_w_values['w_val_mean']):
    plt.text(x,y,'%.0f' %x,ha='left',va = 'center',fontsize = 11)
plt.show()

# 商品销量分布情况分析
sales = data['月销量']
clean_sales = []
for line in sales:
    clean_sale = int(line.replace(',', ''))
    if clean_sale <= 1000:
        clean_sales.append(clean_sale)
plt.figure(figsize=(7, 5))
plt.tick_params(labelsize=14)
plt.hist(clean_sales, bins=15, color='purple')
plt.xlabel('月销量', fontsize=12)
plt.ylabel('商品数量', fontsize=12)
plt.title('不同销量对应的商品数量分布', fontsize=15)
plt.show()

# 商品价格分布情况分析
values = data['价格']
clean_values = []
for line in values:
    clean_value = int(line)
    if clean_value <= 2000:
        clean_values.append(clean_value)
plt.figure(figsize=(7,5))
plt.tick_params(labelsize=14)
plt.hist(clean_values,bins = 15,color = 'purple')
plt.xlabel('价格',fontsize = 12)
plt.ylabel('商品数量',fontsize = 12)
plt.title('不同价格对应的商品数量分布',fontsize = 15)
plt.show()

# 不同价格区间的商品平均销量分布
values = data['价格']
clean_values = []
for line in values:
    clean_value = int(line)
    clean_values.append(clean_value)
sales = data['月销量']
clean_sales = []
for line in sales:
    clean_sale = int(line.replace(',', ''))
    clean_sales.append(clean_sale)
df_values = pd.DataFrame({'values': clean_values})
df_sales = pd.DataFrame({'sales': clean_sales})
df_data = pd.concat([df_sales, df_values], axis=1, ignore_index=True)
df_data.columns = ['sales', 'values']
# print(df_data['values'])
df_data['group'] = pd.qcut(df_data['values'], 12)
df_group = df_data.group.value_counts().reset_index()
df_s_g = df_data[['sales', 'group']].groupby('group').mean().reset_index()
# print(df_s_g)
index = np.arange(df_s_g.group.size)
plt.figure(figsize=(8, 4))
plt.bar(index, df_s_g['sales'], color='purple')
plt.xticks(index, df_s_g['group'], fontsize=11, rotation=30)
plt.xlabel('Group', fontsize=12)
plt.ylabel('平均月销量', fontsize=12)
plt.title('不同价格区间的商品平均月销量', fontsize=15)
plt.show()

# 价格对销量的影响
df_data_2 = df_data[df_data['values'] < 3000]
fig,ax = plt.subplots()
ax.scatter(df_data_2['values'],df_data_2['sales'],color = 'purple')
ax.set_xlabel('价格',fontsize = 12)
ax.set_ylabel('月销量',fontsize = 12)
ax.set_title('价格对月销量的影响',fontsize = 15)
plt.show()

# 价格对销售额的影响
df_data_3 = df_data[df_data['values'] < 10000]
df_data_GMV = df_data_3['values']*df_data_3['sales']
df_data_4 = pd.concat([df_data_3, df_data_GMV], axis=1, ignore_index=True)
df_data_4.columns = ['sales', 'values', 'group', 'GMV']
sns.regplot(x='values', y='GMV', data = df_data_4, color='purple')
ax.set_xlabel('价格',fontsize = 12)
ax.set_ylabel('月销量',fontsize = 12)
ax.set_title('价格对月销量的影响',fontsize = 15)
plt.show()