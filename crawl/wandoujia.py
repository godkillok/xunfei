# 获取软件分类
#爬取豌豆荚

import requests
from bs4 import BeautifulSoup
import pandas as pd

#获取各个分类的url
data = requests.get('https://www.wandoujia.com/category/app')
s = BeautifulSoup(data.text, "html.parser")
divs = [li.div.find_all('a') for li in s.find_all('div')[4].find_all('ul')[0].find_all('li')]

urls_dict = {}
for i in range(len(divs)):
    #print(divs[i])
    for j in range(len(divs[i])):
        title = divs[i][j].attrs['title']
        url = divs[i][j].attrs['href']
        urls_dict[title] = url

base_url = 'https://www.wandoujia.com/wdjweb/api/category/more?catId='
apps = {}
apps_install = {}
for key in urls_dict.keys():
    #    key = '视频'
    num = 1
    page_last = False
    catid = urls_dict[key].split('/')[4].split('_')[0]
    subCatId = urls_dict[key].split('/')[4].split('_')[1]
    title_list = []
    cat_second_list = []
    install_list = []
    while not page_last:  # 每个分类最后一页停止
        # 拼接出每页的url，点击加载更多，page会增1
        url = 'https://www.wandoujia.com/wdjweb/api/category/more?catId={}&subCatId={}&page={}&ctoken=4Op4yfsiSsr8OAzRt5b1MtwE'.format(
            catid, subCatId, num)
        print(url)
        # 爬取对应的网页
        data = requests.get(url)
        # 解析出json
        json = data.json()
        content = json['data']['content']
        if content != '':  # 判断是否最后一页
            soup = BeautifulSoup(content, "html.parser")
            # 获取app的名称
            title_list.extend([li.find_all('a')[1].attrs['title'] for li in soup.find_all('li')])
            # 获取app的二级分类
            cat_second_list.extend([li.find_all('a', {'class': "tag-link"})[0].string for li in soup.find_all('li')])
            # 获取app的安装人数
            install_list.extend(
                ([li.find_all('a')[1].attrs['href'] for li in soup.find_all('li')]))
            # 保存到字典
            apps[key] = dict(zip(title_list, cat_second_list))
            apps_install[key] = dict(zip(title_list, install_list))
            # 加载下一页
            num = num + 1
        else:
            # 触发则表示当前分类已经加载所有页面，即到最后一页
            page_last = True

# 创建空数据框，保存到本地
apps_df = pd.DataFrame(columns=['一级分类', '二级分类', 'app名称', '安装人数'])
app_ls = []
cat_ls = []
ins_ls = []
# 将字典解析出来保存到数据框
for key in apps.keys():
    print(key)
    for app in apps[key].keys():
        app_ls.append(app)
        cat_ls.append(apps[key][app])
        ins_ls.append(apps_install[key][app])

    apps_df_tmp = pd.DataFrame({'app名称': app_ls, '二级分类': cat_ls, '一级分类': key, '安装人数': ins_ls})
    apps_df = apps_df.append(apps_df_tmp)

# 导出
apps_df.to_json('wandoujia_app_cat.csv', orient='records',lines=True)
