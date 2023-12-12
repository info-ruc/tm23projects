import json
import os.path
import re
import requests
from scrapy import Selector
import time

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
}

urls = [
    "https://channel.chinanews.com.cn/cns/cl/fz-ffcl.shtml?pager= 反腐 fanfu",
    "https://channel.chinanews.com.cn/u/gn-la.shtml?pager= 两岸 liangan",
    "https://channel.chinanews.com.cn/cns/cl/cj-fortune.shtml?pager= 金融 jinrong",
    "https://channel.chinanews.com.cn/cns/cl/ty-zhqt.shtml?pager= 体育 tiyu",
    "https://channel.chinanews.com.cn/cns/cl/cj-auto.shtml?pager= 汽车 qiche",
    "https://channel.chinanews.com.cn/cns/cl/cj-house.shtml?pager= 房产 fangchan"
]

reg_page_index = re.compile(r'pager=\d*')
reg_list = re.compile(r'var\s*docArr\s*=\s*(\[\s*\{.*?\}\s*\])\s*;', re.S)
reg_content = re.compile(r'[\u4e00-\u9fa5]+', re.S)
reg_tag = re.compile(r'<[^>]*?>', re.S)

fieldnames = ['tag', 'content']
current_dir = os.getcwd()
contents = set()
detail_urls = set()

ipt = input('即将采集网站数据，6种类别，各100页，\n输入字符y继续，其它字符则停止：')
if 'y' == str(ipt).lower():
    pass
else:
    print('已停止采集数据')
    exit(0)
print('start time: ', end='')
print(time.strftime('%Y-%m-%d %H:%M'))

for chan_index, channel in enumerate(urls):
    print(f'channel: {channel}')
    url = channel.split(' ')[0]
    tag = channel.split(' ')[1]
    filename = channel.split(' ')[2]
    data_file = os.path.join(current_dir, f'../dataset/{filename}.csv')
    # 100页
    for page_num in range(100):
        page_url = url + f'{page_num}'
        print(f'page_url: {page_url}')
        try:
            time.sleep(1)
            result1 = requests.get(page_url, headers=headers)
        except:
            continue
        try:
            items = json.loads(re.search(reg_list, result1.content.decode('utf-8')).groups()[0])
        except:
            continue
        for index, item in enumerate(items):
            detail_url = item.get('url')
            if detail_url in detail_urls:
                continue
            else:
                detail_urls.add(detail_url)
            try:
                time.sleep(1)
                result2 = requests.get(detail_url, headers=headers)
            except:
                continue
            sel = Selector(text=result2.content) if result2.content else None
            content = sel.xpath('//div[@class="left_zw"]').get() if sel else None
            content_zh = ''.join(re.findall(reg_content, content)) if content else None
            if content_zh in contents:
                continue
            else:
                contents.add(content_zh)
            if content_zh:
                if os.path.isfile(data_file):
                    with open(data_file, mode='a+', newline='', encoding='utf-8') as f:
                        f.write(f'{tag},{content_zh}\n')
                else:
                    with open(data_file, mode='a+', newline='', encoding='utf-8') as f:
                        f.write(f'tag,content\n')
                        f.write(f'{tag},{content_zh}\n')

print('end time: ', end='')
print(time.strftime('%Y-%m-%d %H:%M'))
