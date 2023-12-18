import requests
import json
import time
import random

class GetData():
    def __init__(self):
        self.init_dataset()

    def init_dataset(self):
        self.text_class = [
            [100, '民生 故事', 'news_story'],
            [101, '文化 文化', 'news_culture'],
            [102, '娱乐 娱乐', 'news_entertainment'],
            [103, '体育 体育', 'news_sports'],
            [104, '财经 财经', 'news_finance'],
            # [105, '时政 新时代', 'nineteenth'],
            [106, '房产 房产', 'news_house'],
            [107, '汽车 汽车', 'news_car'],
            [108, '教育 教育', 'news_edu' ],
            [109, '科技 科技', 'news_tech'],
            [110, '军事 军事', 'news_military'],
            # [111 宗教 无，凤凰佛教等来源],
            [112, '旅游 旅游', 'news_travel'],
            [113, '国际 国际', 'news_world'],
            [114, '证券 股票', 'stock'],
            [115, '农业 三农', 'news_agriculture'],
            [116, '电竞 游戏', 'news_game']
        ]
         
    def get_data(self, item):
        t = int(time.time()/10000)
        t = random.randint(6*t, 10*t)
        
        url = "http://it.snssdk.com/api/news/feed/v63/"
        headers = {
            'Cache-Control': "max-age=0",
            'Host': "it.snssdk.com",
            'Proxy-Connection': "keep-alive",
            'Upgrade-Insecure-Requests': "1",
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
        }
        
        querystring = {
            "category":item[2], 
            "concern_id":"6215497896830175745",
            "refer":"1",
            "count":"20",
            "max_behot_time":t,
            "last_refresh_sub_entrance_interval":"1524907088",
            "loc_mode":"5",
            "tt_from":"pre_load_more",
            "cp":"51a5ee4f38c50q1",
            "plugin_enable":"0",
            "iid":"31047425023",
            "device_id":"51425358841",
            "ac":"wifi",
            "channel":"tengxun",
            "aid":"13",
            "app_name":"news_article",
            "version_code":"631",
            "version_name":"6.3.1",
            "device_platform":"android",
            "ab_version":"333116,297979,317498,336556,295827,325046,239097,324283,170988,335432,332098,325198,336443,330632,297058,276203,286212,313219,328615,332041,329358,322321,327537,335710,333883,335102,334828,328670,324007,317077,334305,280773,335671,319960,333985,331719,336452,214069,31643,332881,333968,318434,207253,266310,321519,247847,281298,328218,335998,325618,333327,336199,323429,287591,288418,260650,326188,324614,335477,271178,326588,326524,326532",
            "ab_client":"a1,c4,e1,f2,g2,f7",
            "ab_feature":"94563,102749",
            "abflag":"3",
            "ssmix":"a",
            "device_type":"MuMu",
            "device_brand":"Android",
            "language":"zh",
            "os_api":"19",
            "os_version":"4.4.4",
            "uuid":"008796762094657",
            "openudid":"b7215ea70ca32066",
            "manifest_version_code":"631",
            "resolution":"1280*720",
            "dpi":"240",
            "update_version_code":"6310",
            "_rticket":"1524907088018",
            "plugin":"256"
        }

        response = requests.request("GET", url, headers=headers, params=querystring)
        return json.loads(response.text)

    def decode_data(self, item, data):
        count = 0
        dataset = ''
        for i in data['data']:
            i = i['content']
            i = i.replace('\"', '"')
            i = json.loads(i)
            kws = ''
            if i.has_key('keywords'):
                kws = i['keywords']
            if i.has_key('ad_id'):
                print(i['ad'])
            elif not i.has_key('item_id') or not i.has_key('title'):
                print(i['bad'])
            else:
                item_id = i['item_id']
                print(count, item[0], item[2], item['item_id'], item['title'], kws)
                line = u"{}_!_{}_!_{}_!_{}_!_{}".format(item['item_id'], item[0], item[2], item['title'], kws)
                line = line.replace('\n', '').replace('\r', '')
                line = line + '\n'
                dataset = dataset + line
                count += 1

    def get_all_data(self, savepath=''):
        for item in self.text_class:
            data = self.get_data(item)
            datatext = self.decode_data(item, data)
            with open(savepath, 'a', encoding='utf-8') as f:
                f.write(datatext)


if __name__ == '__main__':
    data = GetData()
    data.get_all_data(savepath = 'data/news_data.txt')

