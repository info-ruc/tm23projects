package org.example;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.aliyuncs.DefaultAcsClient;
import com.aliyuncs.IAcsClient;
import com.aliyuncs.alinlp.model.v20200629.*;
import com.aliyuncs.exceptions.ClientException;
import com.aliyuncs.profile.DefaultProfile;

import java.util.ArrayList;
import java.util.List;

/*AccessKey ID
LTAI5tRWp51SMY97Trw2krec

AccessKey Secret
5igeP8TJEBpun2Alrs7l9wXwdEkTbP*/


public class Main {


    static String accessKeyId = "LTAI5tRWp51SMY97Trw2krec";
    static String accessKeySecret ="5igeP8TJEBpun2Alrs7l9wXwdEkTbP";


    public static void main(String[] args) throws ClientException {


        List<Fp> fpList=new ArrayList<>(); //导入需求文档，解析需求文档，初始化fpList


        Fp fp=new Fp("id","id","level1Fp","level2Fp","登录功能","通过手机号接收验证码登录","",0,0,0,0);
        fpList.add(fp);







        /*根据历史数据的关键词和语义相似度计算估算值*/
        for (int i=0;i<fpList.size();i++)
        {

            int pvWorst = 0,pv = 0,pvBest = 0;

            JSONArray keywordJSONArray=analysisKeyword(fpList.get(i));//提取关键词

            List<Fp> sameKeywordFpList=getSameKeywordFpList(keywordJSONArray);//导入需求文档，解析需求文档，初始化fpList


            for (int j=0;j<sameKeywordFpList.size();j++)
            {
                //进度绩效指数spi=ev（挣值）/pv(计划值)，在完工条件下最终的进度绩效指数spi=ac（实际成本）/pv(计划值)
                float spi=sameKeywordFpList.get(j).getAc()/sameKeywordFpList.get(j).getPv();


                float similarityValue=analyticalSimilarity(sameKeywordFpList.get(j).getDescription(),fpList.get(i).getDescription());

                pvWorst+=sameKeywordFpList.get(j).getPvWorst()*spi*similarityValue;
                pv+=sameKeywordFpList.get(j).getPv()*spi*similarityValue;
                pvBest+=sameKeywordFpList.get(j).getPvBest()*spi*similarityValue;

            }


            pvWorst=pvWorst/sameKeywordFpList.size();
            pv=pv/sameKeywordFpList.size();
            pvBest=pvBest/sameKeywordFpList.size();



            fpList.get(i).setPvWorst(pvWorst);
            fpList.get(i).setPv(pv);
            fpList.get(i).setPvBest(pvBest);


        }

        /* 持久化写入数据库*/
        dao(fpList);


    }


    private static List<Fp> getSameKeywordFpList(JSONArray keywordJSONArray)
    {
        /*SELECT * FROM fp WHERE MATCH (fp.keyword) AGAINST ('keyword');*/


        List<Fp> fpList=new ArrayList<>(); //导入需求文档，解析需求文档，初始化fpList


        Fp fp=new Fp("id","id","level1Fp","level2Fp","登录功能","通过邮箱接收验证码登录","验证码",40,24,16,30);
        fpList.add(fp);

        return fpList;
    }

    private static JSONArray analysisKeyword(Fp fp) throws ClientException {

        DefaultProfile defaultProfile = DefaultProfile.getProfile(
                "cn-hangzhou",
                accessKeyId,
                accessKeySecret);


        IAcsClient client = new DefaultAcsClient(defaultProfile);


        //构造请求参数，其中GetPosChEcom是算法的actionName, 请查找对应的《API基础信息参考》文档并替换为您需要的算法的ActionName，示例详见下方文档中的：更换API请求
        GetKeywordChEcomRequest request = new GetKeywordChEcomRequest();
        //固定值，无需更改
        request.setSysEndpoint("alinlp.cn-hangzhou.aliyuncs.com");
        //固定值，无需更改
        request.setServiceCode("alinlp");

        request.setApiVersion("v2");
        //请求参数, 具体请参考《API基础信息文档》进行替换与填写


        request.setText(fp.getDescription());



        long start = System.currentTimeMillis();


        //获取请求结果，注意这里的GetPosChEcom也需要替换
        GetKeywordChEcomResponse response = client.getAcsResponse(request);


        System.out.println(response.hashCode());
        System.out.println(response.getRequestId() + "\n" + response.getData() + "\n" + "cost:" + (System.currentTimeMillis()- start));




        return new JSONArray();

    }


    private static float analyticalSimilarity(String text1,String text2) throws ClientException {


        DefaultProfile defaultProfile = DefaultProfile.getProfile(
                "cn-hangzhou",
                accessKeyId,
                accessKeySecret);


        IAcsClient client = new DefaultAcsClient(defaultProfile);



        GetTsChEcomRequest request = new GetTsChEcomRequest();
        //固定值，无需更改
        request.setSysEndpoint("alinlp.cn-hangzhou.aliyuncs.com");
        //固定值，无需更改
        request.setServiceCode("alinlp");


        //请求参数, 具体请参考《API基础信息文档》进行替换与填写
        request.setOriginQ(text1);
        request.setOriginT(text2);
        request.setType("similarity");



        long start = System.currentTimeMillis();


        //获取请求结果，注意这里的GetPosChEcom也需要替换
        GetTsChEcomResponse response = client.getAcsResponse(request);


        System.out.println(response.hashCode());
        System.out.println(response.getRequestId() + "\n" + response.getData() + "\n" + "cost:" + (System.currentTimeMillis()- start));


        JSONObject jsonObject=JSON.parseObject(response.getData());

        float score=jsonObject.getJSONArray("result").getJSONObject(0).getFloat("score").floatValue();

        return score;
    }



    /*持久化数据*/
    private static void dao( List<Fp> fpList)
    {

        /*此处用一些 dao框架做持久化，如mybatis 持久化到 mysql*/
        System.out.println("持久化成功");

    }
}