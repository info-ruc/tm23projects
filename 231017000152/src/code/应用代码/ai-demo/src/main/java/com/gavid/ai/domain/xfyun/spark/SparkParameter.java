package com.gavid.ai.domain.xfyun.spark;

import com.alibaba.fastjson.annotation.JSONField;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;


@Data
@NoArgsConstructor
public class SparkParameter {

    private Chat chat;

    public SparkParameter(String domain,Double temperature, Double maxTokens,Double topK,String chatId) {
        this.chat = new Chat(domain, temperature, maxTokens,null,chatId);
    }


    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class Chat{
        private String domain;

        private double temperature;

        @JSONField(name = "max_tokens")
        private Double maxTokens;

        @JSONField(name = "top_k")
        private Double topK;

        @JSONField(name = "chat_id")
        private String chatId;
    }
}
