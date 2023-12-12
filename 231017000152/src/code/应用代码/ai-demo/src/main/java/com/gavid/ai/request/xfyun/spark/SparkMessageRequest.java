package com.gavid.ai.request.xfyun.spark;

import java.util.List;

import com.alibaba.fastjson.annotation.JSONField;
import com.gavid.ai.domain.chat.ai.ChatPrompt;
import com.gavid.ai.domain.xfyun.spark.SparkChoices;
import com.gavid.ai.domain.xfyun.spark.SparkParameter;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;



@Data
@AllArgsConstructor
@NoArgsConstructor
public class SparkMessageRequest {

    Header header;
    Payload payload;

    SparkParameter parameter;

    @Data
    @NoArgsConstructor
    public static class Header {

        @JSONField(name = "app_id")
        private String appId;

        private String uid;

        public Header(String appId, String uid) {
            this.appId = appId;
            this.uid = uid;
        }
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class Payload {
        SparkChoices message;

        public Payload(List<ChatPrompt> promptList) {
            this.message = new SparkChoices();
            this.message.setText(promptList);
        }
    }

}
