package com.gavid.ai.response.xfyun.spark;

import java.util.List;

import com.gavid.ai.domain.chat.ai.ChatPrompt;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @Author yangqixin
 * @Date 2023年10月16日
 **/

@Data
@AllArgsConstructor
@NoArgsConstructor
public class SparkMessageResponse {

    Header header;
    Payload payload;

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class Header {
        int code;
        int status;
        String sid;

        String message;


    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class Payload {
        SparkChoices choices;
        Usage usage;


    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class SparkChoices {
        int status;
        int seq;

        List<ChatPrompt> text;


    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class Usage {
        Text text;


    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class Text {
        private String questionTokens;
        private String promptTokens;
        private String completionTokens;
        private String totalTokens;


    }

}
