package com.gavid.ai.domain.xfyun.spark;

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
public class SparkChoices {
    List<ChatPrompt> text;


}
