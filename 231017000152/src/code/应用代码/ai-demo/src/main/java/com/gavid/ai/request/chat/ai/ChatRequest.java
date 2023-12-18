package com.gavid.ai.request.chat.ai;

import java.util.List;

import com.gavid.ai.domain.chat.ai.ChatPrompt;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class ChatRequest {
    private List<ChatPrompt> messageList;
    // 用户id 必传
    private String userId;
    // 需要保障用户下的唯一性，用于关联用户会话
    //  讯飞独有参数
    private String chatId;

    //用于控制生成文本的多样性和创造力。
    //参数的取值范围是 (0, 100]，取值接近0表示最低的随机性，100表示最高的随机性。
    // 一般来说，temperature越低，适合完成确定性的任务。temperature越高，例如90，适合完成创造性的任务。
    private Long temperature;
    // 用于控制生成文本的多样性。它的含义是在每个时间步，只考虑概率累积和小于top_p的词作为候选词
    // 盘古： 取值范围：(0, 100] 默认为50
    // 讯飞： 取值范围：(1, 6] 默认为4
    private Long topP;
    // 模型回答的tokens的最大长度
    // 盘古：nlp2 接口最大4096
    // 讯飞： V1.5取值为[1,4096]，V2.0取值为[1,8192]。默认为2048
    private Long maxTokens;
    //用于控制生成文本中的重复程度。
    // 正值会根据它们到目前为止在文本中的现有频率来惩罚新tokens，从而降低模型逐字重复同一行的可能性
    //最大值：2 最小值：-2
    // 盘古：默认为0
    // 讯飞不提供此参数
    private Long presencePenalty;
}
