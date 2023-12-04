package com.gavid.ai.domain.chat.ai;

import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class ChatPrompt {
    //对话的内容
    private String content;
    //role表示对话的角色，取值是system或user。
    //如果需要模型以某个人设形象回答问题，可以将role参数设置为system。
    // 不使用人设使，可设置为user。在一次会话请求中，人设只需要设置一次。
    private String role;

    @Override
    public String toString() {
        return "ChatPrompt{" +
                "content='" + content + '\'' +
                ", role='" + role + '\'' +
                '}';
    }
}
