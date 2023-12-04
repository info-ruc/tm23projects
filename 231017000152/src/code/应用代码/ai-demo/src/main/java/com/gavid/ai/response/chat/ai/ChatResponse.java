package com.gavid.ai.response.chat.ai;

import java.util.List;

import com.gavid.ai.domain.chat.ai.ChatPrompt;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @Author yangqixin
 * @Date 2023年10月17日
 **/
@Data
@AllArgsConstructor
@NoArgsConstructor
public class ChatResponse {

  List<ChatPrompt> messageList;

  String id;


}
