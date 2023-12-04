package com.gavid.ai.service.impl;

import java.io.IOException;
import java.util.Collections;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.logging.log4j.util.Strings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.gavid.ai.common.exception.HyException;
import com.gavid.ai.common.xfyun.spark.SparkChatComponent;
import com.gavid.ai.domain.chat.ai.ChatPrompt;
import com.gavid.ai.request.AiChatRequest;
import com.gavid.ai.request.chat.ai.ChatRequest;
import com.gavid.ai.response.chat.ai.ChatResponse;
import com.gavid.ai.service.IChatService;

/**  
* @Title: ChatServiceImpl.java  
* @package com.gavid.ai.service.impl
* @Description: TODO(用一句话描述该文件做什么)
* @date 2023年11月27日  
* @version V1.0  
*/
@Service
public class ChatServiceImpl implements IChatService{
	private static final Logger log = LoggerFactory.getLogger(ChatServiceImpl.class);
	@Autowired
    private SparkChatComponent sparkChatComponent;
	

	@Override
	public void streamChatWithWeb(AiChatRequest aiChatRequest, HttpServletRequest request, HttpServletResponse response)
			throws HyException, IOException {
		//先将文本验证审核
//		cloudServiceService.auditContent(content, false);
		log.debug("接收到前端传入的参数为：" + aiChatRequest.toString());
        String content = aiChatRequest.getContent();
        String userName = "admin";

        ChatRequest chatRequest = new ChatRequest();
        ChatPrompt chatPrompt = new ChatPrompt();
        chatPrompt.setContent(content);
        chatPrompt.setRole("user");
        chatRequest.setMessageList(Collections.singletonList(chatPrompt));
        chatRequest.setUserId(userName);
        //chatRequest.setMaxTokens(maxTokens!=null?maxTokens.longValue():0);
        // 发送请求
        sparkChatComponent.send(chatRequest, response, request);
	}
	
	/**  
	* @Title: processChat  
	* @Description:  处理数据  
	* @param data
	* @return    参数  
	* @return String    返回类型  
	*/ 
	public static String processChat(String data) {
        if (Strings.isNotBlank(data)) {
            ChatResponse response = JSON.parseObject(data, ChatResponse.class);
            if (response == null || response.getMessageList() == null) {
                // 响应错误
                JSONObject hyResponse = JSON.parseObject(data);
                return hyResponse.getString("msg");
            } else {
                return response.getMessageList().get(0).getContent();
            }
        }
        return "";
    }

}
