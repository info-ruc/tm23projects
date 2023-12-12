package com.gavid.ai.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**  
* @Title: AiChatRequest.java  
* @package com.gavid.request
* @Description: TODO(用一句话描述该文件做什么)
* @version V1.0  
*/
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class AiChatRequest {
	private Integer cmdId;
	private Integer reqType;
	private Integer questionId;
	private Integer n;
	private Integer maxTokens;
	private String content;
	private String businessId;
	
}
