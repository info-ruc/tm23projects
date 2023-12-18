package com.gavid.ai;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;
import org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.scheduling.annotation.EnableScheduling;

/**  
* @Title: ChatApplication.java  
* @package com.gavid.ai
* @Description: TODO(用一句话描述该文件做什么)
* @date 2023年11月27日  
* @version V1.0  
*/
@EnableScheduling
@ComponentScan({"com.gavid"})
@SpringBootApplication(exclude={DataSourceAutoConfiguration.class, SecurityAutoConfiguration.class})
public class ChatApplication {
	public static void main(String[] args) throws Exception {
		System.setProperty("jasypt.encryptor.password", "hip@hanya");
		try {
			SpringApplication.run(ChatApplication.class, args);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
