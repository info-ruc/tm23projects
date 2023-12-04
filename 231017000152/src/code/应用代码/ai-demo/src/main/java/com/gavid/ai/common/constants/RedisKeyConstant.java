package com.gavid.ai.common.constants;
/**  
* @Title: RedisKeyConstant.java  
* @package com.gavid.common.constants
* @Description: TODO(用一句话描述该文件做什么)
* @version V1.0  
*/
public class RedisKeyConstant {
	
	/**
	 * 第三方api接口key
	 */
	public static final String API_METHOD_KEY = "api_method:key_";
	
	/**
	 * api 接口缓存有效时间
	 */
	public static final Integer API_METHOD_KEY_EXPIRE = 300;
	
	/**
	 * redis 中存储accessToken key
	 */
	public final static String REDIS_ACCESS_TOKEN_KEY = "ACCESS_TOKEN_KEY_";
	
	/**
	 * 第三方应用 api 接口缓存有效时间
	 */
	public static final Integer APP_API_METHOD_KEY_EXPIRE = 86400;
	
	 /**
     * 后台记录密码错误次数redis key
     */
    public static String REDIS_SYS_PWD_RETRY_CACHE = "REDIS_SYS_PWD_RETRY_CACHE_";
    
    /**
     * 是否记住账号密码
     */
    public static String REDIS_SYS_IS_WEAK_PWD = "REDIS_SYS_IS_WEAK_PWD_";
    
	/**
	 * DICT KEY 前缀
	 */
	public final static String REDIS_DICT_MANAGER_KEYPREFIX = "sys_dict:";

	/**
	 * 系统接口referer key
	 */
	public final static String SYS_API_REFERER_WHITELIST_KEY = "SYS_API_REFERER_WHITELIST";
	/**
	 * 系统后台referer key
	 */
	public final static String SYS_MAIN_REFERER_WHITELIST_KEY = "SYS_MAIN_REFERER_WHITELIST";

	/**
	 * 云服务调用token有效时间
	 */
	public final static int CLOUD_SERVICE_ACCESS_TOKEN_KEY_EXPIRE = 120*60;

	/**
	 * 云服务调用token
	 */
	public final static String CLOUD_SERVICE_ACCESS_TOKEN_KEY = "cloud_service_access_token_";
	
	/**
	 * chatgpt 问题答案key
	 */
	public final static String CHATGPT_TQUESTION_ANSWER_KEY = "chatgpt_tquestion_answer_";


}
