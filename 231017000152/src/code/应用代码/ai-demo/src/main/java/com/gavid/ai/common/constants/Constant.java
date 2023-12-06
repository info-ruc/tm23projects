package com.gavid.ai.common.constants;
/**  
* @Title: Constant.java  
* @package com.gavid.common.constants
* @Description: TODO(用一句话描述该文件做什么)
* @version V1.0  
*/
public class Constant {
	
	/**
	 * 获取token api
	 */
	public final static String USER_LOGIN_API = "/api/user/login";
	/**
	 * 用户修改密码api
	 */
	public final static String USER_PWD_CHANGE_API = "/api/user/pwd/change";
	/**
	 * 获取token api
	 */
	public final static String REFRESH_TOKEN_API = "/api/token/refresh";
	/**
	 * accessoken 名字
	 */
	public final static String ACCESS_TOKEN_NAME = "Authorization";
	/**
	 * accessToken的过期时间
	 */
	public final static Long ACCESS_TOKEN_EXP_TIME = 1440 * 60L;
	/**
	 * refreshToken的过期时间
	 */
	public final static Long REFRESH_TOKEN_EXP_TIME = 120 * 60L;
	/**
	 * token中的识别码，也就是用户登录的账号
	 */
	public static final String JWT_USERNAME = "userName";
	/**
	 * token中的识别码，也就是用户登录的账号密匙
	 */
	public static final String JWT_USERPASSWORD = "userPassword";
	/**
	 * token生成时的设备信息，用于鉴别token
	 */
	public static final String JWT_DEVINFO = "devInfo";
	/**
	 * User-Agent 请求头中的设备信息字段
	 */
	public static final String JWT_USER_AGENT = "User-Agent";
	
	/**
	 * 私密
	 */
	public static final String PRIVATE_KEY="MIICdwIBADANBgkqhkiG9w0BAQEFAASCAmEwggJdAgEAAoGBAK7mgJi1GxyFojo1QzMH8fzczjJRC2nXh45EMuQg1/eUMtN/qdejD1STReJRD63Jphr/mtaAZizkkisD4KhnRg4+xEn46TsY1M7G2IM1/vfIu5jf1rdhEY/5N1S8Nlju8Cy9P0QnwBJBQPflnizvsBfV3dkOp4cwPi1VYDkN+ZHJAgMBAAECgYBrM+nasBdgEiDvoLoBy3rtzMGuYbKnO25hKzguUFtP60yECpomDFJXOrX5FEqR8SmZHtbfZ3A5UBivuP64+iQbi7JVwCVbWLAs677KCFIixvxIwYgzwj3f9JoyhsQmTTIXRiRAU8ecoglYsOmfsO1rW5CRSpRk7z6wxLY5+VK1SQJBAPyIhlJ+a8GFiiY8Us4S/zToRbKq+LfUsZz2cC82oE40Dh7CcZANcNfGeQA6nyZTS6b/LdLqK4itz7kWR07yPjcCQQCxTScxIsLPxgYB862JlRnmkI0YPHJCfvebWCjS9NreCi9QjMPBl/01sTwBGKfPXxcLuuLI/bB3tKprsgW8Cq//AkEAw2WLsUbab6m5JC6mz4bJaxGR5FYADpWHPHE+inmU/g2vI0PGhPSxbHPIalHxlMD8l2F4/mpsdtwuDwNa943ebwJAUYgSIrVCcns0XgdpYOAwteb5CxEY1d0/Da9/rmqsjviOA3OHvWmgJeWnmzV0TZcDqQA6s4R9dr6cs8N8gZlEjwJBAMf3K49flmPcXbAsgv33/GsOg62u98pYg/bYtu/bJ/GtdXvb/v1KCl5uhnFN1xBlzKnV3ufHKTtKd4Z7N2A4kFo=";
	
	/**
	 * Hmac加密方式
	 */
	public final static String HMACSHA_TYPE = "HmacSHA256";
	
	/**
	 * 请求类型POST
	 */
	public final static String REQUEST_METHOD_POST = "POST";
	/**
	 * 请求类型GET
	 */
	public final static String REQUEST_METHOD_GET = "GET";
	
	/**
	 * chatgpt 默认用户
	 */
	public final static String CHATGPT_DEFAULT_USER_NAME = "DEFAULT USER";


	/**
	 * 后台记录密码错误次数redis key
	 */
	public static String REDIS_SYS_PWD_RETRY_CACHE = "REDIS_SYS_PWD_RETRY_CACHE_";

	/**
	 * 前端记录密码错误次数redis key
	 */
	public static String REDIS_WEB_PWD_RETRY_CACHE = "REDIS_WEB_PWD_RETRY_CACHE_";

	public static String REDIS_SYS_IS_WEAK_PWD = "REDIS_SYS_IS_WEAK_PWD_";

	//登陆时间
	public static String REDIS_SYS_LOGIN_CACHE = "REDIS_SYS_LOGIN_CACHE_";

	public static final String SYSUSER_ORG_STRUCTURE_KEY = "SYSUSER_ORG_STRUCTURE";

	public static final String CACHE_GEN_RESVENUE_TASK_KEY = "CACHE_GEN_RESVENUE_TASK";
	/**
	 * 对象存储桶
	 */
	public static final String OBS_BUCKET = "hy-ai";
	
	/**
	 * 手机验证正则
	 */
	public static final String REGEX_MOBILE = "^((13[0-9])|(14[5|7])|(15([0-9]))|(17[013678])|(18[0-9])|(19[0-9]))\\d{8}$";
	
	/**  
	* @Title: isMobile  
	* @Description: 验证手机号码
	* @param mobile
	* @param @return    参数  
	* @return Boolean    返回类型  
	*/ 
	public static Boolean isMobile(String mobile) {
		boolean isMobile = false;
		if (mobile.matches(REGEX_MOBILE)) {
			isMobile = true;
		}
		return isMobile;
	}

}
