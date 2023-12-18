package com.gavid.ai.common.enums;
/**  
* @Title: HyCodeEnum.java  
* @package com.gavid.enums
* @Description: TODO(用一句话描述该文件做什么)
* @version V1.0  
*/
public enum HyCodeEnum {
	
	SUCCESS(0, "操作成功!"),
	SMS_SEND_UNKNOWN_ERROR(30000, "短信发送未知错误"),
	ERROR(50000, "服务器开小差了,请稍后再试!"),
	NOT_PARAMETER(50001, "缺少必要的参数"),
	ERROR_PARAMETER(50003, "参数传递错误"),
	INVALID_PARAMETER(50004, "无效参数"),
	NOSUCHMETHOD_ERROR(50006, "未知方法调用错误!"),
	INVALID_ACCOUNT(50007, "账号未生效或已禁用!"),
	ILLEGAL_ACCESS(50008, "非法访问!"),
	ILLEGAL_ARGUMENT(50009, "非法参数或属性!"),
	API_CALL_ERROR(50010, "接口调用错误!"),
	INVALID_FILE_ERROR(50011, "无效的文件"),
	API_DOC_PARAMS_NOT_FOUND(50012, "接口文档参数未找到!"),
	INVALID_API(50030,"无当前接口权限或已禁用"),
	APP_API_LIMITED_BY_ACCESS_COUNT(50031,"当前API已限流"),
	PERMISSION_IP_WHITELIST_LIMIT (50032,"当前IP限制不允许访问"),
	MISSING_REQUIRED_ARGUMENTS(50040, "缺少所需参数:"),
	INVALID_ARGUMENTS(50041, "无效参数:"),
	USERNAME_OR_PASSWORD_ERROR(50053, "用户名或密码错误!"),
	USER_NAME_EXIST_ERROR(50054, "用户名称已存在!"),
	USER_NAME_FORMAT_ERROR(50055, "用户名格式不正确!"),
	UNAUTHORIZED_OPERATION(50056, "非法操作!"),
	ERROR_NO_PERMISSION(50058, "权限不足!"),
	ACCESS_TOKEN_REFRESH_FAIL(50090, "token刷新失败,请重新获取!"),
	PLEASE_ABIDE_BY_LAWS_AND_REGULATIONS(50091, "请遵循国家相关法律法规!"),
	ERROR_MESSAGE_SEND(50101, "短信验证码发送失败,请稍后再试!"),
	ERROR_DAY_MAX_MESSAGE_SEND(50102, "当天短信验证码发送已上限,请合理发送!"),
	ERROR_SMS_SEND_TOO_OFTEN(50103, "短信验证码发送过于频繁,请稍后再试!"),
	FAIL_SIGN(50104, "签名验证错误!"),
	AUTHORIZATION_FAIL(50105, "服务器鉴权失败!"),
	INVALID_ACCESS_TOKEN(50106, "无效的access token!"),
	ACCESS_TOKEN_EXPIRATION(50107, "token已过期,请重新获取!"),
	FAIL_HMACSHA(50108, "生成授权签名失败,请检查相关参数是否正确!"),
	CREATE_EXCEL_ERROR(50200, "创建Excel文件失败!"),
	EXPORT_EXCEL_FAIL(50201, "导出Excel失败!"),
	APP_PARAMS_SET_ERROR(50202, "应用参数配置异常,请到管理后台修改!"),
	SERVICE_IS_NOT_SUPPORTED_ERROR(50203, "暂不支持的服务调用!"),
	
	FAILED_TO_GENERATE_ANSWER(60000, "AI服务器正忙，请稍后重试!"),
	MODEL_SELECTION_ERROR(60001, "当前模型不存在，请联系管理员!"),
	FAILED_TO_GENERATE_IMAGE(60002, "生成图像错误，请联系管理员!"),
	ERROR_RESPONSE_FORMAT(60003, "请检查格式是否正确!"),
	DOWNLOAD_IMAGE_ERROR(60004, "下载图片错误!"),
	SERVICE_RESPONSE_TIMEOUT(50204, "响应超时!");
	
	private HyCodeEnum(int code, String msg) {
		this.code = code;
		this.msg = msg;
	}
	private int code;
	private String msg;
	public int getCode() {
		return code;
	}
	public void setCode(int code) {
		this.code = code;
	}
	public String getMsg() {
		return msg;
	}
	public void setMsg(String msg) {
		this.msg = msg;
	}
}
