package com.gavid.ai.common.exception;

/**
 * @Title: HyException.java
 * @package com.gavid.common.exception
 * @Description: TODO(用一句话描述该文件做什么)
 * @version V1.0
 */
public class HyException extends RuntimeException{
	private static final long serialVersionUID = 1L;

	private String msg;
	private int code = 500;

	public HyException(String msg) {
		super(msg);
		this.msg = msg;
	}

	public HyException(String msg, Throwable e) {
		super(msg, e);
		this.msg = msg;
	}

	public HyException(int code, String msg) {
		super(msg);
		this.code = code;
		this.msg = msg;
	}

	public HyException(int code, String msg, Throwable e) {
		super(msg, e);
		this.code = code;
		this.msg = msg;
	}

	public String getMsg() {
		return msg;
	}

	public void setMsg(String msg) {
		this.msg = msg;
	}

	public int getCode() {
		return code;
	}

	public void setCode(int code) {
		this.code = code;
	}
}
