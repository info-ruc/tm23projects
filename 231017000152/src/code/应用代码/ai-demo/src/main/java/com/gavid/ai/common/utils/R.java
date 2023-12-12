package com.gavid.ai.common.utils;

import java.util.HashMap;
import java.util.Map;

import com.gavid.ai.common.enums.HyCodeEnum;

public class R extends HashMap<String, Object> {
	private static final long serialVersionUID = 1L;

	public R() {
		put("code", HyCodeEnum.SUCCESS.getCode());
		put("msg", HyCodeEnum.SUCCESS.getMsg());
	}

	public static R error() {
		return error(HyCodeEnum.ERROR.getCode(), HyCodeEnum.ERROR.getMsg());
	}

	public static R error(String msg) {
		return error(HyCodeEnum.ERROR.getCode(), msg);
	}

	public static R error(int code, String msg) {
		R r = new R();
		r.put("code", code);
		r.put("msg", msg);
		return r;
	}

	public static R ok(String msg) {
		R r = new R();
		r.put("msg", msg);
		return r;
	}

	public static R ok(Object list, long total) {
		R r = new R();
		r.put("list", list);
		r.put("total", total);
		return r;
	}

	public static R correct(Object data, long count) {
		R r = new R();
		r.put("data", data);
		r.put("count", count);
		return r;
	}
	
	public static R ok(String fieldName, Object fieldValue) {
		R r = new R();
		r.put(fieldName, fieldValue);
		return r;
	}

	public static R ok(Object list, long total, long totalPage, long currPage) {
		R r = new R();
		r.put("list", list);
		r.put("total", total);
		r.put("totalPage", totalPage);
		r.put("currPage", currPage);
		return r;
	}

	public static R ok(Object list, long currPage, long pageSize, long totalPage, long total) {
		R r = new R();
		r.put("data", list);
		r.put("currPage", currPage);
		r.put("pageSize", pageSize);
		r.put("totalPage", totalPage);
		r.put("total", total);
		return r;
	}

	public static R ok(Map<String, Object> map) {
		R r = new R();
		r.putAll(map);
		return r;
	}

	public static R ok(Object obj) {
		R r = new R();
		r.put("data", obj);
		return r;
	}

	public static R ok() {
		return new R();
	}

	@Override
	public R put(String key, Object value) {
		super.put(key, value);
		return this;
	}
}
