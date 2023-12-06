package com.gavid.ai.common.exception.handler;

import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.ConversionNotSupportedException;
import org.springframework.beans.TypeMismatchException;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.http.converter.HttpMessageNotWritableException;
import org.springframework.web.HttpMediaTypeNotAcceptableException;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;

import com.gavid.ai.common.enums.HyCodeEnum;
import com.gavid.ai.common.exception.HyException;
import com.gavid.ai.common.utils.R;


/**  
* @Title: HyExceptionHandler.java  
* @package com.gavid.bjggwhy.common.handler
* @Description: 自定义异常拦截
* @version V1.0  
*/
@CrossOrigin
@ControllerAdvice
@ResponseBody
public class HyExceptionHandler {
	private static final Logger log = LoggerFactory.getLogger(HyExceptionHandler.class);
	
    /**  
    * @Title: runtimeExceptionHandler  
    * @Description: 运行时异常  
    * @param @param ex
    * @param @return    参数  
    * @return R    返回类型  
    */ 
    @ExceptionHandler(RuntimeException.class)
    public R runtimeExceptionHandler(RuntimeException ex) {
    	log.error("运行时异常",ex);
        return R.error(HyCodeEnum.ERROR.getCode(), HyCodeEnum.ERROR.getMsg());
    }
    
	@ExceptionHandler(HyException.class)
	public R handler(HyException e) {
		log.error("运行时异常",e);
		return R.error(e.getCode(), e.getMsg());
	}

    /**  
    * @Title: nullPointerExceptionHandler  
    * @Description: 空指针异常  
    * @param @param ex
    * @param @return    参数  
    * @return R    返回类型  
    */ 
    @ExceptionHandler(NullPointerException.class)
    public R nullPointerExceptionHandler(NullPointerException ex) {
        log.error("空指针异常",ex);
        return R.error(HyCodeEnum.NOT_PARAMETER.getCode(), HyCodeEnum.NOT_PARAMETER.getMsg());
    }

    /**  
    * @Title: classCastExceptionHandler  
    * @Description: 类型转换异常  
    * @param @param ex
    * @param @return    参数  
    * @return R    返回类型  
    */ 
    @ExceptionHandler(ClassCastException.class)
    public R classCastExceptionHandler(ClassCastException ex) {
        log.error("类型转换异常",ex);
        return R.error(HyCodeEnum.ERROR_PARAMETER.getCode(), HyCodeEnum.ERROR_PARAMETER.getMsg());
    }

    /**  
    * @Title: iOExceptionHandler  
    * @Description: IO异常  
    * @param @param ex
    * @param @return    参数  
    * @return R    返回类型  
    */ 
    @ExceptionHandler(IOException.class)
    public R iOExceptionHandler(IOException ex) {
        log.error("IO异常",ex);
        return R.error(HyCodeEnum.ERROR.getCode(), HyCodeEnum.ERROR.getMsg());
    }

    /**  
    * @Title: noSuchMethodExceptionHandler  
    * @Description: 未知方法异常 
    * @param @param ex
    * @param @return    参数  
    * @return R    返回类型  
    */ 
    @ExceptionHandler(NoSuchMethodException.class)
    public R noSuchMethodExceptionHandler(NoSuchMethodException ex) {
        log.error("未知方法异常",ex);
        return R.error(HyCodeEnum.NOSUCHMETHOD_ERROR.getCode(), HyCodeEnum.NOSUCHMETHOD_ERROR.getMsg());
    }
    
    /**  
    * @Title: indexOutOfBoundsExceptionHandler  
    * @Description: 数组越界异常  
    * @param @param ex
    * @param @return    参数  
    * @return R    返回类型  
    */ 
    @ExceptionHandler(IndexOutOfBoundsException.class)
    public R indexOutOfBoundsExceptionHandler(IndexOutOfBoundsException ex) {
        log.error("数组越界异常",ex);
        return R.error(HyCodeEnum.ERROR.getCode(), HyCodeEnum.ERROR.getMsg());
    }

    /**  
    * @Title: requestNotReadable  
    * @Description: 400错误 
    * @param @param ex
    * @param @return    参数  
    * @return R    返回类型  
    */ 
    @ExceptionHandler({HttpMessageNotReadableException.class})
    public R requestNotReadable(HttpMessageNotReadableException ex) {
        log.error("400..requestNotReadable");
        return exceptionFormat(ex);
    }

    /**  
    * @Title: requestTypeMismatch  
    * @Description: 400错误  
    * @param @param ex
    * @param @return    参数  
    * @return R    返回类型  
    */ 
    @ExceptionHandler({TypeMismatchException.class})
    public R requestTypeMismatch(TypeMismatchException ex) {
        log.error("400..TypeMismatchException");
        return exceptionFormat(ex);
    }

    /**  
    * @Title: requestMissingServletRequest  
    * @Description: 400错误  
    * @param @param ex
    * @param @return    参数  
    * @return R    返回类型  
    */ 
    @ExceptionHandler({MissingServletRequestParameterException.class})
    public R requestMissingServletRequest(MissingServletRequestParameterException ex) {
        log.error("400..MissingServletRequest");
        return exceptionFormat(ex);
    }

    /**  
    * @Title: request405  
    * @Description: 405错误
    * @param @param ex
    * @param @return    参数  
    * @return R    返回类型  
    */ 
    @ExceptionHandler({HttpRequestMethodNotSupportedException.class})
    public R request405(HttpRequestMethodNotSupportedException ex) {
        log.error("405...异常",ex);
        return exceptionFormat(ex);
    }

    /**  
    * @Title: request406  
    * @Description: 406错误  
    * @param @param ex
    * @param @return    参数  
    * @return R    返回类型  
    */ 
    @ExceptionHandler({HttpMediaTypeNotAcceptableException.class})
    public R request406(HttpMediaTypeNotAcceptableException ex) {
        log.error("406...",ex);
        return exceptionFormat(ex);
    }

    /**  
    * @Title: server500  
    * @Description: 500错误  
    * @param @param ex
    * @param @return    参数  
    * @return R    返回类型  
    */ 
    @ExceptionHandler({ConversionNotSupportedException.class, HttpMessageNotWritableException.class})
    public R server500(RuntimeException ex) {
        log.error("500...",ex);
        return exceptionFormat(ex);
    }

    /**  
    * @Title: requestStackOverflow  
    * @Description: 栈溢出  
    * @param @param ex
    * @param @return    参数  
    * @return R    返回类型  
    */ 
    @ExceptionHandler({StackOverflowError.class})
    public R requestStackOverflow(StackOverflowError ex) {
        log.error("栈溢出异常",ex);
        return exceptionFormat(ex);
    }

    /**  
    * @Title: exception  
    * @Description: 其他错误  
    * @param @param ex
    * @param @return    参数  
    * @return R    返回类型  
    */ 
    @ExceptionHandler({Exception.class})
    public R exception(Exception ex) {
        log.error("其他异常",ex);
        return exceptionFormat(ex);
    }

    private <T extends Throwable> R exceptionFormat(T ex) {
        ex.getMessage();
        return R.error(HyCodeEnum.ERROR.getCode(), HyCodeEnum.ERROR.getMsg());
    }

}
