package com.gavid.ai.common.xfyun.spark;

import java.io.IOException;

import javax.servlet.AsyncEvent;
import javax.servlet.AsyncListener;
import javax.servlet.http.HttpServletResponse;

import org.springframework.stereotype.Component;

import com.alibaba.fastjson.JSON;
import com.gavid.ai.common.enums.HyCodeEnum;
import com.gavid.ai.common.utils.R;
import com.gavid.ai.common.utils.ResponseUtils;

@Component
public class AsyncContextListener implements AsyncListener {

    @Override
    public void onComplete(AsyncEvent asyncEvent) throws IOException {
      // 异步任务释放资源
    }

    @Override
    public void onTimeout(AsyncEvent asyncEvent) throws IOException {
        HttpServletResponse response = (HttpServletResponse) asyncEvent.getAsyncContext().getResponse();
        if (response != null) {
            // 响应超时
          R error = R.error(HyCodeEnum.SERVICE_RESPONSE_TIMEOUT.getCode(),
                  HyCodeEnum.SERVICE_RESPONSE_TIMEOUT.getMsg());
          ResponseUtils.write(response, JSON.toJSONString(error));
        }
    }

    @Override
    public void onError(AsyncEvent asyncEvent) throws IOException {
      HttpServletResponse response = (HttpServletResponse) asyncEvent.getAsyncContext().getResponse();
      if (response != null) {
        // 响应超时
        R error = R.error(HyCodeEnum.ERROR.getCode(),
                HyCodeEnum.ERROR.getMsg());
        ResponseUtils.write(response, JSON.toJSONString(error));
      }
    }

    @Override
    public void onStartAsync(AsyncEvent asyncEvent) throws IOException {
         // 异步任务开始
    }
}
