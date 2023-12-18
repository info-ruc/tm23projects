package com.plm.pms;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

@RunWith(SpringRunner.class)
@SpringBootTest
@Ignore
public class PmsServiceTest {

    @Test
    public void testCreate() {
        String email = "zhi.yang@bitmain.com";
        String ssoId = "zhi.yang";
        String subAccount = "zhiyang";
        System.out.println(subAccount);
    }
}
