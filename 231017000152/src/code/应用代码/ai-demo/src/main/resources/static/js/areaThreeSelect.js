var defaults = {
    selectProv: 'province',
    selectCity: 'city',
    selectCounty: 'county',
    valProv: null,
    valCity: null,
    valCounty: null
};
var $body;
var form;
var $;
layui.define(['jquery', 'form'], function () {
    $ = layui.jquery;
    form = layui.form;
    $body = $('body');
    treeSelect(defaults);
});
function treeSelect(config) {
    config.valProv = config.valProv ? config.valProv : '';
    config.valCity = config.valCity ? config.valCity : '';
    config.valCounty = config.valCounty ? config.valCounty : '';
    findSysAreaByParentId(0, function(ajaxData){
    	var threeSelectData = ajaxData.data;
		$.each(threeSelectData, function (k, v) {
			if(v.areaId == '0'){
				return;
			}
	        appendOptionTo($body.find('select[name=' + config.selectProv + ']'), v.areaId, v.areaName, config.valProv);
		    form.on('select(' + config.selectProv + ')', function (data) {
		    	if(data.value == ''){
		    		config.valProv = '';
		    	}
		        cityEvent(data);
		        form.on('select(' + config.selectCity + ')', function (data) {
		        	if(data.value == ''){
			    		config.valCity = '';
			    	}
		        	countyEvent(data);
		        });
		    });
	    });
		form.render();
		if(config.valProv != ''){//如果有默认值,加载城市
			cityEvent(config);
		}
		if(config.valCity != ''){//如果有默认值,加载区县
			countyEvent(config);
		}
    });
    
    function cityEvent(data) {
        $body.find('select[name=' + config.selectCity + ']').html("");
        config.valProv = data.value ? data.value : config.valProv;
        findSysAreaByParentId(config.valProv, function(ajaxData){
        	var threeSelectData = ajaxData.data;
        	$.each(threeSelectData, function (k, v) {
        		if (v.value == config.v1) {
        			appendOptionTo($body.find('select[name=' + config.selectCity + ']'), v.areaId, v.areaName, config.valCity);
        		}
            });
        	
            config.valCity = $('select[name=' + config.selectCity + ']').val();
            countyEvent(config);
        });
        
        
    }
    function countyEvent(data) {
        $body.find('select[name=' + config.selectCounty + ']').html("");
        config.valCity = data.value ? data.value : config.valCity;
        findSysAreaByParentId(config.valCity, function(ajaxData){
        	var threeSelectData = ajaxData.data;
        	$.each(threeSelectData, function (k, v) {
        		appendOptionTo($body.find('select[name=' + config.selectCounty + ']'), v.areaId, v.areaName, config.valCounty);
            });
            form.render();
            form.on('select(' + config.selectCounty + ')', function (data) { });
        });
        
    }
    function appendOptionTo($o, k, v, d) {
        var $opt = $("<option>").text(v).val(k);
        if (k == d) { $opt.attr("selected", "selected") }
        $opt.appendTo($o);
    }
}

function findSysAreaByParentId(parentId, callBackFunction) {
    $.ajax({
    	type: "POST",
    	url: "/sys/area/findByParentId",
    	data:{
    		"parentId":parentId
    	},
    	success: function(data) {
    		if(data.code == "0") {
    			if(typeof (callBackFunction) == 'function') {
    				callBackFunction(data);
    			}
    		} else {
		    	layer.msg(data.msg);
		    }
    	}
    });
}