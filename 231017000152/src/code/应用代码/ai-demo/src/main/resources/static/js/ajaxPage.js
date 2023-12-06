/**
 * <p>
 * 这是一个异步请求支持JSON数据的分页表格插件
 * </p>
 * this is plugin of synchronize request table ， can page and support JOSN data.
 * 
 * @auth Ron
 * @param $
 *            v1.0
 */
// 创建一个闭包
(function($) {
	var v_url = '#';
	var v_condition = {};
	var v_tmp_condition = {};
	var v_data_mapping = {};
	var $table = "table";
	var v_details_arry = [];
	var v_data = [];
	var v_currPage = 1;
    var successFun = null;
	// 插件的定义
	$.fn.ajaxPage = function(options) {
		$table = this;
		// get specific parameter to opts
		var opts = $.extend({}, $.fn.ajaxPage.defaults, options);
		// window.console.log(JSON.stringify(opts));
		v_tmp_condition = opts.condition;
		v_details_arry = [];
		v_data_mapping = opts.data_mapping;
        successFun = opts.successFun;
		buildHeader(v_data_mapping);
		// debug(JSON.stringify(v_tmp_condition));
		v_url = opts.url;
		ajaxReqestData(opts.currPage);
		return null;
	};
	function buildCondtion(condition) {
		var r_data = {};
		for (var i = 0, l = condition.length; i < l; i++) {
			// debug("aaa="+$('#'+condition[i]).val());
			r_data[condition[i]] = $('#' + condition[i]).val();
		}
		return r_data;
	}
	;
	// build header ,have a bug ,multiple table thead , it take effect all
	// table.
	// 多个表格，有thead 都会有效
	function buildHeader(dataMapping) {
		$($table).find('thead tr').remove();// 删除之前的数据
		var th = '<tr>';
		for (j = 0; j < dataMapping.length; j++) {
			colObj = dataMapping[j];
			if (colObj.type == 'checkbox') {
				th += '<th width="20px;"> <input type="checkbox" id="input-select-all">' + '</th>';
			}
			if (colObj.type == 'text' || colObj.type == 'dict' || colObj.type == 'custom' || colObj.type == 'oper') {
				var width = (colObj.width != undefined && typeof colObj.width != "undefined") ? 'width=' + colObj.width : '';
				th += '<th ' + width + ' >' + colObj.colname + '</th>';
			}
		}
		th += '</tr>';
		$($table).find('thead').append(th);
	}
	// build table tbody data
	function buildData(data, dataMapping) {
		v_data = data.list;
		v_details_arry = [];
		var td = '';
		if (data==undefined || data.list==undefined || data.list.length <= 0) {
			td = '<tr><td style="color:red;" colspan=' + dataMapping.length + ' >没有数据</td></tr>'
		}else{
			for (i = 0; i < data.list.length; i++) {
				// debug(JSON.stringify(data.list[i]));
				// debug(JSON.stringify("dataMapping="+JSON.stringify(dataMapping)));
				td += '<tr> <input  rowidx  type="hidden" value="' + i + '" />';

				var colDatas = [];
				var details = {};
				var primary = "";
				for (j = 0; j < dataMapping.length; j++) {
					colObj = dataMapping[j];
					var realData = data.list[i][colObj.data];
					realData = (realData == null || realData == undefined) ? '' : realData;
					var disBtn = "";
					var tdData = "";
					var tdStyle = "";
					if (colObj.primary == "yes") {
						primary = realData;
						td += '<input  primary  type="hidden" value="' + primary + '" />';
					}
					if (colObj.style != undefined && typeof colObj.style != "undefined") {
						tdStyle = colObj.style;
					}
					switch (colObj.type) {
					case 'checkbox':
						tdData = '<input name="' + colObj.data + '" type="checkbox" value="' + realData + '"/>';
						break;
					case 'text':
						tdData = realData;
						break;
					case 'dict':
						if (colObj.dict != undefined && typeof colObj.dict != "undefined") {
							json = colObj.dict;
							for ( var key in json) {
								if (key == realData) {
									realData = json[key];
									break;
								}
							}
						}

						if (colObj.btn != undefined && typeof colObj.btn != "undefined") {
							btn_json = colObj.btn;
							for ( var btn_key in btn_json) {
								if (btn_key == data.list[i][colObj.data]) {
									disBtn = btn_json[key];
									break;
								}
							}
						}
						tdData = realData + disBtn;
						break;
					case 'oper':
						tdData = colObj.data;
						break;
					case 'custom':
						if (colObj.custom != undefined && typeof colObj.custom != "undefined") {
							if(typeof colObj.custom == 'function'){
								realData =  colObj.custom(data.list[i]);
							}else{
								realData = (colObj.custom).replace("#(data)", realData);
							}
						}
						tdData = realData;
						break;
					case 'tt':
						tdData = function(realData) {
						
						};
						break;
					}
					// detail不加到表格
					if (colObj.type != 'detail') {
						td += '<td style="' + tdStyle + '">' + tdData + '</td>';
					}
					// 所有数据都将加到detail
					if (colObj.type != 'oper') {
						var colData = {};
						colData["colname"] = colObj.colname;
						colData["value"] = realData;
						colDatas.push(colData);
					}
				}
				td += '</tr>';
				// details
				details["unique"] = primary;
				details["data"] = colDatas;
				v_details_arry.push(details);
			}
		}
		$($table).find('tbody tr').remove();// 删除之前的数据
		$($table).find('tbody').append(td);
        if(typeof successFun == 'function') {
            successFun();
        }
		try{
			var s = new Dropdown;
		    s.render();
		}catch(err){
			console.log(err);
		}
	}
	;
	// 私有函数：debugging
	function debug(msg) {
		if (window.console && window.console.log)
			window.console.log('debug msg: ' + msg);
	}
	;
	// render iCheck box event
	function renderIcheck() {
		// here $ use lyaui.jquery
		var $ = layui.jquery;
		if ($('input').iCheck != undefined && typeof $('input').iCheck != "undefined") {
			$('input').iCheck({
				checkboxClass : 'icheckbox_flat-green'
			});
			$('#input-select-all').iCheck(event.currentTarget.checked ? 'check' : 'uncheck');
			$('#input-select-all').on('ifChanged', function(event) {
				var $input = $($table).find('tbody tr td input');
				$input.iCheck(event.currentTarget.checked ? 'check' : 'uncheck');
			});
		}
	};
	// 插件的defaults
	$.fn.ajaxPage.defaults = {
		url : '#',
		currPage : '1',
		condition : {},
		data_mapping : {}
	};
	// this is asynchronize requeset of javascript page plugin ,here was
	// initiated.
	function pageInit(page_total, curr, total_num) {
		layui.use([ 'laypage' ], function() {
			var laypage = layui.laypage;
			laypage.render({
			  elem: 'page'
			  ,curr : curr
			  ,count: total_num //数据总数，从服务端得到
			  ,skip : true
			  ,groups : 5 // 连续显示分页数
			  ,jump: function(obj, first){
				// 得到了当前页，用于向服务端请求对应数据
				var curr = obj.curr;
				v_currPage = curr;
				if (!first) {
					ajaxReqestData(curr);
					layer.msg('第 ' + obj.curr + ' 页');
				}
			  }
			});
		});
	}
	;
	// asynchronize reqeuest data from web server
	function ajaxReqestData(curr) {
		v_condition = buildCondtion(v_tmp_condition);
		v_condition['currPage'] = curr;
		$.ajax({
			type : "post",
			dataType : "json",
			url : v_url,
			data : v_condition,
			success : function(data) {
				// layer.msg(curr+"==="+data.total+"==="+data.totalPage);
				if(curr>1&&curr>data.totalPage){
					ajaxReqestData(data.totalPage);
				}
				pageInit(data.totalPage, curr, data.total);
				buildData(data, v_data_mapping);
				renderIcheck();
			},
			error : function(xhr, textStatus, errorThrown) {
				console.log("进入error---");
		　　　　　　　　console.log("状态码："+xhr.status);
		　　　　　　　　console.log("状态:"+xhr.readyState);//当前状态,0-未初始化，1-正在载入，2-已经载入，3-数据进行交互，4-完成。
		　　　　　　　　console.log("错误信息:"+xhr.statusText );
		　　　　　　　　console.log("返回响应信息："+xhr.responseText );//这里是详细的信息
		　　　　　　　　console.log("请求状态："+textStatus); 　　　　　　　　
		　　　　　　　　console.log(errorThrown); 　　　　　　　　
		　　　　　　　　console.log("请求失败"); 
				console.log("查询请求失败")
			}
		});
	}
	;
	$.fn.ajaxPage.refresh = function(cp) {
		if(cp!=undefined){
			v_currPage = cp;
		}
		ajaxReqestData(v_currPage);
	};
	// 获取当前操作的唯一主键 get current unique primary value
	$.fn.ajaxPage.getPrimaryID = function(idx) {
		var id = $(idx).parent().parent().find("input[primary]").val();
		return id;
	};
	// 获取当前行所有数据
	$.fn.ajaxPage.getRowData = function(rowid) {
		return v_data[rowid];
	};
	// 获取当前行号0开始
	$.fn.ajaxPage.getRowID = function(idx) {
		var rowid = $(idx).parent().parent().find("input[rowidx]").val();
		return rowid;
	};
	// 获取当前页选择的ID get current page all already seleced ids.
	$.fn.ajaxPage.getCheckSelected = function() {
		var str = ""
		var ids = "";
		var $input = $($table).find('tbody tr td input');
		$input.each(function() {
			if (true == $(this).is(':checked')) {
				str += $(this).val() + ",";
			}
		});
		if (str.substr(str.length - 1) == ',') {
			ids = str.substr(0, str.length - 1);
		}
		return ids;
	};
	$.fn.ajaxPage.getCheckedData = function() {
		var row = 0;
		var checkedData = [];
		var $input = $($table).find('tbody tr td input');
		$input.each(function() {
			if (true == $(this).is(':checked')) {
				checkedData.push(v_data[row]);
			}
			row++;
		});
		return checkedData;
	};
	$.fn.ajaxPage.getDetails = function(unique_idx) {
		mydata = [];
		flag = true;
		for (var i = 0, l = v_details_arry.length; flag && i < l; i++) {
			for ( var key in v_details_arry[i]) {
				// debug(key+":"+v_details_arry[i][key]);
				// 找到
				if ("unique" == key && unique_idx == v_details_arry[i][key]) {
					mydata = v_details_arry[i]["data"];
					flag = false;
					break;
				}
			}
		}
		s = '<div class="layui-field-box"><div>';
		for (var i = 0, l = mydata.length; i < l; i++) {
			b = '';
			for ( var k in mydata[i]) {
				// debug(JSON.stringify(v_data[i]));

				b = '<div class="layui-form-item"><label class="layui-form-label">' + mydata[i].colname + '</label>';
				b += '<div class="layui-input-block">';
				// b+='<input class="layui-input"
				// value="'+mydata[i].value+'"/>';
				if (mydata[i].value.length > 60) {
					b += '<textarea  class="layui-textarea" >' + mydata[i].value + '</textarea>';
				} else {
					b += '<label class="layui-input" >' + mydata[i].value + '</label>';
				}
				b += '</div></div>';
			}
			s += b;
		}
		s += '</div></div>';
		return s;
	};
	// 闭包结束
})(jQuery);