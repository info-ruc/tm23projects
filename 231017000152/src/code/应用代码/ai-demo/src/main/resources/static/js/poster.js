var Poster = function(data, obsParams){
	//-----------开始------------//
	var posterContentId = data.posterContentId;//剪切框contentId
	var img = data.img;//处理后显示的图片id
	var ossPath = data.ossPath;//图片上传成功后保存的input隐藏域id
	var triggerBtn = data.triggerBtn;//触发弹框的标签id
	var tailorW = data.tailorW;//设置裁剪框宽
	var tailorH = data.tailorH;//设置裁剪框高
	var html = '<div style="display: none" class="tailoring-container">'+
	'  <div class="black-cloth closeTailor"></div>'+
	'  <div class="tailoring-content">'+
	'    <div class="tailoring-content-one">'+
	'      <label title="上传图片" for="chooseImg" class="l-btn choose-btn">'+
	'      <input type="file" class="choose-file" accept="image/jpg,image/jpeg,image/png" name="file" id="chooseImg" style="display: none;">选择图片</label>'+
	'      <div class="close-tailoring closeTailor">×</div>'+
	'    </div>'+
	'    <div class="tailoring-content-two">'+
	'      <div class="tailoring-box-parcel"><img class="tailoringImg"></div>'+
	'    </div>'+
	'    <div class="tailoring-content-three">'+
	'     <div style="padding-right: 2px;">'+
	'      	  <div style="float: left;"  >'+
	"    	    W: <input type=\"text\" class=\"frameWidth\" size=\"1\" style=\"margin-top: 5px;\">"+
	'    	  </div>'+
	'    	  <div style="float: left;" >'+
	" 	        H:<input type=\"text\" class=\"frameHeight\" size=\"1\" style=\"margin-top: 5px;\">"+
	' 	      </div>'+
	' 	      <div style="float: left;margin-left: 5px;">'+
	' 	           <button type="button" class="l-btn setWH" >设置</button>'+
	' 	      </div>'+
	'	 </div>'+
	'     <div style="float: left;margin-left: 5px;">'+
	'       <a class="l-btn cropper-reset-btn" style="cursor: pointer;">复位</a>'+
	'       <a class="l-btn cropper-rotate-btn" style="cursor: pointer;">旋转</a>'+
	'       <a class="l-btn cropper-scaleX-btn" style="cursor: pointer;">换向</a>'+
	'       <a class="l-btn sureCut" style="cursor: pointer;margin-left: 5px;">确定</a>'+
	'      </div>'+
	'    </div>'+
	'  </div>'+
	'</div>';
	$("#"+posterContentId).html(html);
	$("#"+triggerBtn).on("click",function() {
	    $("#"+posterContentId+" .tailoring-container").toggle();
	});
	$("#"+posterContentId+" .choose-btn").attr("for","chooseImg_"+posterContentId);
	$("#"+posterContentId+" .choose-file").attr("id","chooseImg_"+posterContentId);
	//获取当前frame宽高
	var frameHeight = 0;
	var frameWidth = 0;
	var dom = document;
	var str = "";
	for(var i=0;i<10;i++){
		var c = $(dom).find(".layui-side");
		if(c.length>0){
			break;
		}else{
			str += "parent."
			dom = eval("("+str+"document)");
		}
	}
	frameHeight = $(dom).find(".layui-side").height()-180;
	frameWidth = $(dom).find(".layui-header").width()-200;
	//alert(frameHeight+"======="+frameWidth)
	var ml = parseInt((frameWidth - 560)/2);
	var mt = parseInt((frameHeight - 540)/2);
	$("#"+posterContentId+" .tailoring-content").css({"margin-left":ml+"px","margin-top":mt+"px"})
	//图像上传
	$("#chooseImg_"+posterContentId).on("change",function(){
		if (!$(this).prop('files')) {
	        return;
	    }
	    var reader = new FileReader();
	    reader.onload = function(evt) {
	        var replaceSrc = evt.target.result;
	        //更换cropper的图片
	        $("#"+posterContentId+' .tailoringImg').cropper('replace', replaceSrc, false); //默认false，适应高度，不失真
	    }
	    file=$(this).prop('files')[0];
	    if($(this).prop('files')[0].size > 3*1024*1024){
	    	top.layer.msg("为了保障用户体验，海报图片大小不能大于3M");
	    	return;
	    }
	    reader.readAsDataURL($(this).prop('files')[0]);
	});
	//cropper图片裁剪
	$("#"+posterContentId+' .tailoringImg').cropper({
	    //aspectRatio: 16 / 9,
	    preview: '.previewImg',
	    //预览视图
	    guides: false,
	    //裁剪框的虚线(九宫格)
	    autoCropArea: 1,
	    //0-1之间的数值，定义自动剪裁区域的大小，默认0.8
	    dragCrop: true,
	    //是否允许移除当前的剪裁框，并通过拖动来新建一个剪裁框区域
	    movable: true,
	    //是否允许移动剪裁框
	    resizable: true,
	    //是否允许改变裁剪框的大小
	    zoomable: true,
	    //是否允许缩放图片大小
	    mouseWheelZoom: false,
	    //是否允许通过鼠标滚轮来缩放图片
	    touchDragZoom: true,
	    //是否允许通过触摸移动来缩放图片
	    rotatable: true,
	    //是否允许旋转图片
	    crop: function(e) {
	        // 输出结果数据裁剪图像。
	    	var data = e;
	    	$("#"+posterContentId+" .frameWidth").val(parseInt(data.width));
			$("#"+posterContentId+" .frameHeight").val(parseInt(data.height));
	    },
	    ready:function(){
			var obj = $("#"+posterContentId+' .tailoringImg').cropper("getImageData");
			if(data.tailorW==undefined || data.tailorW==null || data.tailorW=='' || !typeof(data.tailorW)===Number){
				tailorW= obj.width;
				tailorH = obj.height;
			}
			if(tailorW < obj.width){
				tailorW = obj.width;
			}
			if(tailorH < obj.height){
				tailorH = obj.height;
			}
			$("#"+posterContentId+" .frameWidth").val(parseInt(tailorW));
			$("#"+posterContentId+" .frameHeight").val(parseInt(tailorH));
			$("#"+posterContentId+' .tailoringImg').cropper("setCropBoxData",{width:tailorW,height:tailorH});
	    }
	});
	//旋转
	$("#"+posterContentId+" .cropper-rotate-btn").on("click",
	function() {
	    $("#"+posterContentId+' .tailoringImg').cropper("rotate", 45);
	});
	//复位
	$("#"+posterContentId+" .cropper-reset-btn").on("click",function() {
	    $("#"+posterContentId+' .tailoringImg').cropper("reset");
		$("#"+posterContentId+' .tailoringImg').cropper("setCropBoxData",{width:tailorW,height:tailorH});
	});
	//换向
	var flagX = true;
	$("#"+posterContentId+" .cropper-scaleX-btn").on("click",
	function() {
	    if (flagX) {
	        $("#"+posterContentId+' .tailoringImg').cropper("scaleX", -1);
	        flagX = false;
	    } else {
	        $("#"+posterContentId+' .tailoringImg').cropper("scaleX", 1);
	        flagX = true;
	    }
	    flagX != flagX;
	});

	//裁剪后的处理
	$("#"+posterContentId+" .sureCut").on("click",function() {
	    function toBlob(urlData, fileType) {
	        var bytes = window.atob(urlData),
	        n = bytes.length,
	        u8arr = new Uint8Array(n);
	        while (n--) {
	            u8arr[n] = bytes.charCodeAt(n);
	        }
	        return new Blob([u8arr], {
	            type: fileType
	        });
	    }
	    if ($("#"+posterContentId+" .tailoringImg").attr("src") == null) {
	        return false;
	    } else {
	        var cas = $("#"+posterContentId+' .tailoringImg').cropper('getCroppedCanvas'); //获取被裁剪后的canvas
	        var base64url = cas.toDataURL('image/png'); //转换为base64地址形式
	        //图像数据 (e.g. data:image/png;base64,iVBOR...yssDuN70DiAAAAABJRU5ErkJggg==)
	        var dataUrl = base64url;
	        var base64 = dataUrl.split(',')[1];
	        var fileType = dataUrl.split(';')[0].split(':')[1];
	        // base64转blob
	        var blob = toBlob(base64, fileType);
	        // blob转arrayBuffer

	        var reader = new FileReader();
	        reader.readAsArrayBuffer(blob);
	        var idx = layer.msg('上传中，请稍候',{icon: 16,time:false,shade:0.8});
	        reader.onload = function(event) {
	            // 配置
	            var obsClient = new ObsClient({
	            	access_key_id: obsParams.ak, 
	            	secret_access_key: obsParams.sk,
	            	security_token: obsParams.securitytoken,
	            	server : obsParams.server
	            });
	            // 文件名
	            var storeAs = "poster/" + new Date().getTime() + '_' + Math.round(Math.random() * 10000) + "." + blob.type.split('/')[1];
	            // arrayBuffer转Buffer
	            // 上传
	            obsClient.putObject({
	            	Bucket : obsParams.bucket,
	            	Key : storeAs,
	            	SourceFile  : blob
	            },function (err, result) {
	            	if(err){
	            		console.error('Error-->' + err);
	            	}else{
	            		if(result.CommonMsg.Status < 300){
	            			layer.close(idx);
	    	            	layer.msg("上传成功");  
	            			$("#"+ossPath).val(storeAs);
	     	                $("#"+img).attr("src", obsParams.cdn+storeAs); //显示为图片的形式
	            		}else{
	            			layer.close(idx);
	    	                layer.msg("上传失败!");
	            			console.log('Code-->' + result.CommonMsg.Code);
	            			console.log('Message-->' + result.CommonMsg.Message);
	            		}
	            	}
	            });

	            /*client.put(storeAs, buffer).then(function(result) {
	            	layer.close(idx);
	            	layer.msg("上传成功");  
	                $("#"+ossPath).val(result.name);
	                $("#"+img).attr("src", "http://osshanyatemp.oss-cn-hangzhou.aliyuncs.com/"+result.name); //显示为图片的形式
	            }).catch(function(err) {
	                console.log(err);
	                layer.close(idx);
	                layer.msg("上传失败!");
	            });*/
	        }
	        //关闭裁剪框
	        $("#"+posterContentId+" .tailoring-container").toggle();
	    }
	});

	//关闭裁剪框
	$("#"+posterContentId+" .closeTailor").on('click',function(){
		$("#"+posterContentId+" .tailoring-container").toggle();
	})
	$("#"+posterContentId+" .setWH").on("click",function(){
		var widthV=parseInt($("#"+posterContentId+" .frameWidth").val());
		var heightV=parseInt($("#"+posterContentId+" .frameHeight").val());
		$("#"+posterContentId+' .tailoringImg').cropper("setCropBoxData",{width:widthV,height:heightV});
	})
}