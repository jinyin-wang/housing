$(function() {
$('#agreementSub').click(function (){
	e.preventDefault();

    $.ajax({
        url:"/predictprice",
        type:'POST',
        data: $("#inputform").serialize(),   // 这个序列化传递很重要
        headers:{

        },
        success:function (resp) {
            // window.location.href = "/admin/page";
            print(resp);
            $("#price").html(resp)
            if(resp.error){
                console.log(resp.errmsg);
            }
        }
    });
});
});

function get_ajax() {
    var bedrooms = $("#bedrooms").val();
    var bathrooms = $("#bathrooms").val();
    var sqft_living = $("#sqft_living").val();
    var sqft_lot = $("#sqft_lot").val();
    var floors = $("#floors").val();
    var waterfront = $("#waterfront").val();
    var condition = $("#condition").val();
    var grade = $("#grade").val();
    var sqft_basement = $("#sqft_basement").val();
    var yr_built = $("#yr_built").val();
    var zipcode = $("#zipcode").val();
    var string =  bedrooms + ","  + bathrooms +","  + sqft_living +","  + sqft_lot + ","  + floors + "," + waterfront + ","  + condition
    + ","  + grade + ","  + sqft_basement + "," + yr_built + ","  + zipcode;
    var data = {
        "bedrooms": bathrooms,
        "bathrooms": bathrooms,
        "floors": floors,
        "waterfront": waterfront,
        "condition": condition,
        "grade": grade,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "zipcode": zipcode,
    };
    alert(string)
    $.ajax({
	  type: 'POST',
	  url: '/predictprice',
      contentType: 'application/json; charset=utf-8',
        dataType: 'text',
        data: string,
	  xhrFields: {
		withCredentials: false
	  },
	  headers: {
	  },
	  success: function(result) {

          if(result) {
              //这里数据不只是一个预测值，还有推荐值
              alert("success")
		     $("#price").text(result)
		    }
	  },
	});
}