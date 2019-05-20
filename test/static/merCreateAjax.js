/**
 * to submit the merchants binding groups information
 * Created by Administrator on 2018/8/15.
 */
$(document).ready(function () {
    $("#agreementSub").click(function () {
        var bedrooms = $("#bedrooms").val();
        var bathrooms = $("#bathrooms").val();
        var floors = $("#floors").val();
        var waterfront = $("#waterfront").val();
        var condition = $("#condition").val();
        var grade = $("#grade").val();
        var sqft_basement = $("#sqft_basement").val();
        var yr_built = $("#yr_built").val();
        var zipcode = $("#zipcode").val();
        var string =  bedrooms + ","  + bathrooms + ","  + floors + "," + waterfront + ","  + condition
        + ","  + grade + ","  + sqft_basement + "," + yr_built + ","  + zipcode;

        // var string = "[{ merName:'" + merName + "'," + "mobile:'" + mobile + "'," + "email:'" + email + "'," +
        //     "address:'" + address + "'," + "oauthType:'" + oauthType + "'}]";
        $.ajax(      //ajax方式提交表单
            {
                url: '/merchants/createMer',
                type: 'POST',
                dataType: 'text',
                data: {
                    userMer: string
                },
                success: function (message) {
                    if(message=="success"){
                        alert("创建成功");
                        window.location.href="/merchants/queryAllMers";
                    }else{
                        alert(message)
                        window.location.reload();
                    }


                    // window.open("queryAllMers");
                },
                clearForm: false,//禁止清楚表单
                resetForm: false //禁止重置表单
            });
    });
})