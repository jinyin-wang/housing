/**
 * Created by Administrator on 2018/8/22.
 */
$(document).ready(function () {

    $("#agreementSub").click(function () {
        var time = $("#time").val();
        // time = time.substring(0,3) + time.substring(5,6) + time.substring(8,9);
        // document.getElementById("currentTime").innerHTML = time;
        var xAxisData = ["00时", "01时", "02时", "03时", "04时", "05时", "06时", "07时", "08时", "09时", "10时", "11时", "12时", "13时", "14时", "15时", "16时", "17时", "18时", "19时", "20时", "21时", "22时", "23时"];
        var myChart = echarts.init(document.getElementById('line'));
        // var string = "[{ merName:'" + merName + "'," + "mobile:'" + mobile + "'," + "email:'" + email + "'," +
        //     "address:'" + address + "'," + "oauthType:'" + oauthType + "'}]";
        $.ajax(      //ajax方式提交表单
            {
                url: '/report/getAggVisitHour',
                type: 'POST',
                dataType: 'text',
                data: {
                    time: time
                },
                success: function (data) {
                    // if(message=="success"){
                    //     alert("创建成功");
                    //     window.location.href="/report/getAggVisitHour";
                    // }else{
                    //     alert(message)
                    //     // window.location.reload();
                    // }
                    var jsonObj = JSON.parse( data );

                    // alert(dataTotal);
                    // alert(dataTotalFailed);
                    // alert(dataTotalAPI);

                    // window.open("queryAllMers");
                    // var myChart = echarts.init(document.getElementById('line'));
                    var dataTotal = jsonObj.dataTotal;
                    var dataTotalFailed = jsonObj.dataTotalFailed;
                    var dataTotalAPI = jsonObj.dataTotalAPI;
                    var barOption = {
                        title: {
                            text: '哗啦啦开放平台',
                            subtext: ''
                        },
                        tooltip: {
                            trigger: 'axis' //item 点在哪条线上显示哪条线上的数据，axis点在哪个坐标点上显示对于点上所有数据
                        },
                        legend: {
                            data: ['总访问次数','总访出错次数','api访问出错次数']
                        },
                        // toolbox: {
                        //     show: true,
                        //     orient: 'vertical',
                        //     x: 'right',
                        //     y: 'center',
                        //     feature: {
                        //         mark: {show: true},
                        //         dataView: {show: true, readOnly: false},
                        //         magicType: {show: true, type: ['line', 'bar']},
                        //         restore: {show: true},
                        //         saveAsImage: {
                        //             show: true,
                        //             name: '折线图'//保存的图片名次
                        //         }
                        //     }
                        // },
                        // calculable: true,
                        xAxis: [
                            {
                                name: '/24h',
                                boundaryGap: false,
                                data: xAxisData
                            }
                        ],
                        yAxis: [
                            {
                                name: time,
                                type: 'value'
                            }
                        ],
                        grid: {
                            left: '3%',
                            right: '4%',
                            bottom: '3%',
                            containLabel: true
                        },
                        series: [
                            {
                                name: '总访问次数',
                                type: 'bar',
                                data: dataTotal
                            },
                            {
                                name: '总访出错次数',
                                type: 'bar',
                                data: dataTotalFailed
                            },
                            {
                                name: 'api访问出错次数',
                                type: 'bar',
                                data: dataTotalAPI
                            }
                        ]
                    };
                    myChart.setOption(barOption);

                    // var ajax = function () {
                    //     $.ajax({
                    //         url: '/echarts/showImage',
                    //         success: function (responseText) {
                    //             //请求成功时处理
                    //             alert("shiiiiiiiiit");
                    //             var responseText = eval('(' + responseText + ')');
                    //             barOption.legend.data = responseText.legend;
                    //             barOption.xAxis[0].data = responseText.xAxis;
                    //             var serieslist = responseText.series;
                    //             //alert(serieslist);
                    //             for (var i = 0; i < serieslist.length; i++) {
                    //                 barOption.series[i] = serieslist[i];
                    //             }
                    //             //alert(lineOption.series);
                    //             myChart.setOption(barOption, true);
                    //         },
                    //         complete: function () {
                    //             //请求完成的处理
                    //         },
                    //         error: function () {
                    //             //请求出错处理
                    //             alert("加载失败");
                    //         }
                    //     })
                    // }

                    // window.setTimeout(ajax, 100);
                    // alert("sssss");
                },
                clearForm: false,//禁止清楚表单
                resetForm: false //禁止重置表单
            });
    });
})
//window.setInterval(ajax,1000*60*5);