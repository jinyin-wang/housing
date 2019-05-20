var height = 200;
var width = 400;


$(document).ready(function () {
    get_ajax('/buildtime_price', true, false, 'buildtime_price', 'buildtime_price');
    get_ajax('/soldtime_price', true, false, 'soldtime_price', 'soldtime_price');

    get_ajax('/geo_price', true, false, 'geo_price', 'geo_price');
    get_ajax('/geo_rooms', true, false, 'geo_rooms', 'geo_rooms');

});


function get_ajax(url, isRandSamples, matrix, isScree, chart_title) {
    $.ajax({
        type: 'GET',
        url: url,
        contentType: 'application/json; charset=utf-8',
        xhrFields: {
            withCredentials: false
        },
        headers: {},
        success: function (result) {

            if (isScree == 'buildtime_price') {
                drawTimeLine(result, chart_title);
            } else if (isScree == 'geo_price') {
                drawGeo(result, chart_title)
            } else if(isScree == 'soldtime_price'){
                drawSoldtimePrice(result, chart_title)
            } else {
                drawGeoRooms (result, chart_title)
            }
        },
    });
}

function drawTimeLine(result, title) {
    var data = JSON.parse(result);
//  To get column names of most weighted attributes/columns
    var date = Object.keys(data);
    data = Object.values(data);
    var built_time_price = echarts.init(document.getElementById('built_time_price'));
    var option = {
        tooltip: {
            trigger: 'axis',
            position: function (pt) {
                return [pt[0], '10%'];
            }
        },
        title: {
            left: 'center',
            text: 'relationship between price and age of house ',
        },
        xAxis: {
            type: 'category',
            boundaryGap: false,
            data: date
        },
        yAxis: {
            type: 'value',
            boundaryGap: [0, '100%'],
            min: 300,
            max: 800
        },
        dataZoom: [{
            type: 'inside',
            start: 0,
            end: 10
        }, {
            start: 0,
            end: 10,
            handleIcon: 'M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
            handleSize: '70%',
            handleStyle: {
                color: '#fff',
                shadowBlur: 3,
                shadowColor: 'rgba(0, 0, 0, 0.6)',
                shadowOffsetX: 2,
                shadowOffsetY: 2
            }
        }],
        series: [
            {
                name: 'average data',
                type: 'line',
                smooth: true,
                symbol: 'none',
                sampling: 'average',
                itemStyle: {
                    color: 'rgb(255, 70, 131)'
                },
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                        offset: 0,
                        color: 'rgb(255, 158, 68)'
                    }, {
                        offset: 1,
                        color: 'rgb(255, 70, 131)'
                    }])
                },
                data: data
            }
        ]
    };
    built_time_price.setOption(option);
}

function drawGeoRooms(result, title) {
    var data = JSON.parse(result);
//  To get column names of most weighted attributes/columns
//     var date = Object.keys(data);

    data = Object.values(data);

    var date = Object.keys(data[0]);
    var data1 = Object.values(data[0]);
    var data2 = Object.values(data[1]);

    var geo_rooms = echarts.init(document.getElementById('geo_rooms'));
    var option = {
        tooltip: {
            trigger: 'axis',
            position: function (pt) {
                return [pt[0], '10%'];
            }
        },
        title: {
            left: 'center',
            text: 'amounts of houses and rooms sold in each region',
        },
        xAxis: {
            type: 'category',
            boundaryGap: false,
            data: date
        },
        yAxis: {
            type: 'value',
            boundaryGap: [0, '100%'],
            max: 2200

        },
        dataZoom: [{
            type: 'inside',
            start: 0,
            end: 10
        }, {
            start: 0,
            end: 10,
            handleIcon: 'M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
            handleSize: '70%',
            handleStyle: {
                color: '#fff',
                shadowBlur: 3,
                shadowColor: 'rgba(0, 0, 0, 0.6)',
                shadowOffsetX: 2,
                shadowOffsetY: 2
            }
        }],
        series: [
            {
                name: 'line1',
                type: 'bar',
                smooth: true,
                symbol: 'none',
                sampling: 'average',
                itemStyle: {
                    color: 'rgb(255, 70, 131)'
                },
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                        offset: 0,
                        color: 'rgb(255, 158, 68)'
                    }, {
                        offset: 1,
                        color: 'rgb(255, 70, 131)'
                    }])
                },
                data: data1
            },
            {
                name: 'line2',
                type: 'bar',
                smooth: true,
                symbol: 'none',
                sampling: 'average',
                itemStyle: {
                    color: 'rgb(255, 70, 131)'
                },
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                        offset: 0,
                        color: 'rgb(255, 158, 68)'
                    }, {
                        offset: 1,
                        color: 'rgb(255, 70, 131)'
                    }])
                },
                data: data2
            }
        ]
    };
    geo_rooms.setOption(option);
}
function drawSoldtimePrice(result, chart_title) {
    var data = JSON.parse(result);
//  To get column names of most weighted attributes/columns
//     date.push([now.getFullYear(), now.getMonth() + 1, now.getDate()].join('/'));

    var olddate = Object.keys(data);
    var date = [];
    for (var i = 0; i < olddate.length; i++) {
        var year = olddate[i].substring(0, 4);
        var month = olddate[i].substring(4, 6);
        var day = olddate[i].substring(6);

        date.push(year + '/' + month + '/' + day);
    }

    data = Object.values(data);
    var sold_time_price = echarts.init(document.getElementById('sold_time_price'));
    var option = {
        tooltip: {
            trigger: 'axis',
            position: function (pt) {
                return [pt[0], '10%'];
            }
        },
        title: {
            left: 'center',
            text: 'relationship between price and sold time ',
        },
        xAxis: {
            type: 'category',
            boundaryGap: false,
            data: date
        },
        yAxis: {
            type: 'value',
            boundaryGap: [0, '100%'],

        },
        dataZoom: [{
            type: 'inside',
            start: 0,
            end: 10
        }, {
            start: 0,
            end: 10,
            handleIcon: 'M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
            handleSize: '70%',
            handleStyle: {
                color: '#fff',
                shadowBlur: 3,
                shadowColor: 'rgba(0, 0, 0, 0.6)',
                shadowOffsetX: 2,
                shadowOffsetY: 2
            }
        }],
        series: [
            {
                name: 'average data',
                type: 'line',
                smooth: true,
                symbol: 'none',
                sampling: 'average',
                itemStyle: {
                    color: 'rgb(255, 70, 131)'
                },
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                        offset: 0,
                        color: 'rgb(255, 158, 68)'
                    }, {
                        offset: 1,
                        color: 'rgb(255, 70, 131)'
                    }])
                },
                data: data
            }
        ]
    };
    sold_time_price.setOption(option);
}

function drawGeo(result, title) {
    var data = JSON.parse(result);
//  To get column names of most weighted attributes/columns
    var geo = Object.keys(data);
    data = Object.values(data);
    var newgeo = []
    var newdata = [];
    for (var j = 0; j < geo.length; j++) {
        var dat = geo[j].substring(1, geo[j].length - 2)
        newgeo.push(dat);
    }
    for (var i = 0; i < geo.length; i++) {
        var geos = newgeo[i].split(",");
        var x = parseFloat(geos[0]);
        var y = parseFloat(geos[1]);

        newdata.push([x, y, data[i]]);
    }
    var drawGeo = echarts.init(document.getElementById('drawGeo'));
    var option = {
        tooltip: {
            trigger: 'axis',
            showDelay: 0,
            axisPointer: {
                show: true,
                type: 'cross',
                lineStyle: {
                    type: 'dashed',
                    width: 1
                }
            }
        },
        title: {
            left: 'center',
            text: 'relationship between price and geo ',
        },
        legend: {
            data: ['scatter1', 'scatter2']
        },
        // toolbox: {
        //     show: true,
        //     feature: {
        //         mark: {show: true},
        //         dataZoom: {show: true},
        //         dataView: {show: true, readOnly: false},
        //         restore: {show: true},
        //         saveAsImage: {show: true}
        //     }
        // },
        xAxis: [
            {
                type: 'value',
                splitNumber: 4,
                scale: true
            }
        ],
        yAxis: [
            {
                type: 'value',
                splitNumber: 4,
                scale: true
            }
        ],
        series: [
            {
                name: 'price',
                type: 'scatter',
                symbolSize: function (value) {
                    return Math.round(value[2] / 170);
                },
                data: newdata
            }
            // },
            // {
            //     name:'scatter2',
            //     type:'scatter',
            //     symbolSize: function (value){
            //         return Math.round(value[2] / 5);
            //     },
            //     data: n
            // }
        ]
    };
    drawGeo.setOption(option);

}




