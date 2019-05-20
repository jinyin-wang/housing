/**
 * Created by Administrator on 2019/5/13.
 */
/**
 * Created by Administrator on 2018/8/15.
 */
$(document).ready(function () {
        // draw Graph by Echarts of the main page, get the info of all counties
        var schoolChart = echarts.init(document.getElementById('schools'));
        var schoolOption = {
            title: {
                text: 'poverty rate of school'
            },
            tooltip: {},
            legend: {
                data:['%']
            },
            xAxis: {
                data: ["avg","underhi","high","colleg","bachelor"]
            },
            yAxis: {},
            series: [{
                name: 'poverty~education',
                type: 'bar',
                data: [16.48, 26.23, 14.02, 10.89, 4.55]
            }]
        };
        schoolChart.setOption(schoolOption);


        var genderChart = echarts.init(document.getElementById('genders'));
        var genderOption = {
            title: {
                text: 'poverty rate of genders'
            },
            tooltip: {},
            legend: {
                data:['rate']
            },
            xAxis: {
                data: ["average","male","female"]
            },
            yAxis: {},
            series: [{
                name: 'poverty~genders',
                type: 'bar',
                data: [16.48, 14.94, 17.96]
            }]
        };
        genderChart.setOption(genderOption);


        var factorsChart = echarts.init(document.getElementById('factors'));

        var factorsOption = {
            title : {
                text: 'biggest factor',
                x:'center'
            },
            tooltip : {
                trigger: 'item',
                formatter: "{a} <br/>{b} : {c} ({d}%)"
            },
            toolbox: {
                show : true,
                feature : {
                    restore : {show: true},
                }
            },
            calculable : false,
            series : [
                {
                    name:'',
                    type:'pie',
                    radius : '55%',
                    center: ['50%', '60%'],
                    data:[
                        {value:0.088, name:'disability'},
                        {value:0.18, name:'avghouseage'},
                        {value:0.04, name:'partjob'},
                        {value:0.49, name:'nojob'},
                        {value:0.2, name:'avgfueluseage'},
                        {value:0.29, name:'fulljob'}]
                }
            ]
        };
        factorsChart.setOption(factorsOption);
})
