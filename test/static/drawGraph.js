/**
 * Created by Administrator on 2018/8/15.
 */
$(document).ready(function () {
    $("#getCounty").click(function (e) {
        // clear the data for the graph of the second page, get the info of one county
        var data = $("#data").val();
        var dataBfSplit = data.toString();
        var dataAf1Split = dataBfSplit.substring(11);
        var dataAf2Split = dataAf1Split.split(",");

        var countyname;
        var foodstamprate;
        var povertyrate;
        var malepovertyrate;
        var femalepovertyrate;
        var underhighschoolpovertyrate;
        var highschoolpovertyrate;
        var collegepovertyrate;
        var bachelorpovertyrate;
        var disabilitycorelation;
        var avghouseagecorelation;
        var avgfueluseagecorelation;
        var fulljobcorelation;
        var partjobcorelation;
        var nojobcorelation;
        var list = [];

        for (var i = 0; i < dataAf2Split.length; i++) {
            var temp = dataAf2Split[i].split("=");
            list[i] =  temp[1];
        }
        countyname = list[2];
        foodstamprate = parseFloat(list[3]);
        povertyrate = parseFloat(list[4]);
        malepovertyrate = parseFloat(list[5]);
        femalepovertyrate = parseFloat(list[6]);
        underhighschoolpovertyrate = parseFloat(list[7]);
        highschoolpovertyrate = parseFloat(list[8]);
        collegepovertyrate = parseFloat(list[9]);
        bachelorpovertyrate = parseFloat(list[10]);
        disabilitycorelation = Math.abs(parseFloat(list[11]));
        avghouseagecorelation = Math.abs(parseFloat(list[12]));
        avgfueluseagecorelation = Math.abs(parseFloat(list[13]));
        fulljobcorelation = Math.abs(parseFloat(list[14]));
        partjobcorelation = Math.abs(parseFloat(list[15]));
        nojobcorelation = Math.abs(parseFloat(list[16]));

        var sum = disabilitycorelation + avgfueluseagecorelation+ avghouseagecorelation+ fulljobcorelation+ partjobcorelation + nojobcorelation;
        var per_disability = disabilitycorelation/sum;
        var per_avghouseage = avghouseagecorelation/sum;
        var per_avgfueluseage = avgfueluseagecorelation/sum;
        var per_fulljob = fulljobcorelation/sum;
        var per_partjob = partjobcorelation/sum;
        var per_nojob = nojobcorelation/sum;


        // draw Graph by Echarts

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
                data: [povertyrate, underhighschoolpovertyrate, highschoolpovertyrate, collegepovertyrate, bachelorpovertyrate]
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
                data: [povertyrate, malepovertyrate, femalepovertyrate]
            }]
        };
        genderChart.setOption(genderOption);


        var factorsChart = echarts.init(document.getElementById('factors'));

        var factorsOption = {
            title : {
                text: 'biggest factor',
                // subtext: '（Pie Chart）',
                x:'center'
            },
            tooltip : {
                trigger: 'item',
                formatter: "{a} <br/>{b} : {c} ({d}%)"
            },
            // legend: {
            //     orient : 'vertical',
            //     x : 'left',
            //     data:['disability','avghouseage','avgfueluseage','fulljob','partjob','nojob']
            // },
            toolbox: {
                show : true,
                feature : {
                    //mark : {show: true},
                    //dataView : {show: true, readOnly: false},
                    restore : {show: true},
                    //saveAsImage : {show: true}
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
                        {value:per_disability, name:'disability'},
                        {value:per_avghouseage, name:'avghouseage'},
                        {value:per_partjob, name:'partjob'},
                        {value:per_nojob, name:'nojob'},
                        {value:per_avgfueluseage, name:'avgfueluseage'},
                        {value:per_fulljob, name:'fulljob'}]
                }
            ]
        };
        factorsChart.setOption(factorsOption);
    })
})
