var height = 200;
var width = 400;


$(document).ready(function () {
    get_ajax('/random', true, false, 'pca', 'PCA Random Sampling');
    get_ajax('/euclidean_random', true, false, 'mds', 'MDS via Euclidean distance on Random Samples');
    get_ajax('/correlation_random', true, false, 'mdscorr', 'MDS via Correlation distance on Random Samples');
    data = [1.92686970e-01, 1.54948659e-01, 8.16960543e-02, 6.60939443e-02,
        4.23744916e-02, 3.49265867e-02, 2.61797594e-02, 2.00090705e-02,
        1.62668722e-02, 1.39324203e-02, 9.13821707e-03, 5.71256938e-03,
        5.03592943e-03, 2.76767351e-03, -1.57706866e-18];
    drawScreePlot(data, 'sampling scree plot');
    data = [1.96011256e-01, 1.14484571e-01, 6.38653645e-02, 4.64705839e-02,
        3.69259502e-02, 2.56092469e-02, 1.83850288e-02, 1.33175819e-02,
        6.22604888e-03, 5.89560115e-03, 2.78080358e-03, 1.89728889e-03,
        6.06109690e-04, 4.53620385e-04, 2.22192888e-19];
    drawScreePlotOverall(data, "original scree plot");
    buildPie()

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

            if (isScree == 'pca') {
                drawScatterPCA(result, isRandSamples, chart_title);
            } else if (isScree == 'mds') {
                drawScatterMDS(result, isRandSamples, chart_title);
            } else {
                drawScatterMDSCorr(result, isRandSamples, chart_title);

            }
        },
    });
}

function drawScreePlot(eigen_values, chart_title) {
    data = eigen_values;


    var margin = {top: 10, right: 10, bottom: 10, left: 10},
        width = 400 - margin.left - margin.right,
        height = 200 - margin.top - margin.bottom;

    var chart_width = width;
    var chart_height = 400 + margin.top + margin.bottom;

    var x = d3.scaleLinear().domain([1, data.length + 0.5]).range([0, chart_width - 120]);
    var y = d3.scaleLinear().domain([0, d3.max(data)]).range([height, 0]);

    var xAxis = d3.axisBottom().scale(x);
    var yAxis = d3.axisLeft().scale(y)

    var markX
    var markY
    var color = d3.scaleOrdinal(d3.schemeCategory10);

    var line = d3.line()
        .x(function (d, i) {
            if (i == 3) {
                markX = x(i);
                markY = y(d)
            }
            return x(i);
        })
        .y(function (d) {
            return y(d);
        })

    // Add an SVG element with the desired dimensions and margin.
    var svg = d3.select("#pca-scree").append("svg")

    var graph = svg.attr("id", "chart")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom + 10)

    var g = graph.append("g")
        .attr("transform", "translate(50,10)");

    g.append("g") // add xAxis to the bottom
        .attr("class", "x_axis")
        .attr("transform", "translate(10," + height + ")")
        .call(xAxis);

    // add the yAxis to the left
    g.append("g") // add the yAxis to the left
        .attr("class", "y_axis")
        .attr("transform", "translate(8,0)")
        .call(yAxis);

    g.append("text") // add the Y-axix instruction
        .attr("class", "axis_label")
        .attr("text-anchor", "end")
        .attr("transform", "translate(20,0)rotate(-90)")
        .text("Eigen Values");

    g.append("path") // add the line
        .attr("d", line(data))
        .attr("transform", "translate(33,0)")
        .attr("fill", "none")
        .attr("stroke", "green")
        .attr("stroke-width", "4px")

    g.append("circle") // add the point K
        .attr("cx", markX)
        .attr("cy", markY)
        .attr("r", 7)
        .attr("transform", "translate(33,0)")
        .style("fill", "black");

    g.append("text") // put the K to show the circle
        .attr("class", "axis_label")
        .attr("text-anchor", "middle")
        .attr("transform", "translate(110, 280)")
        .text("K");

    g.append("text")
        .attr("x", 150)
        .attr("y", 10)
        .attr("text-anchor", "middle")
        .style("font-size", "14px")
        .style("font-weight", "bold")
        .text(chart_title);
}


function drawScreePlotOverall(eigen_values, chart_title) {
    data = eigen_values;


    var margin = {top: 10, right: 10, bottom: 10, left: 10},
        width = 400 - margin.left - margin.right,
        height = 200 - margin.top - margin.bottom;

    var chart_width = width;
    var chart_height = 400 + margin.top + margin.bottom;

    var x = d3.scaleLinear().domain([1, data.length + 0.5]).range([0, chart_width - 120]);
    var y = d3.scaleLinear().domain([0, d3.max(data)]).range([height, 0]);

    var xAxis = d3.axisBottom().scale(x);
    var yAxis = d3.axisLeft().scale(y)

    var markX
    var markY
    var color = d3.scaleOrdinal(d3.schemeCategory10);

    var line = d3.line()
        .x(function (d, i) {
            if (i == 3) {
                markX = x(i);
                markY = y(d)
            }
            return x(i);
        })
        .y(function (d) {
            return y(d);
        })

    // Add an SVG element with the desired dimensions and margin.
    var svg = d3.select("#pca-scree-overall").append("svg")

    var graph = svg.attr("id", "chart")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom + 10)

    var g = graph.append("g")
        .attr("transform", "translate(50,10)");

    g.append("g") // add xAxis to the bottom
        .attr("class", "x_axis")
        .attr("transform", "translate(10," + height + ")")
        .call(xAxis);

    // add the yAxis to the left
    g.append("g") // add the yAxis to the left
        .attr("class", "y_axis")
        .attr("transform", "translate(8,0)")
        .call(yAxis);

    g.append("text") // add the Y-axix instruction
        .attr("class", "axis_label")
        .attr("text-anchor", "end")
        .attr("transform", "translate(20,0)rotate(-90)")
        .text("Eigen Values");

    g.append("path") // add the line
        .attr("d", line(data))
        .attr("transform", "translate(33,0)")
        .attr("fill", "none")
        .attr("stroke", "green")
        .attr("stroke-width", "4px")

    g.append("circle") // add the point K
        .attr("cx", markX)
        .attr("cy", markY)
        .attr("r", 7)
        .attr("transform", "translate(33,0)")
        .style("fill", "black");

    g.append("text") // put the K to show the circle
        .attr("class", "axis_label")
        .attr("text-anchor", "middle")
        .attr("transform", "translate(110, 280)")
        .text("K");

    g.append("text")
        .attr("x", 150)
        .attr("y", 10)
        .attr("text-anchor", "middle")
        .style("font-size", "14px")
        .style("font-weight", "bold")
        .text(chart_title);
}


function drawScatterPCA(two_div_data, isRandSamples, chart_title) {
    var margin = {top: 20, right: 20, bottom: 20, left: 20},
        width = 400 - margin.left - margin.right,
        height = 200 - margin.top - margin.bottom;
    var data = JSON.parse(two_div_data);
    var array = [];
//  To get column names of most weighted attributes/columns
    keyNames = Object.keys(data);
    //get the column of "0" and "1"
    for (var i = 0; i < Object.keys(data[0]).length; ++i) {
        obj = {}
        obj.x = data[0][i];
        obj.y = data[1][i];
        obj.clusterid = data['clusterid'][i]
        obj.ftr1 = data[keyNames[2]][i]
        obj.ftr2 = data[keyNames[3]][i]
        array.push(obj);
    }
    data = array;

    var xPoint = function (d) {
            return d.x;
        },
        x = d3.scaleLinear().range([0, width]),

        xAxis = d3.axisBottom().scale(x);

    var yPoint = function (d) {
            return d.y;
        },
        y = d3.scaleLinear().range([height, 0]),

        yAxis = d3.axisLeft().scale(y)

    var color = d3.scaleOrdinal(d3.schemeCategory10);

    var clusterValue
    if (isRandSamples) { // if random samples
        clusterValue = function (d) {
            return d.clusteridx;
        }
    } else { //if stratified samples
        clusterValue = function (d) {
            return d.clusterid;
        }
    }

    var svg = d3.select("#pca").append("svg")
        .attr('id', 'chart')
        .attr("height", height + margin.top + margin.bottom)
        .attr("width", width + margin.left + margin.right)

    var g = svg.append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


    x.domain([d3.min(data, xPoint) - 1, d3.max(data, xPoint) + 1]);
    y.domain([d3.min(data, yPoint) - 1, d3.max(data, yPoint) + 1]);

    g.append("g") //set yAxis
        .attr("class", "y_axis")
        .call(yAxis);

    g.append("text")
        .attr("class", "label")
        .attr("y", 7)
        .attr("transform", "rotate(-90)")
        .attr("dy", ".71em")
        .text("Component 2")
        .style("text-anchor", "end");

    g.append("g") // set xAxis
        .attr("transform", "translate(0," + height + ")")
        .attr("class", "x_axis")
        .call(xAxis);

    g.append("text")
        .attr("class", "label")
        .attr("y", -6)
        .attr("x", width)
        .attr("transform", "translate(0," + height + ")")
        .text("Component 1")
        .style("text-anchor", "end");

    g.append("text")
        .attr("x", (width / 2.1))
        .attr("y", "20")
        .attr("text-anchor", "middle")
        .style("font-weight", "bold")
        .style("font-size", "14px")
        .text(chart_title);

    var spotTool = d3.select("#pca").append('div').style('position', 'absolute');

    g.selectAll(".dot")
        .data(data)
        .enter().append("circle")
        .attr("class", "dot")
        .attr("cx", function (d) {
            return x(xPoint(d));
        })
        .attr("cy", function (d) {
            return y(yPoint(d));
        })
        .attr("r", 3.6)
        .style("fill", function (d) {
            return color(clusterValue(d));
        })
        .on("mouseover", function (d) {
            spotTool.transition().style('opacity', .8).style('color', 'blue')
            spotTool.html(keyNames[2] + " = " + d.ftr1 + ", " + keyNames[3] + " = " + d.ftr2)
                .style("top", (d3.event.pageY - 30) + "px")
                .style("left", (d3.event.pageX + 2) + "px");
        })
        .on("mouseout", function (d) {
            spotTool.transition()
                .duration(600)
                .style("opacity", 0);
            spotTool.html('');
        });

}

function drawScatterMDS(two_div_data, isRandSamples, chart_title) {
    var margin = {top: 20, right: 20, bottom: 20, left: 20},
        width = 400 - margin.left - margin.right,
        height = 200 - margin.top - margin.bottom;
    var data = JSON.parse(two_div_data);
    var array = [];
//  To get column names of most weighted attributes/columns
    keyNames = Object.keys(data);
    //get the column of "0" and "1"
    for (var i = 0; i < Object.keys(data[0]).length; ++i) {
        obj = {}
        obj.x = data[0][i];
        obj.y = data[1][i];
        obj.clusterid = data['clusterid'][i]
        obj.ftr1 = data[keyNames[2]][i]
        obj.ftr2 = data[keyNames[3]][i]
        array.push(obj);
    }
    data = array;

    var xPoint = function (d) {
            return d.x;
        },
        x = d3.scaleLinear().range([0, width]),

        xAxis = d3.axisBottom().scale(x);

    var yPoint = function (d) {
            return d.y;
        },
        y = d3.scaleLinear().range([height, 0]),

        yAxis = d3.axisLeft().scale(y)

    var color = d3.scaleOrdinal(d3.schemeCategory10);

    var clusterValue
    if (isRandSamples) { // if random samples
        clusterValue = function (d) {
            return d.clusteridx;
        }
    } else { //if stratified samples
        clusterValue = function (d) {
            return d.clusterid;
        }
    }

    var svg = d3.select("#mds").append("svg")
        .attr('id', 'chart')
        .attr("height", height + margin.top + margin.bottom)
        .attr("width", width + margin.left + margin.right)

    var g = svg.append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


    x.domain([d3.min(data, xPoint) - 1, d3.max(data, xPoint) + 1]);
    y.domain([d3.min(data, yPoint) - 1, d3.max(data, yPoint) + 1]);

    g.append("g") //set yAxis
        .attr("class", "y_axis")
        .call(yAxis);

    g.append("text")
        .attr("class", "label")
        .attr("y", 7)
        .attr("transform", "rotate(-90)")
        .attr("dy", ".71em")
        .text("Component 2")
        .style("text-anchor", "end");

    g.append("g") // set xAxis
        .attr("transform", "translate(0," + height + ")")
        .attr("class", "x_axis")
        .call(xAxis);

    g.append("text")
        .attr("class", "label")
        .attr("y", -6)
        .attr("x", width)
        .attr("transform", "translate(0," + height + ")")
        .text("Component 1")
        .style("text-anchor", "end");

    g.append("text")
        .attr("x", (width / 2.1))
        .attr("y", "20")
        .attr("text-anchor", "middle")
        .style("font-weight", "bold")
        .style("font-size", "14px")
        .text(chart_title);

    var spotTool = d3.select("body").append('div').style('position', 'absolute');

    g.selectAll(".dot")
        .data(data)
        .enter().append("circle")
        .attr("class", "dot")
        .attr("cx", function (d) {
            return x(xPoint(d));
        })
        .attr("cy", function (d) {
            return y(yPoint(d));
        })
        .attr("r", 3.6)
        .style("fill", function (d) {
            return color(clusterValue(d));
        })
        .on("mouseover", function (d) {
            spotTool.transition().style('opacity', .8).style('color', 'blue')
            spotTool.html(keyNames[2] + " = " + d.ftr1 + ", " + keyNames[3] + " = " + d.ftr2)
                .style("top", (d3.event.pageY - 30) + "px")
                .style("left", (d3.event.pageX + 2) + "px");
        })
        .on("mouseout", function (d) {
            spotTool.transition()
                .duration(600)
                .style("opacity", 0);
            spotTool.html('');
        });

}


function drawScatterMDSCorr(two_div_data, isRandSamples, chart_title) {
    var margin = {top: 20, right: 20, bottom: 20, left: 20},
        width = 400 - margin.left - margin.right,
        height = 200 - margin.top - margin.bottom;
    var data = JSON.parse(two_div_data);
    var array = [];
//  To get column names of most weighted attributes/columns
    keyNames = Object.keys(data);
    //get the column of "0" and "1"
    for (var i = 0; i < Object.keys(data[0]).length; ++i) {
        obj = {}
        obj.x = data[0][i];
        obj.y = data[1][i];
        obj.clusterid = data['clusterid'][i]
        obj.ftr1 = data[keyNames[2]][i]
        obj.ftr2 = data[keyNames[3]][i]
        array.push(obj);
    }
    data = array;

    var xPoint = function (d) {
            return d.x;
        },
        x = d3.scaleLinear().range([0, width]),

        xAxis = d3.axisBottom().scale(x);

    var yPoint = function (d) {
            return d.y;
        },
        y = d3.scaleLinear().range([height, 0]),

        yAxis = d3.axisLeft().scale(y)

    var color = d3.scaleOrdinal(d3.schemeCategory10);

    var clusterValue
    if (isRandSamples) { // if random samples
        clusterValue = function (d) {
            return d.clusteridx;
        }
    } else { //if stratified samples
        clusterValue = function (d) {
            return d.clusterid;
        }
    }

    var svg = d3.select("#mdscorr").append("svg")
        .attr('id', 'chart')
        .attr("height", height + margin.top + margin.bottom)
        .attr("width", width + margin.left + margin.right)

    var g = svg.append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


    x.domain([d3.min(data, xPoint) - 1, d3.max(data, xPoint) + 1]);
    y.domain([d3.min(data, yPoint) - 1, d3.max(data, yPoint) + 1]);

    g.append("g") //set yAxis
        .attr("class", "y_axis")
        .call(yAxis);

    g.append("text")
        .attr("class", "label")
        .attr("y", 7)
        .attr("transform", "rotate(-90)")
        .attr("dy", ".71em")
        .text("Component 2")
        .style("text-anchor", "end");

    g.append("g") // set xAxis
        .attr("transform", "translate(0," + height + ")")
        .attr("class", "x_axis")
        .call(xAxis);

    g.append("text")
        .attr("class", "label")
        .attr("y", -6)
        .attr("x", width)
        .attr("transform", "translate(0," + height + ")")
        .text("Component 1")
        .style("text-anchor", "end");

    g.append("text")
        .attr("x", (width / 2.1))
        .attr("y", "20")
        .attr("text-anchor", "middle")
        .style("font-weight", "bold")
        .style("font-size", "14px")
        .text(chart_title);

    var spotTool = d3.select("body").append('div').style('position', 'absolute');

    g.selectAll(".dot")
        .data(data)
        .enter().append("circle")
        .attr("class", "dot")
        .attr("cx", function (d) {
            return x(xPoint(d));
        })
        .attr("cy", function (d) {
            return y(yPoint(d));
        })
        .attr("r", 3.6)
        .style("fill", function (d) {
            return color(clusterValue(d));
        })
        .on("mouseover", function (d) {
            spotTool.transition().style('opacity', .8).style('color', 'blue')
            spotTool.html(keyNames[2] + " = " + d.ftr1 + ", " + keyNames[3] + " = " + d.ftr2)
                .style("top", (d3.event.pageY - 30) + "px")
                .style("left", (d3.event.pageX + 2) + "px");
        })
        .on("mouseout", function (d) {
            spotTool.transition()
                .duration(600)
                .style("opacity", 0);
            spotTool.html('');
        });

}

function buildPie() {

    var factorsChart = echarts.init(document.getElementById('maincorr'));

    var bedrooms = 0.01;
    var bathrooms = 0.02;
    var sqft_living = 0.04;
    var sqft_lot = 0.08;
    var floors = 0.16;
    var waterfront = 0.32;
    var condition = 0.16;
    var grade = 0.08;
    var sqft_basement = 0.04;
    var yr_built = 0.09;
    var factorsOption = {
        title: {
            text: 'overall biggest factor',
            x: 'center'
        },
        tooltip: {
            trigger: 'item',
            formatter: "{a} <br/>{b} : {c} ({d}%)"
        },
        toolbox: {
            show: true,
            feature: {
                restore: {show: true},
            }
        },
        calculable: false,
        series: [
            {
                name: '',
                type: 'pie',
                radius: '55%',
                center: ['50%', '60%'],
                data: [
                    {value: bedrooms, name: 'bedrooms'},
                    {value: bathrooms, name: 'bathrooms'},
                    {value: sqft_living, name: 'sqft_living'},
                    {value: sqft_lot, name: 'sqft_lot'},
                    {value: floors, name: 'floors'},
                    {value: waterfront, name: 'waterfront'},
                    {value: condition, name: 'condition'},
                    {value: grade, name: 'grade'},
                    {value: sqft_basement, name: 'sqft_basement'},
                    {value: yr_built, name: 'yr_built'}]
            }
        ]
    };
    factorsChart.setOption(factorsOption);
}


