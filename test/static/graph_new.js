var height = 200;
var width = 400;


$(function() {
      $('#multiselect').change(function(){
        graphSelect();
    });
});

function graphSelect() {
    var selection = document.getElementById("multiselect");
    var selectedValue = selection.options[selection.selectedIndex].value;
    if(selectedValue == "PCA_RANDOM_SAMPLING") {
        get_ajax('/random', true, false, false, 'PCA Random Sampling');
    } else if(selectedValue == "PCA_STRATIFIED_SAMPLING") {
        get_ajax('/stratify', false, false, false, 'PCA Stratified Sampling');
    } else if(selectedValue == "MDS_EUCLIDEAN_RANDOM_SAMPLING") {
        get_ajax('/euclidean_random', true, false, false, 'MDS via Euclidean distance on Random Samples');
    } else if(selectedValue == "MDS_CORRELATION_RANDOM_SAMPLING") {
        get_ajax('/correlation_random', true, false, false, 'MDS via Correlation distance on Random Samples');
    }  else if(selectedValue == "PCA_SCREE") {
        // data = [2.38332766e+00, 1.45174543e+00, 1.19655353e+00, 9.81637368e-01, 9.49882852e-01, 7.67465931e-01,
        //     7.23376648e-01, 6.33389219e-01,2.32692613e-16];
        data = [ 1.92686970e-01,  1.54948659e-01,  8.16960543e-02,  6.60939443e-02,
  4.23744916e-02,  3.49265867e-02,  2.61797594e-02,  2.00090705e-02,
  1.62668722e-02,  1.39324203e-02,  9.13821707e-03, 5.71256938e-03,
  5.03592943e-03,  2.76767351e-03, -1.57706866e-18];
        drawScreePlot(data,  'sampling scree plot to find intrinsic dimensioanlity');
    } else if(selectedValue == "PCA_SCREE_ORIGINAL") {
 //        data = [2.26412583e+00, 1.61822451e+00, 1.19488198e+00, 1.08384641e+00,
 // 9.58600053e-01, 8.61413233e-01, 8.51117581e-01, 7.25193763e-01,
 // 4.61258620e-01, 9.97474640e-05];
        data = [1.96011256e-01, 1.14484571e-01, 6.38653645e-02, 4.64705839e-02,
 3.69259502e-02, 2.56092469e-02, 1.83850288e-02, 1.33175819e-02,
 6.22604888e-03, 5.89560115e-03, 2.78080358e-03, 1.89728889e-03,
 6.06109690e-04, 4.53620385e-04, 2.22192888e-19];
        drawScreePlot(data, "original scree plot to find intrinsic dimensioanlity");
    }

    d3.select('#scree').remove();
}

function get_ajax(url, isRandSamples, matrix, isScree, chart_title) {
	$.ajax({
	  type: 'GET',
	  url: url,
      contentType: 'application/json; charset=utf-8',
	  xhrFields: {
		withCredentials: false
	  },
	  headers: {
	  },
	  success: function(result) {

          if(isScree) {
		     drawScreePlot(result, chart_title)
		    }  else {
                drawScatter(result, isRandSamples, chart_title);
                }
	  },
	});
}

function drawScreePlot(eigen_values, chart_title) {

    data = eigen_values;

    d3.select('#chart').remove();

    var margin = {top: 20, right: 20, bottom: 20, left: 20};
    var width = 1200 - margin.left - margin.right;
    var height = 600 - margin.top - margin.bottom;

    var chart_width = 1000;
    var chart_height = 400 + margin.top + margin.bottom;

    var x = d3.scaleLinear().domain([1, data.length + 0.5]).range([0, chart_width - 120]);
    var y = d3.scaleLinear().domain([0, d3.max(data)]).range([height, 0]);

    var xAxis = d3.axisBottom().scale(x);
    var yAxis = d3.axisLeft().scale(y)

    var markX
    var markY
    var color = d3.scaleOrdinal(d3.schemeCategory10);

    var line = d3.line()
        .x(function(d,i) {
            if (i == 3) {
                markX = x(i);
                markY = y(d)
            }
            return x(i);
        })
        .y(function(d) {
            return y(d);
        })

    // Add an SVG element with the desired dimensions and margin.
    var svg = d3.select("pca-scree").append("svg")

    var graph = svg.attr("id", "chart")
          .attr("width", width + margin.left + margin.right)
          .attr("height", height + margin.top + margin.bottom + 10)

    var g = graph.append("g")
          .attr("transform", "translate(50,10)");

    g.append("g") // add xAxis to the bottom
          .attr("class", "x_axis")
          .attr("transform", "translate(110," + height + ")")
          .call(xAxis);

    // add the yAxis to the left
    g.append("g") // add the yAxis to the left
          .attr("class", "y_axis")
          .attr("transform", "translate(108,0)")
          .call(yAxis);

    g.append("text") // add the Y-axix instruction
        .attr("class", "axis_label")
        .attr("text-anchor", "end")
        .attr("transform", "translate(56,0)rotate(-90)")
        .text("Eigen Values");

    g.append("path") // add the line
        .attr("d", line(data))
        .attr("transform", "translate(213,0)")
        .attr("fill", "none")
        .attr("stroke", "green")
        .attr("stroke-width", "4px")

    g.append("circle") // add the point K
              .attr("cx", markX)
              .attr("cy", markY)
              .attr("r", 7)
              .attr("transform", "translate(213,0)")
              .style("fill", "black");

    g.append("text") // put the K to show the circle
    .attr("class", "axis_label")
    .attr("text-anchor", "middle")
    .attr("transform", "translate(410, 280)")
    .text("K");

    g.append("text")
        .attr("x", 500)
        .attr("y", 10)
        .attr("text-anchor", "middle")
        .style("font-size", "px")
        .style("font-weight", "bold")
        .text(chart_title);
}

function drawScatter(two_div_data, isRandSamples, chart_title) {
    d3.select('#chart').remove();
    var margin = {top: 20, right: 20, bottom: 20, left: 20},
    width = 1200 - margin.left - margin.right,
    height = 600 - margin.top - margin.bottom;
    var data = JSON.parse(two_div_data);
    var array = [];
//  To get column names of most weighted attributes/columns
    keyNames = Object.keys(data);
    //get the column of "0" and "1"
    for(var i=0; i< Object.keys(data[0]).length; ++i){
        obj = {}
        obj.x = data[0][i];
        obj.y = data[1][i];
        obj.clusterid = data['clusterid'][i]
        obj.ftr1 = data[keyNames[2]][i]
        obj.ftr2 = data[keyNames[3]][i]
        array.push(obj);
    }
    data = array;

    var xPoint = function(d) { return d.x;},
        x = d3.scaleLinear().range([0, width]),

        xAxis = d3.axisBottom().scale(x);

    var yPoint = function(d) { return d.y;},
        y = d3.scaleLinear().range([height, 0]),

        yAxis = d3.axisLeft().scale(y)

    var color = d3.scaleOrdinal(d3.schemeCategory10);

    var clusterValue
    if(isRandSamples) { // if random samples
        clusterValue = function(d) { return d.clusteridx;}
    } else { //if stratified samples
        clusterValue = function(d) { return d.clusterid;}
    }

    var svg = d3.select("body").append("svg")
        .attr('id', 'chart')
        .attr("height", height + margin.top + margin.bottom)
        .attr("width", width + margin.left + margin.right)

    var g = svg.append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


    x.domain([d3.min(data, xPoint)-1, d3.max(data, xPoint)+1]);
    y.domain([d3.min(data, yPoint)-1, d3.max(data, yPoint)+1]);

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
        .style("font-size", "28px")
        .text(chart_title);

    var spotTool = d3.select("body").append('div').style('position','absolute');

    g.selectAll(".dot")
          .data(data)
          .enter().append("circle")
          .attr("class", "dot")
          .attr("cx", function(d) { return x(xPoint(d));})
          .attr("cy", function(d) { return y(yPoint(d));})
          .attr("r", 3.6)
          .style("fill", function(d) { return color(clusterValue(d));})
          .on("mouseover", function(d) {
              spotTool.transition().style('opacity', .8).style('color','blue')
              spotTool.html(keyNames[2] + " = " + d.ftr1 + ", "+ keyNames[3] +" = " + d.ftr2)
                   .style("top", (d3.event.pageY - 30) + "px")
                   .style("left", (d3.event.pageX + 2) + "px");
          })
          .on("mouseout", function(d) {
              spotTool.transition()
                   .duration(600)
                   .style("opacity", 0);
              spotTool.html('');
          });

}



