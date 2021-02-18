function getCookie(name)
{
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function generate_graphs()
{
    var stock_select = document.getElementById("id_stock");
    var selected_stock = stock_select.options[stock_select.selectedIndex].value;
    var start_date = document.getElementById("id_start_date");
    var end_date = document.getElementById("id_end_date");
    var interval_select = document.getElementById("id_interval");
    var selected_interval = interval_select.options[interval_select.selectedIndex].value;
    var close_btn = document.getElementById('close_modal');
    var sender = new XMLHttpRequest();
    var payload = JSON.stringify({
    'selected_stock':selected_stock,
    'start_date':start_date.value,
    'end_date':end_date.value,
    'selected_interval':selected_interval})

    sender.open('POST','generate_graphs')
    var csrf = getCookie('csrftoken');
    sender.setRequestHeader('X-CSRFToken', csrf);
    sender.setRequestHeader("Content-Type", "application/JSON");
    sender.onload = function(){
        if(this.readyState == 4)
            {
                var response = JSON.parse(this.responseText);
                if(this.status == 200)
                {
                    var graph_div = document.getElementById('graph_div');

                    graph_div.innerHTML = response.candlestick_graph;
                    toastr.success(response.message,'success')
                    close_btn.click();
                }

                else
                {
                   toastr.error(response.message,'Error');

                }

             }
    }
    sender.send(payload);


}