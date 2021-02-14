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

function delete_record(option, unique_id){

    var sender = new XMLHttpRequest();
    var payload = JSON.stringify({'option':option,
    "unique_id":unique_id
    })

    sender.open('POST','/admintr/auth/delete_record/'+unique_id+'/'+option)
    var csrf = getCookie('csrftoken');
    sender.setRequestHeader('X-CSRFToken', csrf);
    sender.setRequestHeader("Content-Type", "application/JSON");
    sender.onload = function(){
        if(this.readyState == 4)
            {

                var response = JSON.parse(this.responseText);
                if(this.status == 200)
                {
//                    toastr.success("The record was successfully deleted",'Success')
                    console.log(response.redirect_url);
                    window.location.href = response.redirect_url
                }

                else
                {
                   toastr.error(response.error,'Error');

                }

             }
    }
    sender.send(payload);
}


function get_page(model_name, page_number, per_page, detail_id=null)
{
    var sender = new XMLHttpRequest();
    var payload = JSON.stringify({'model_name':model_name,
    'page_number':page_number,
    'per_page':per_page,
    'detail_id': detail_id
    })

    sender.open('POST','/paginate')
    list_name = 'paginate_list'+'_'+model_name;
    var csrf = getCookie('csrftoken');
    sender.setRequestHeader('X-CSRFToken', csrf);
    sender.setRequestHeader("Content-Type", "application/JSON");

    sender.onload = function(){
        if(this.readyState == 4)
            {

                var response = JSON.parse(this.responseText);
                if(this.status == 200)
                {
//                    console.log(list_name);
                    page = document.getElementById(list_name);
                    page.innerHTML=response.page;

                }

                else
                {
                   toastr.error(response.error,'Error');

                }

             }
    }
    sender.send(payload);

}


function pagination_search(model_name, detail_id)
{
/*
this method makes the restful search
properly based on the given values.

returns a listing value for the rendering of the pagination.
*/

var query = document.getElementById('search_'+model_name);
var val = query.value;
 var sender = new XMLHttpRequest();
    if (!val){

    return get_page(model_name, 1, 10, detail_id=detail_id);
    }
    var payload = JSON.stringify({'model_name':model_name,
    'search_query':val,
    'detail_id': detail_id
    })

    sender.open('POST','/search')
    list_name = 'paginate_list'+'_'+model_name;
    var csrf = getCookie('csrftoken');
    sender.setRequestHeader('X-CSRFToken', csrf);
    sender.setRequestHeader("Content-Type", "application/JSON");

    sender.onload = function(){
        if(this.readyState == 4)
            {
                console.log(list_name);
                var response = JSON.parse(this.responseText);
                console.log(query.value, val, model_name, detail_id);
                if(this.status == 200)
                {
                    if(response.empty == true){
                        toastr.warning("There wasn't any records matching the value given: "+val, 'Warning')

                    }
                    else{
                        page = document.getElementById(list_name);
                        page.innerHTML=response.page;
                    }


                }

                else
                {
                   toastr.error(response.error,'Error');

                }

             }
    }
    sender.send(payload);


}

function load_scs(){
val = document.getElementById('search_store');
store_selector = document.getElementById('id_store');
var sender = new XMLHttpRequest();


if (val){
    var payload = JSON.stringify({
    'store_name':val.value
    })

    sender.open('POST','/admintr/find_store')
    var csrf = getCookie('csrftoken');
    sender.setRequestHeader('X-CSRFToken', csrf);
    sender.setRequestHeader("Content-Type", "application/JSON");

    sender.onload = function(){
        if(this.readyState == 4)
            {
                var response = JSON.parse(this.responseText);
                if(this.status == 200)
                {
                        id = response.store_id;
                       for(i=0;i<store_selector.options.length;i++)

                       {

                            if (parseInt(store_selector.options.item(i).value) == id)
                            {
                                store_selector.options.selectedIndex = i;
                                break;

                            }
                       }
                }

                else
                {
                   toastr.error(response.error,'Error');

                }

             }
    }
    sender.send(payload);

}
else{
toastr.error("You need to provided a valid name for the store to be looked for","Error");
}

/*
if(this.status == 200)
            {


                    page = document.getElementById('id_store');
                    i = 0
                    [].forEach.call(document.querySelectorAll('#id_store')  , function(elm){

                    console.log(elm.value,response.store_id);
                    if (elm.value == response.store_id){
                    page.selectedIndex = i;
                    break;
                    }

                    i++
                    })



            }



*/

}


/*  CHAT METHODS */
function select_chat(chat_id)
{
    xhr = new XMLHttpRequest;
    xhr.open('POST','/chat/change_selected_chat')
    data = JSON.stringify({"chat_id":chat_id})
    var csrf = getCookie('csrftoken');
    xhr.setRequestHeader('X-CSRFToken', csrf);
    xhr.setRequestHeader("Content-Type", "application/JSON");
    xhr.onload = function(){
        if(this.readyState == 4)
            {
                var response = JSON.parse(this.responseText);
                if(this.status == 200)
                {
                        page = document.getElementById("chat-box");
                        page.innerHTML=response.chat_page;
                        // here we load the messages there is.
                }

                else
                {
                   toastr.error(response.error,'Error');

                }

             }
    }
    xhr.send(data);


}

function check_chat_selected()
{
    sender = new XMLHttpRequest;
    sender.open('GET','/chat/check_selected');
    sender.onload = function(){
    if (this.readyState == 4)
        {
            console.log("RUNNING BOI");
            var response = JSON.parse(this.responseText);
            if(this.status == 200)
            {
                    console.log("RUNNING BOI",response);
                    if (response.chat_selected)
                    {
                       change_selected_chat(response.selected_chat);
                    }
            }

            else
            {
               toastr.error(response.error,'Error');

            }
        }

    }
    sender.send();


}