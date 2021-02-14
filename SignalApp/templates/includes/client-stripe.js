var execution_mode = document.getElementById('mode-ex');
var key = (execution_mode.value == "DEV" ? "pk_test_Gc68KaHPuajzfgnCbcmGcbgC001gYPnK8C" : "pk_live_VXeFxTdzMvwmWR1IJYQuvWiX00lnSquolS")
var stripe = Stripe(key);
var Payment = {};
  function selected_loaded(sel)
    {
        var load_value = document.getElementById("invoice_price");
        var selected_id = sel.options[sel.selectedIndex].value;
        $.ajax(
            {
                  url: 'load_invoice_amount',
                  data: {'id':selected_id},
                  type: "GET",
                  dataType: 'json',
                  success: function(data){

                                        load_value.value='$ '+data.invoice_amount;
                                        Payment.invoice_amount = data.invoice_amount;
                                        Payment.invoice_id = data.invoice_id;
                                        Payment.client_secret=data.client_secret;
                                        var elements = stripe.elements();
                                        var style = {
                                          base: {
                                            color: "#32325d",
                                          }
                                        };

                                        var card = elements.create("card", { style: style });
                                        card.mount("#card-element");
                                        Payment.card = card;


                                         },
                 error: function(data)
                    {
                        toastr.error(data.responseJSON.error_message);

                    }
             }

        )


    }

Payment.card.on('change', ({error}) => {
  let displayError = document.getElementById('card-errors');
  if (error) {
    displayError.textContent = error.message;
  } else {
    displayError.textContent = '';
  }
});


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


function finish_payment_server()
{
    $("#processing-payment").modal({
                        show: true
                    });
    var sender = new XMLHttpRequest();
    var payload = JSON.stringify({'invoice_id':Payment.invoice_id,
    'payment_object':Payment.payment_intent})
    sender.open('POST','finish_payment')
    var csrf = getCookie('csrftoken');
    sender.setRequestHeader('X-CSRFToken', csrf);
    sender.setRequestHeader("Content-Type", "application/JSON");
    sender.onload = function(){
        if(this.readyState == 4)
            {
                var response = JSON.parse(this.responseText);
                if(this.status == 200)
                {
//                    console.log(response);
//                    toastr.success(response.message,'success')
                    window.location.replace(response.redirect_url);
                    window.location = response.redirect_url;
                }

                else
                {
                   toastr.error(response.error_message,'Error');

                }

             }
    }
    sender.send(payload);

}


function process_payment_api() {
  console.log(Payment);
//  this.preventDefault();
  stripe.confirmCardPayment(Payment.client_secret,
  {
    payment_method: {
      card: Payment.card
    }
  }).then(function(result) {
    if (result.error) {
        toastr.error(result.error.message);

    } else {
      // The payment has been processed!
      if (result.paymentIntent.status === 'succeeded') {
        Payment.payment_intent = result.paymentIntent;
//        console.log('Payment successful confirmed in the status',Payment,result);
        finish_payment_server();
      }
      else{
              toastr.error(paymentIntent.status);
      }
    }
  });

}


//
//function process_payment(){
//  var api_endpoint = "https://api.stripe.com/v1/payment_intents/"+Payment.id+"/confirm";
//  console.log(Payment,api_endpoint);
//  $.ajax(
//            {
//                  url: api_endpoint,
//                  data: {'payment_method': Payment.card},
//                  type: "POST",
//                  dataType: 'json',
//                  headers:{'Authentication':"Bearer "+Payment.key},
//                  success: function(data){
//
//                    if (data.paymentIntent.status === 'succeeded') {
//                    // Show a success message to your customer
//                    // There's a risk of the customer closing the window before callback
//                    // execution. Set up a webhook or plugin to listen for the
//                    // payment_intent.succeeded event that handles any business critical
//                    // post-payment actions.
//                        $ajax(
//                        {
//                              url: redirect_url,
//                              data: {'invoice_id':Payment.invoice_id, 'stripe_confirmation_id':data.paymentIntent.id},
//                              type: "POST",
//                              headers:{'X-CSRFToken':getCookie('csrftoken')},
//                              dataType: 'json',
//                              success: function(data){
//                                                    nil;
//                                                    },
//                              error: function(data){
//                                                    toastr.error(data.error_message);
//
//                                                    }
//
//                         });
//                    }
//
//
//                  },
//                  error:function(data){
//                                        toastr.error(data.error.message);
//
//                                        }
//
//             }
//
//        )
//
//
//}

