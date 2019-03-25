$(document).ready(function(){
    $("#nav").load("/menu.html");
    $("a").find(`[href="` + document.location.pathname.match(/[^\/]+$/)[0] + `"]`).attr('class', 'w3-theme');
  });