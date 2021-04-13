$(document).ready(function () {
    var page_url = window.location.href;
    var page_id = page_url.substring(page_url.lastIndexOf("#") + 1);
    // alert(page_id);

    if (page_id == 'section1') {
        $('html, body').animate({
            scrollTop: $('#scroll-' + page_id).offset().top
        }, 500);
    } else if (page_id == 'section2') {
        $('html, body').animate({
            scrollTop: $('#scroll-' + page_id).offset().top
        }, 500);
    } else if (page_id == 'section3') {
        $('html, body').animate({
            scrollTop: $('#scroll-' + page_id).offset().top
        }, 500);
    } else if (page_id == 'section4') {
        $('html, body').animate({
            scrollTop: $('#scroll-' + page_id).offset().top
        }, 500);
    }

});
