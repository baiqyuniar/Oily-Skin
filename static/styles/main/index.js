$(document).ready(function () {
  $('form input').change(function () {
    $('from p').text(this.files.length + ' file(s) selected');
  });
});


