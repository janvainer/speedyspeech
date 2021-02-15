
function synt(){
  let audio = $('#audio');
  $('#audioSource').attr('src',"/synt/" + $('#textInput').val());
  audio[0].load();
  audio.removeClass('hidden');

}
