var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function showPicked(input) {
  el("upload-label").innerHTML = input.files[0].name;
  var reader = new FileReader();
  reader.onload = function(e) {
    el("image-picked").src = e.target.result;
    el("image-picked").className = "";
    console.log(el("image-picked"));
  };
  reader.readAsDataURL(input.files[0]);
}

function analyze() {
  var uploadFiles = el("file-input").files;
  console.log(uploadFiles);
  console.log(uploadFiles[0]);
  if (uploadFiles.length !== 1) alert("Please select a file to analyze!");

  el("analyze-button").innerHTML = "Analyzing..."

  var fileData = new FormData();
  fileData.append("file", uploadFiles[0]);

  var xhr = new XMLHttpRequest();
  //var loc = window.location;
  xhr.open("POST", 'https://link-to-your--cloud-function.cloudfunctions.net/name-of-your-cloud-function', true);
  xhr.send(fileData);

  xhr.onerror = function() {
    alert(xhr.responseText);
  };
  xhr.onload = function(e) {
    if (this.readyState === 4) {
      var response = JSON.parse(e.target.responseText);
      // console.log(response);
      // console.log(response.Confidence);
      el("result-label").innerHTML = `Result = ${response["Predicted Class"]}; Condifence = ${Math.round(response.Confidence*100, 2)}%`;
    }
    el("analyze-button").innerHTML = "Analyze";
  };
}