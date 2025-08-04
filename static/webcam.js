const video = document.getElementById("webcam");

if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
      video.srcObject = stream;
    })
    .catch(function (err) {
      console.error("Webcam error:", err);
      alert("Webcam error: " + err.message);
    });
} else {
  alert("Your browser does not support webcam access.");
}

