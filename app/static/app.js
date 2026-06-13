const form = document.querySelector("#ocr-form");
const fileInput = document.querySelector("#file-input");
const tabButtons = document.querySelectorAll("[data-input-mode]");
const inputPanels = document.querySelectorAll("[data-panel]");
const startCameraButton = document.querySelector("#start-camera");
const captureButton = document.querySelector("#capture-photo");
const mirrorToggle = document.querySelector("#mirror-camera");
const video = document.querySelector("#camera-video");
const canvas = document.querySelector("#capture-canvas");
const capturedPreview = document.querySelector("#captured-preview");
const cameraMessage = document.querySelector("#camera-message");

let cameraStream = null;
let activeMode = "upload";

function setMessage(message) {
  if (cameraMessage) {
    cameraMessage.textContent = message || "";
  }
}

function setMode(mode) {
  activeMode = mode;
  tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.inputMode === mode);
  });
  inputPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === mode);
  });

  if (mode === "upload") {
    setMessage("");
  }
}

function setFileFromBlob(blob) {
  const file = new File([blob], `camera-capture-${Date.now()}.jpg`, { type: "image/jpeg" });
  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(file);
  fileInput.files = dataTransfer.files;
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setMessage("Camera is not available in this browser.");
    return;
  }

  try {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
    }

    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
      audio: false,
    });

    video.srcObject = cameraStream;
    await video.play();
    video.classList.add("active");
    captureButton.disabled = false;
    setMessage("Camera ready. Align the prescription and capture.");
  } catch (error) {
    setMessage(`Camera failed: ${error.message}`);
  }
}

function updateMirror() {
  video.classList.toggle("mirrored", mirrorToggle.checked);
  capturedPreview.classList.toggle("mirrored", mirrorToggle.checked);
}

function capturePhoto() {
  if (!video.videoWidth || !video.videoHeight) {
    setMessage("Camera is not ready yet.");
    return;
  }

  const context = canvas.getContext("2d");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  if (mirrorToggle.checked) {
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
  }

  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  context.setTransform(1, 0, 0, 1, 0, 0);

  canvas.toBlob((blob) => {
    if (!blob) {
      setMessage("Could not capture image.");
      return;
    }

    setFileFromBlob(blob);
    capturedPreview.src = URL.createObjectURL(blob);
    capturedPreview.classList.add("active");
    setMessage("Captured image ready. Run OCR when you are ready.");
  }, "image/jpeg", 0.95);
}

tabButtons.forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.inputMode));
});

if (startCameraButton) {
  startCameraButton.addEventListener("click", startCamera);
}

if (captureButton) {
  captureButton.addEventListener("click", capturePhoto);
}

if (mirrorToggle) {
  mirrorToggle.addEventListener("change", updateMirror);
}


if (fileInput) {
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length && activeMode === "upload") {
      capturedPreview.removeAttribute("src");
      capturedPreview.classList.remove("active");
    }
  });
}

if (form) {
  form.addEventListener("submit", (event) => {
    if (!fileInput.files.length) {
      event.preventDefault();
      if (activeMode === "camera") {
        setMessage("Capture a camera image before running OCR.");
      } else {
        alert("Choose an image before running OCR.");
      }
    }
  });
}

window.addEventListener("beforeunload", () => {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
  }
});
