let imageBlobs = [];
let model;

async function loadBlazeFace() {
    model = await blazeface.load();
    console.log("BlazeFace Model Loaded");
}

async function captureFace(canvas, ctx, video) {
    if (!model) return;

    const predictions = await model.estimateFaces(video, false);
    if (predictions.length > 0) {
        canvas.style.display = "block";
        const face = predictions[0];
        const [x, y, width, height] = face.topLeft.concat(face.bottomRight);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, x, y, width - x, height - y, 0, 0, 200, 200);

        canvas.toBlob(blob => {
            if (blob) {
                imageBlobs.push(blob);
                console.log("Captured Face:", imageBlobs.length);

                // Trigger alert and send images when reaching 5
                if (imageBlobs.length === 5) {
                    console.log("5 face captures reached. Starting recognition...");
                    sendImagesToFlask(imageBlobs);
                    imageBlobs = []; // Clear for next batch
                }
            }
        }, "image/jpeg", 0.9);
    } else {
        canvas.style.display = "none";
    }
}

function sendImagesToFlask(images) {
    const formData = new FormData();
    images.forEach((blob, index) => {
        formData.append("face", blob, `face_${index}.jpg`);
    });

    fetch("http://192.168.100.88:6600/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log("Server Response:", data);
        if (data.message) {
            console.log(`Recognition Successful: ${data.recognized_id}`);
        } else {
            console.log(`Recognition Failed: ${data.error}`);
        }
    })
    .catch(error => {
        console.error("Upload error:", error);
        console.log("Error uploading images.");
    });
}

async function startVideo() {
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.play();

    await loadBlazeFace();

    setInterval(() => {
        captureFace(canvas, ctx, video);
    }, 500);
}

startVideo().catch(console.error);
